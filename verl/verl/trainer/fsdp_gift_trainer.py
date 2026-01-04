# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import re
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn, optim
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    fsdp2_clip_grad_norm_
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup, get_constant_schedule_with_warmup
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outpus_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


class FSDPSFTWithKLTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")

        # Load GIFT hyperparameters
        # beta: inverse temperature gain for target token (default: 20.0)
        # smooth_lambda: smoothing coefficient to prevent numerical instability (default: 0.0)
        self.beta = getattr(self.config.trainer, "beta", 20.0)
        self.smooth_lambda = getattr(self.config.trainer, "smooth_lambda", 0.0)
        if self.device_mesh.get_rank() == 0:
            print(f"GIFT Hyperparameters: beta={self.beta}, smooth_lambda={self.smooth_lambda}")

        self._build_dataloader(train_dataset, val_dataset)
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get("lora_rank", 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(self.config.model.target_modules),
                    "bias": "none",
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                             cast_forward_inputs=True)

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "constant":
            self.lr_scheduler = get_constant_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=num_warmup_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

        # === Load Reference Model ===
        if self.device_mesh.get_rank() == 0:
            print(f"Loading Reference Model")
        
        log_gpu_memory_usage("Before ref model allocation", logger=logger)
        
        # Load ref model with same architecture
        with init_context():
            self.ref_model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
                trust_remote_code=trust_remote_code,
            )
            
            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=self.ref_model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)
        
        # Freeze ref model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Wrap with FSDP (force CPU offload for ref model to save GPU memory)
        ref_cpu_offload = CPUOffload(offload_params=True)  # Always offload ref model params
        
        if fsdp_strategy == "fsdp":
            self.ref_fsdp_model = FSDP(
                self.ref_model,
                cpu_offload=ref_cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            # Force CPU offload for ref model
            ref_offload_policy = CPUOffloadPolicy(offload_params=True) if CPUOffloadPolicy else None
            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": ref_offload_policy,
                "reshard_after_forward": True,
            }
            full_state = self.ref_model.state_dict()
            apply_fsdp2(self.ref_model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.ref_model, full_state, self.device_mesh, ref_offload_policy)
            self.ref_fsdp_model = self.ref_model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")
        
        log_gpu_memory_usage("After ref model allocation and FSDP wrapping", logger=logger)

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """
        GIFT: Gibbs Initialization with Finite Temperature
        
        Implements Algorithm 1 from the paper. During training, constructs advantage-adjusted
        target logits by adding inverse temperature gain β to the ground-truth token positions
        over the base model's log-probabilities.
        
        Training objective (Eq. in Algorithm 1):
            L(θ) = -1/B Σ_i Σ_t Σ_v π*_sft(v|x,y*_<t) log π_θ(v|x,y*_<t)
        
        where the target distribution π*_sft is constructed from advantage-adjusted logits:
            ẑ_{t,k} = log p_ref(k|x,y*_<t) + β · 𝟙[k = y*_t]
        
        Note: smooth_lambda is an engineering trick for numerical stability (mentioned in paper
        but not shown in the algorithm pseudocode).
        """
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Inputs
        input_ids = batch["input_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, :-1].reshape(-1).to(self.device_name)
        vocab_size = self.model.config.vocab_size
        
        # GIFT hyperparameters
        beta = self.beta  # Inverse temperature gain (β in Algorithm 1)
        smooth_lambda = self.smooth_lambda  # Smoothing coefficient (engineering trick for stability)

        # Reference model (π_base in Algorithm 1)
        ref_model = getattr(self, "fsdp_ref_model", getattr(self, "ref_fsdp_model", None))
        
        # Standard cross-entropy loss function (for validation)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # =======================================================
                # Branch A: Standard FSDP (No Sequence Parallel)
                # =======================================================
                
                # Step 1: Forward pass with trainable model (π_θ in Algorithm 1)
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(input_ids=input_ids, attention_mask=batch["attention_mask"], use_cache=False)
                student_logits = output.logits[..., :-1, :].contiguous().view(-1, vocab_size)
                labels_flat = labels.contiguous().view(-1).to(student_logits.device)

                if do_backward and ref_model is not None:
                    # === Training Phase: Construct GIFT target distribution ===
                    with torch.no_grad():
                        # Step 2: Forward pass with base model to get logits z_base (Algorithm 1, Line 6)
                        ref_output = ref_model(input_ids=input_ids, attention_mask=batch["attention_mask"], use_cache=False)
                        ref_logits = ref_output.logits[..., :-1, :].contiguous().view(-1, vocab_size)
                        
                        # Step 3: Compute log-probabilities log p_ref(v|x,y*_<t) (Algorithm 1, Line 7)
                        # Apply smoothing for numerical stability (engineering trick)
                        ref_probs = F.softmax(ref_logits, dim=-1)
                        mixed_ref_probs = (1 - smooth_lambda) * ref_probs + smooth_lambda * (1.0 / vocab_size)
                        log_p_ref = torch.log(mixed_ref_probs + 1e-10)
                        
                        # Step 4: Construct advantage-adjusted target logits ẑ (Algorithm 1, Line 8-9)
                        # ẑ_{t,k} = log p_ref(k|x,y*_<t) + β if k=y*_t, else log p_ref(k|x,y*_<t)
                        target_logits = log_p_ref.clone()
                        gain_tensor = torch.tensor(beta, device=target_logits.device, dtype=target_logits.dtype)
                        gain_tensor = gain_tensor.expand(labels_flat.size(0), 1)
                        target_logits.scatter_add_(1, labels_flat.unsqueeze(1), gain_tensor)
                        
                        # Step 5: Compute target distribution π*_sft (Algorithm 1, Line 10)
                        target_probs = F.softmax(target_logits, dim=-1)

                    # Step 6: Compute cross-entropy loss (Algorithm 1, Line 11)
                    # L(θ) = -Σ_v π*_sft(v|x,y*_<t) log π_θ(v|x,y*_<t)
                    student_log_probs = F.log_softmax(student_logits, dim=-1)
                    loss_per_token = -torch.sum(target_probs * student_log_probs, dim=-1)
                    loss_final = loss_per_token * loss_mask
                else:
                    # Validation phase: standard cross-entropy loss
                    ce_loss = loss_fct(student_logits, labels_flat)
                    loss_final = ce_loss * loss_mask

            else:
                # =======================================================
                # Branch B: Ulysses Sequence Parallel (SP)
                # =======================================================
                batch_size, seqlen = input_ids.shape
                # Unpad and slice inputs for sequence parallelism
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), batch["attention_mask"])
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
                position_ids = batch["position_ids"]
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
                
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                # Step 1: Forward pass with trainable model (local shard)
                output = self.fsdp_model(input_ids=input_ids_rmpad_sliced, attention_mask=None, position_ids=position_ids_rmpad_padded, use_cache=False)
                student_logits_local = output.logits.squeeze(0)
                labels_local = input_ids_rmpad_rolled.to(student_logits_local.device)

                if do_backward and ref_model is not None:
                    # === Training Phase: Construct GIFT target distribution (local shard) ===
                    with torch.no_grad():
                        # Step 2-3: Forward base model and compute log p_ref
                        ref_output = ref_model(input_ids=input_ids_rmpad_sliced, attention_mask=None, position_ids=position_ids_rmpad_padded, use_cache=False)
                        ref_logits_local = ref_output.logits.squeeze(0)
                        
                        # Apply smoothing for numerical stability
                        ref_probs_local = F.softmax(ref_logits_local, dim=-1)
                        mixed_ref_probs_local = (1 - smooth_lambda) * ref_probs_local + smooth_lambda * (1.0 / vocab_size)
                        log_p_ref_local = torch.log(mixed_ref_probs_local + 1e-10)
                        
                        # Step 4-5: Construct target logits ẑ and target distribution π*_sft
                        target_logits_local = log_p_ref_local.clone()
                        gain_tensor = torch.tensor(beta, device=target_logits_local.device, dtype=target_logits_local.dtype)
                        gain_tensor = gain_tensor.expand(labels_local.size(0), 1)
                        target_logits_local.scatter_add_(1, labels_local.unsqueeze(1), gain_tensor)
                        target_probs_local = F.softmax(target_logits_local, dim=-1)
                    
                    # Step 6: Compute cross-entropy loss
                    student_log_probs_local = F.log_softmax(student_logits_local, dim=-1)
                    loss_local = -torch.sum(target_probs_local * student_log_probs_local, dim=-1)
                else:
                    # Validation phase
                    loss_local = loss_fct(student_logits_local, labels_local)

                # Gather results across sequence parallel ranks and restore original shape
                loss_gathered = gather_outpus_and_unpad(loss_local, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                def restore_shape(flat_tensor):
                    padded = pad_input(hidden_states=flat_tensor.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen)
                    return padded.squeeze(-1)[:, :-1].reshape(-1)

                loss_final = restore_shape(loss_gathered) * loss_mask

            # === Final Reduction ===
            valid_token_count = torch.sum(loss_mask)
            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_count)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            denom = valid_token_count + 1e-8
            
            avg_loss = torch.sum(loss_final) / denom * dp_size

            if do_backward:
                avg_loss.backward()

            return avg_loss

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        
        step_loss = 0.0

        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch)
            step_loss += loss.item() / n_micro_batches

        if self.config.model.strategy == 'fsdp':
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == 'fsdp2':
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        # Aggregate loss across all ranks
        step_loss = torch.tensor(step_loss).to(self.device_name)
        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.ulysses_device_mesh.size(0)

        metrics = {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
        }
        
        return metrics

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            # Validation uses standard cross-entropy loss (do_backward=False)
            ce_loss = self._compute_loss_and_backward(batch, do_backward=False)
            
            if is_cuda_available:
                torch.distributed.all_reduce(ce_loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(ce_loss)
                ce_loss /= self.ulysses_device_mesh.size(0)
        
        return ce_loss

    def save_checkpoint(self, step):
        # save checkpoint
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            # FSDP1 checkpoint saving
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
                state_dict = self.fsdp_model.state_dict()

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)
        elif fsdp_strategy == "fsdp2":
            # FSDP2 checkpoint saving
            from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

            # Get full state dict with FSDP2
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            # save huggingface model
            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            from omegaconf import OmegaConf
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = 0
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # TODO (zhangchi.usc1992) add back checkpoint manager.
        # Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(
                self.train_dataloader,
                total=self.steps_per_epoch,
                desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                disable=rank != 0
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(self.device_name)
                        val_ce_loss = self.validation_step(val_data)
                        val_losses.append(val_ce_loss)
                    if rank == 0:
                        val_ce_loss = torch.mean(torch.stack(val_losses))
                        metric = {
                            "val/loss": val_ce_loss.detach().item(),
                        }
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(dp_size, config.ulysses_sequence_parallel_size), mesh_dim_names=("dp", "sp"))
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPSFTWithKLTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh, tokenizer=tokenizer, train_dataset=train_dataset, val_dataset=val_dataset)

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path="config", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
    return dataset


if __name__ == "__main__":
    main()
