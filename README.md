# GIFT: Gibbs Initialization with Finite Temperature

GIFT is a unified post-training framework designed to bridge the optimization gap within the prevailing SFT-to-RL paradigm. By replacing traditional one-hot SFT with a finite-temperature Gibbs distribution, GIFT establishes a distributional bridge that preserves base priors while ensuring consistency with global post-training objectives. We theoretically and empirically demonstrate that GIFT provides an optimal initialization for RL in mathematical reasoning. Specifically, we show that standard SFT is merely a degenerate zero-temperature limit of this ideal policy. Our results indicate that GIFT significantly outperforms robust SFT variants across diverse and out-of-distribution benchmarks. Furthermore, geometric and distributional analyses reveal that GIFT preserves the exploration landscape, facilitating accelerated convergence and superior asymptotic performance to unlock the model’s full reasoning potential.

## Installation

```bash
cd verl
pip install -e .
```

## Usage

### Quick Start

```bash
bash exp_scripts/run_gift_training.sh
```

### Configuration

Edit `exp_scripts/run_gift_training.sh` to set your own paths:

```bash
train_file="/path/to/your/train.parquet"
val_file="/path/to/your/val.parquet"
model_path="/path/to/your/base_model"
```

### Data Format

Your parquet files should contain:
- `prompt_key`: column name for input prompts (default: `sft_prompt`)
- `response_key`: column name for target responses (default: `solution`)

### Acknowledgement

GIFT builds upon [veRL](https://github.com/volcengine/verl), [deepmath](https://github.com/zwhe99/DeepMath), and utilizes [vLLM](https://github.com/vllm-project/vllm) for inference. We utilize [Math-Verify](https://github.com/huggingface/Math-Verify) for math reasoning evaluation. We thank the open-source community for codes, datasets and backbones, including [veRL](https://github.com/volcengine/verl), [LUFFY](https://github.com/ElliottYan/LUFFY), [ReLIFT](https://github.com/TheRoadQaQ/ReLIFT).