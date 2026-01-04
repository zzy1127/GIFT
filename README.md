# GIFT: Gibbs Initialization with Finite Temperature

A training framework for supervised fine-tuning with KL-regularized target distribution.

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

### Key Hyperparameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `beta` | Inverse temperature gain (β) - controls the strength of signal for target tokens | 20.0 |
| `smooth_lambda` | Smoothing coefficient for numerical stability | 0.0 |

### Data Format

Your parquet files should contain:
- `prompt_key`: column name for input prompts (default: `sft_prompt`)
- `response_key`: column name for target responses (default: `solution`)

## Algorithm

GIFT constructs an advantage-adjusted target distribution during training:

```
ẑ_{t,k} = log p_ref(k|x,y*_<t) + β · 𝟙[k = y*_t]
```

The model is trained to match this target distribution via cross-entropy loss.

## Acknowledgement

Built on top of [veRL](https://github.com/volcengine/verl).

