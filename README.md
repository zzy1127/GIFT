# GIFT: Gibbs Initialization with Finite Temperature

GIFT is a unified post-training framework that addresses the intrinsic optimization mismatch in the prevailing SFT $\to$ RL paradigm, which replaces traditional one-hot sft with a finite temperature Gibbs distribution, creating a distributional bridge that preserves base priors while ensuring consistency with global post-training objectives.

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
