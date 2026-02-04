# Knowledge Graph-Guided Reinforcement Learning for LLMs

This repository contains the implementation code for training large language models using knowledge graph-guided reinforcement learning, as described in our paper.

## 📄 Paper

**[Knowledge Graph-Guided Reinforcement Learning for LLMs](https://arxiv.org/abs/2601.15160)**

For detailed methodology, experimental results, and theoretical foundations, please refer to the paper.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

### Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU training)
- DeepSpeed
- Transformers
- TRL (Transformer Reinforcement Learning)
- PEFT (Parameter-Efficient Fine-Tuning)

## 📁 Repository Structure

```
.
├── data_loader.py               # Data loading utilities (placeholder for training data)
├── data_prep.py                 # Dataset preprocessing and train/test splitting
├── create_filtered_dataset.py  # Diversity-based filtering for SFT/RL splits
├── sft_training.py              # Supervised Fine-Tuning with LoRA
├── rl_training.py               # Reinforcement Learning (GRPO) training
├── configs/
│   ├── deepspeed_config.json    # DeepSpeed ZeRO-3 configuration
│   └── slurm_template.sh        # SLURM job submission template
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Usage

### 1. Data Preparation

Our training pipeline uses a sophisticated **diversity-based filtering** approach to split data between SFT and RL:

#### Step 1: Create Filtered Dataset for RL

Create a high-diversity subset (e.g., 5k examples) for RL training:

```bash
python create_filtered_dataset.py \
    --input_path <path-to-full-dataset> \
    --output_path <path-to-filtered-dataset> \
    --target_size 5000 \
    --min_per_category 2
```

This script uses stratified sampling to maximize:
- **Category coverage**: All categories represented
- **Concept diversity**: Rare source/target concepts prioritized
- **Path pattern variety**: Diverse knowledge graph paths
- **Node coverage**: Maximum unique nodes from KG

The **filtered dataset** (5k examples) is used for **RL training**, while the **remaining examples** (~19k) are used for **SFT training**.

#### Step 2: Preprocess for Training (Optional)

The training scripts automatically handle data preprocessing. However, you can optionally preprocess datasets in advance:

**For SFT training:**
```bash
python data_prep.py \
    --input_path <path-to-sft-dataset> \
    --output_path <path-to-processed-sft-data> \
    --mode sft
```

**For RL training:**
```bash
python data_prep.py \
    --input_path <path-to-filtered-dataset> \
    --output_path <path-to-processed-rl-data> \
    --mode rl \
    --enable_thinking
```

> **Note:** If you skip this step, the training scripts will automatically convert the data to the correct format during training.

### 2. Supervised Fine-Tuning (SFT)

Train a base model using LoRA for parameter-efficient fine-tuning:

```bash
# Single-node multi-GPU training
torchrun --nproc_per_node=8 sft_training.py \
    --model_name "Qwen/Qwen3-14B" \
    --dataset_path <path-to-training-data> \
    --output_dir ./sft_models/qwen3-14b-lora \
    --learning_rate 2e-4 \
    --num_train_epochs 20 \
    --deepspeed configs/deepspeed_config.json

# Or submit via SLURM
sbatch configs/slurm_template.sh
```

### 3. Reinforcement Learning Training (GRPO)

Continue training with RL using the SFT checkpoint:

```bash
torchrun --nproc_per_node=8 rl_training.py \
    --model_name "Qwen/Qwen3-14B" \
    --sft_checkpoint_path ./sft_models/qwen3-14b-lora/checkpoint-XXX \
    --dataset_path <path-to-rl-data> \
    --output_dir ./rl_models/qwen3-14b-grpo \
    --learning_rate 8e-6 \
    --num_train_epochs 2
```

## 🎯 Key Features

- **Diversity-Based Data Splitting**: Sophisticated filtering that maximizes concept, category, and KG path coverage
- **Parameter-Efficient Training**: Uses LoRA for memory-efficient fine-tuning of large models
- **Knowledge Graph Integration**: Incorporates KG path information in reward functions
- **Multi-Stage Training**: SFT followed by RL for improved reasoning capabilities
- **Distributed Training**: Supports multi-GPU and multi-node training with DeepSpeed ZeRO-3
- **Flexible Architecture**: Easily adaptable to different model architectures (Qwen, LLaMA, etc.)

## 📊 Training Configuration

### SFT Hyperparameters
- Learning rate: 2e-4
- Batch size: 1 per device, gradient accumulation: 32
- LoRA rank: 16, alpha: 16
- Max sequence length: 2048 tokens
- Epochs: 20

### RL Hyperparameters
- Learning rate: 8e-6
- Beta (KL penalty): 0.05
- Number of generations: 2
- Max prompt length: 896 tokens
- Max completion length: 896 tokens
- Epochs: 2

## 🛠️ Reward Functions

The RL training uses multiple reward signals:

1. **Correctness Reward**: Binary reward for correct answer extraction
2. **Path Alignment Reward**: Measures alignment with knowledge graph paths using token overlap
3. **Thinking Quality Reward**: Evaluates reasoning structure and coherence (step-by-step reasoning)
4. **Semantic Similarity Reward**: Compares model reasoning with correct option text

### Data Splitting Strategy

The training uses a **two-stage data split**:

1. **Filtered Dataset (RL)**: 5k examples selected for maximum diversity
   - Ensures coverage of all categories
   - Prioritizes rare concepts and path patterns
   - Maintains long-tail coverage
   
2. **Remaining Dataset (SFT)**: ~19k examples for supervised fine-tuning
   - Provides broad coverage and pattern learning
   - Builds strong base capabilities

This split ensures the RL stage focuses on diverse, challenging examples while the SFT stage provides comprehensive coverage.

## 💾 Model Checkpointing

Models are saved at regular intervals during training. To load a checkpoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-14B")

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./sft_models/qwen3-14b-lora")

# For inference, merge adapters
model = model.merge_and_unload()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")
```

## 📝 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{kansal2026knowledge,
  title={Knowledge Graphs are Implicit Reward Models: Path-Derived Signals Enable Compositional Reasoning},
  author={Kansal, Yuval and Jha, Niraj K},
  journal={arXiv preprint arXiv:2601.15160},
  year={2026}
}
```

## 📧 Contact

For questions or issues, please open a GitHub issue or refer to the paper for contact information.

## 📜 License

This project is licensed under the Princeton License - see the LICENSE file for details.

## 🙏 Acknowledgments

This work builds upon:
- [Transformers](https://github.com/huggingface/transformers)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [PEFT](https://github.com/huggingface/peft)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
