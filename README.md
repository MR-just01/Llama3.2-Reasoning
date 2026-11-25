# Llama 3.2 Reasoning — Fine-Tuning and Evaluation

A fine-tuning project focused on improving the mathematical, logical, and general reasoning capabilities of **Meta Llama 3.2 3B Instruct** using parameter-efficient fine-tuning techniques.

The project covers the complete experimentation workflow, including dataset preparation, preprocessing, QLoRA fine-tuning, model evaluation, and manual auditing.

## Project Overview

Large Language Models can generate fluent responses but may struggle with multi-step mathematical and logical reasoning.

This project explores whether fine-tuning an open-source language model on reasoning-oriented datasets can improve its ability to solve structured problems and generate step-by-step explanations.

The primary model used is:

- **Base Model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-Tuning Method:** QLoRA (Quantized Low-Rank Adaptation)
- **Training Framework:** Hugging Face Transformers and TRL
- **Experimentation Environment:** Kaggle GPU

## Dataset Preparation

The training dataset combines multiple publicly available reasoning datasets.

| Dataset | Approximate Samples |
|---|---:|
| GSM8K | 7,473 |
| AQUA-RAT | 19,999 |
| StrategyQA | 1,603 |
| ARC-Challenge | 1,118 |
| **Total** | **30,193** |

The datasets cover different reasoning tasks:

- **Mathematical reasoning:** Multi-step arithmetic and word problems.
- **Commonsense reasoning:** Questions requiring everyday knowledge and logical inference.
- **Science reasoning:** Science-related questions and conceptual problem-solving.

The preprocessing pipeline includes dataset merging, formatting, and deduplication.

The resulting dataset contains the following fields:

- `instruction`
- `input`
- `reasoning`
- `answer`
- `dataset`
- `task_type`

## Fine-Tuning Methodology

The primary model was fine-tuned using QLoRA to reduce GPU memory requirements.

### Training Configuration

| Parameter | Configuration |
|---|---|
| Base model | Llama 3.2 3B Instruct |
| Quantization | 4-bit NF4 |
| Fine-tuning method | QLoRA |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Learning Rate | 2e-4 |
| Epochs | 1 |
| Training Samples | 27,173 |
| Validation Samples | 3,020 |

Approximately 24.3 million parameters were trainable, representing around 0.75% of the model parameters.

This approach makes fine-tuning more memory-efficient than updating the entire model.

## Model Checkpoint

The fine-tuned LoRA adapter is available on Hugging Face:

**[MR023/Llama3.2-Reasoning](https://huggingface.co/MR023/Llama3.2-Reasoning)**

The adapter is designed to be loaded alongside its compatible base model.

## Evaluation

The project includes evaluation experiments to assess the model's answer generation and reasoning performance.

Evaluation work includes:

- Generating model responses for reasoning questions.
- Extracting final answers from generated responses.
- Comparing predicted answers against reference answers.
- Reviewing incorrect and incomplete responses.
- Conducting manual audits of answer correctness and reasoning quality.
- Identifying repetitive or poorly formatted generations.

### Evaluation Observations

In one evaluation run involving 3,000 responses:

- 2,720 responses contained an identifiable `Final Answer:` marker.
- 1,703 out of 3,000 responses matched the reference answer using the initial exact-match evaluation.
- A later normalized comparison produced 1,699 matches out of 2,720 extractable answers.

These figures use different extraction and comparison procedures. They should not be interpreted as a definitive mathematical reasoning accuracy score. Exact-match evaluation can also incorrectly penalize mathematically equivalent answers.

Manual auditing was used to examine answer correctness separately from reasoning quality.

## Repository Structure

```text
Llama3.2-Reasoning/
│
├── data/
│   └── Dataset files and processed data
│
├── notebooks/
│   ├── Dataset preparation
│   ├── Llama fine-tuning
│   ├── Qwen fine-tuning experiments
│   └── Model evaluation
│
├── outputs/
│   └── Model outputs and checkpoints
│
├── reports/
│   └── Evaluation and manual auditing reports
│
├── src/
│   └── Source-code documentation
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- BitsAndBytes
- TRL
- Pandas
- Scikit-learn
- Kaggle GPU

## Running the Experiments

The experiments are organized as Jupyter notebooks.

1. Clone the repository.
2. Install the required dependencies in a compatible environment.
3. Configure access to the gated Llama model through Hugging Face.
4. Open the notebooks in the appropriate execution order.
5. Run dataset preparation, fine-tuning, and evaluation experiments.

```bash
git clone https://github.com/MR-just01/Llama3.2-Reasoning.git

cd Llama3.2-Reasoning
```

A GPU environment is recommended for fine-tuning.

## Future Improvements

- Implement reusable Python modules for the training pipeline.
- Improve mathematical answer normalization and evaluation.
- Introduce automated reasoning-quality evaluation.
- Expand the evaluation dataset.
- Compare additional open-source reasoning models.
- Improve experiment tracking and reproducibility.

## Project Focus

This repository focuses on **LLM experimentation and fine-tuning**, rather than hosting a live inference service.

The corresponding API implementation is maintained separately in [llamaReasoningFineTuneModel](https://github.com/MR-just01/llamaReasoningFineTuneModel).
