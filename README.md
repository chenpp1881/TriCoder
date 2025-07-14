# TriCoder

This repository contains the source code for the method proposed in our paper, "Decoupling Code Classification for Semantic Transfer: Learning from LLM-Generated Explanations". TriCoder leverages LLM-generated explanations to enhance code classification tasks by decoupling the understanding of code semantics from the classification process.

## Overview

TriCoder introduces a novel approach to code classification that:

- Generates task-specific explanation perspectives using LLMs
- Creates detailed explanations for code samples based on these perspectives
- Fuses code representations with explanation embeddings for improved classification performance

## 1. Setup and Execution

Follow these steps to set up the environment and run the experiments:

### 1.1 Environment Configuration

1. **PyTorch Environment:** The code was developed and tested with the following specifications:

   - PyTorch: `2.1.0`
   - Python: `3.10`
   - CUDA: `12.1`
   - OS: Ubuntu 22.04

   You can set up a similar Conda environment or use a Docker image with these base requirements.

2. **Install Dependencies:** Navigate to the project root directory and install the required packages:

   ```bash
   pip install transformers pandas openai accelerate scikit-learn
   ```

### 1.2 Dataset Preparation

1. Create a `Datasets` directory in the project root to store the raw datasets.

2. The directory structure should be:

   ```
   Datasets/
   ├── Devign/
   │   └── extracted_data.jsonl
   ├── Reveal/
   │   └── extracted_data.jsonl
   ...
   ```

3. Each line in `extracted_data.jsonl` represents a sample with the following format:

   ```json
   {"code": "...", "label": "..."}
   ```

### 1.3 Generating Explanation Perspectives

1. **Configure API Key:** Add your OpenAI API key in `get_explanations/vd_oai_model_interface.py`

2. **Generate Perspectives:** Run the following command to generate task-specific explanation perspectives:

   ```bash
   python get_perspectives.py
   ```

3. **Output:** The generated perspectives will be stored in the `criteria/` directory with the structure:

   ```
   criteria/
   └── [task_id]/
       └── final_prompt_set_[model].json
   ```

### 1.4 Generating Explanations

1. **Generate Explanations for Code Samples:** Execute the following script to generate explanations for each code sample in your dataset:

   ```bash
   python get_explanations/vd_answer_check.py
   ```

2. **Output:** The generated explanations will be stored in the `Filtered_DS/` directory:

   ```
   Filtered_DS/
   └── [task_id]/
       └── [model].jsonl
   ```

### 1.5 Model Training

1. **Train the Model:** Run the training script with appropriate arguments:

   ```bash
   python run.py --task_id [TASK_NAME] --model_path [MODEL]
   ```

   Example:

   ```bash
   python run.py --task_id Devign --model_path codet5-base
   ```

2. **Arguments:**

   - `--task_id`: Task identifier (Devign, Reveal, etc.)
   - `--model_path`: Pre-trained model path (codet5-base, graphcodebert-base, etc.)
   - `--epoch`: Number of training epochs (default: 50)
   - `--batch_size`: Batch size for training (default: 16)
   - `--lr_IF`: Learning rate for the Information Fusion module (default: 1e-5)

3. **Output:** Training results and model checkpoints will be saved in the `Results/` directory.

## 2. File Descriptions

### Core Modules

- **`get_perspectives.py`**: Generates task-specific explanation perspectives using LLMs. It implements a multi-stage process including task description generation, perspective creation, filtering, and scoring.
- **`IFmodule.py`**: Information Fusion Module that combines code embeddings with explanation embeddings using multi-head attention mechanisms.
- **`train.py`**: Training logic implementation including model training loops, evaluation metrics calculation, and checkpoint saving functionality.
- **`run.py`**: Main entry point for model training. Handles argument parsing, environment setup, and orchestrates the training process.

### Utility Files

- **`utils.py`**: Utility functions for data processing, JSON extraction, sampling, and formatting. Includes functions for extracting structured data from LLM responses.
- **`data_utils.py`**: Dataset implementation and data loading utilities. Handles tokenization, label encoding, and stratified train-test splitting.

### Directories

- **`get_explanations/`**: Contains scripts and prompt templates for generating code explanations:
  - `vd_oai_model_interface.py`: OpenAI API interface
  - `vd_answer_check.py`: Explanation generation pipeline
  - `prompt_templates.py`: Prompt templates for various tasks
- **`modules/`**: Some neural network building blocks.