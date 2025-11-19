# MedChat: AI-Powered Medical Dialogue System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow Phases](#workflow-phases)
- [Usage Guide](#usage-guide)
- [Model Details](#model-details)
- [Results and Performance](#results-and-performance)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [Contributors](#contributors)
- [License](#license)

## 🎯 Overview

**MedChat** is an advanced AI-powered medical chatbot system built on the **Mistral 7B Instruct** model. The project implements a complete pipeline for creating a production-ready medical dialogue assistant with human-in-the-loop active learning and dynamic prompt adaptation.

This system is designed to:
- Provide accurate, contextual medical information based on patient queries
- Adapt response tone and detail level based on patient context (urgency, anxiety, knowledge level)
- Continuously improve through expert feedback and active learning
- Handle medical conversations with appropriate professional tone and safety

### 🎓 Educational & Research Focus
This project demonstrates state-of-the-art NLP techniques including:
- **Parameter-efficient fine-tuning** with QLoRA (Quantized Low-Rank Adaptation)
- **Human-in-the-loop machine learning** for quality assurance
- **Uncertainty quantification** for identifying low-confidence predictions
- **Dynamic few-shot prompting** for context-aware responses

## ✨ Key Features

### 1. **Fine-tuned Medical Dialogue Model**
- Base Model: Mistral 7B Instruct v0.3
- Specialized for medical conversations using domain-specific dialogue data
- Efficient fine-tuning using QLoRA for reduced memory footprint

### 2. **Human-in-the-Loop Active Learning (Phase 3)**
- **Uncertainty Quantification**: Identifies low-confidence model predictions
- **Expert Review Pipeline**: Routes uncertain responses for medical expert annotation
- **Continuous Improvement**: Incorporates expert feedback for model refinement
- **Quality Metrics**: Tracks confidence scores, uncertainty levels, and expert ratings

### 3. **Dynamic Tone & Detail Modulation (Phase 4)**
- **Context Detection**: Analyzes patient input for urgency, anxiety, knowledge level, and cultural sensitivity
- **Few-Shot Prompting**: Uses curated examples to guide response generation
- **Adaptive Responses**: Adjusts medical explanations based on patient needs
- **Multiple Tone Modes**: Professional, empathetic, simplified, or technical responses

### 4. **Robust Data Preprocessing**
- Cleans and formats medical dialogue datasets
- Handles patient queries and doctor response pairs
- Generates training, validation, and test splits
- Ensures data quality and consistency

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input: Patient Query                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Phase 4: Context Analysis & Prompt Gen              │
│  • Detect urgency, anxiety, knowledge level                      │
│  • Select appropriate few-shot examples                          │
│  • Generate dynamic prompt                                       │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              Fine-tuned Mistral 7B + QLoRA Adapter               │
│  • Efficient 4-bit quantization                                  │
│  • LoRA adapters (rank=16, alpha=32)                             │
│  • Medical dialogue fine-tuned weights                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Generate Response                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│           Phase 3: Uncertainty Analysis (Optional)               │
│  • Calculate entropy & confidence scores                         │
│  • If uncertainty > threshold → Expert Review                    │
│  • Otherwise → Return to user                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Output: Medical Response                       │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Technologies

### Core Framework
- **Python 3.8+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Hugging Face Transformers**: Model infrastructure
- **Jupyter Notebooks**: Interactive development and documentation

### Model & Training
- **Mistral 7B Instruct v0.3**: Base large language model
- **PEFT (Parameter-Efficient Fine-Tuning)**: LoRA implementation
- **BitsAndBytes**: 4-bit quantization for memory efficiency
- **Accelerate**: Distributed training support

### Data & Processing
- **Datasets (Hugging Face)**: Data loading and processing
- **Pandas & NumPy**: Data manipulation
- **JSON**: Data storage format

### Monitoring & Analysis
- **tqdm**: Progress tracking
- **Logging**: Comprehensive activity logging
- **Matplotlib/Seaborn**: Visualization (uncertainty analysis)

## 📁 Project Structure

```
medchat/
│
├── data_preprocessing.ipynb           # Phase 1: Data cleaning & preparation
├── train_mistral7b.ipynb             # Phase 2: Model fine-tuning with QLoRA
├── phase3_active_learning.ipynb      # Phase 3: Uncertainty-based expert review
├── phase4_dynamic_prompting.ipynb    # Phase 4: Context-aware prompt generation
│
├── dataset/                          # Raw medical dialogue data (not tracked)
│   ├── english-train.json
│   ├── english-dev.json
│   └── english-test.json
│
├── processed_data/                   # Preprocessed training data (not tracked)
│   ├── processed_train.json
│   ├── processed_dev.json
│   └── processed_test.json
│
├── model_output/                     # Fine-tuned model checkpoints (not tracked)
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
│
├── expert_feedback/                  # Active learning artifacts
│   ├── pending_samples.jsonl        # Samples awaiting expert review
│   ├── annotations.jsonl            # Expert annotations
│   ├── processed_feedback.jsonl     # Processed expert feedback
│   ├── high_confidence_samples.jsonl
│   ├── all_uncertainty_results.jsonl
│   ├── phase3_summary.json          # Pipeline statistics
│   ├── pipeline_statistics.json
│   └── uncertainty_analysis.png     # Visualization
│
└── README.md                         # This file
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended: 16GB+ VRAM for training)
- 32GB+ RAM recommended
- Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/Karthikeya-B19/medchat.git
cd medchat
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n medchat python=3.8
conda activate medchat
```

### Step 3: Install Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate peft bitsandbytes
pip install pandas numpy scipy matplotlib seaborn
pip install jupyter ipywidgets tqdm
pip install -U huggingface_hub
```

### Step 4: Setup Data
1. Download or prepare your medical dialogue dataset in JSON format
2. Place files in `dataset/` directory:
   - `english-train.json`
   - `english-dev.json`
   - `english-test.json`

### Step 5: Login to Hugging Face (for model access)
```bash
huggingface-cli login
```

## 📚 Workflow Phases

### Phase 1: Data Preprocessing (`data_preprocessing.ipynb`)

**Purpose**: Transform raw medical dialogue JSON data into a format suitable for fine-tuning.

**Key Steps**:
1. **Load raw JSON data** containing medical conversations
2. **Clean and validate** dialogue structure
3. **Format conversations** into instruction-following format
4. **Split data** into train/validation/test sets
5. **Save processed datasets** for training

**Input**: Raw medical dialogue JSON files
**Output**: Cleaned, formatted JSON datasets

**Example Data Format**:
```json
{
  "instruction": "You are a medical assistant. Respond to the patient's query professionally.",
  "input": "I've been having severe headaches for the past week. What could be causing this?",
  "output": "Headaches can have various causes including tension, dehydration, lack of sleep..."
}
```

### Phase 2: Model Fine-tuning (`train_mistral7b.ipynb`)

**Purpose**: Fine-tune Mistral 7B Instruct model on medical dialogue data using QLoRA.

**Key Steps**:
1. **Load base model** with 4-bit quantization (memory efficient)
2. **Configure LoRA adapters**:
   - Rank: 16
   - Alpha: 32
   - Dropout: 0.05
   - Target modules: q_proj, k_proj, v_proj, o_proj
3. **Setup training parameters**:
   - Epochs: 5
   - Learning rate: 2e-4
   - Batch size: 1 (with gradient accumulation)
   - Max sequence length: 256
4. **Train model** on medical dialogues
5. **Evaluate performance** on validation set
6. **Save LoRA adapters** for inference

**Training Configuration**:
```python
QLoRA Config:
- 4-bit quantization (NF4)
- Double quantization enabled
- Compute dtype: bfloat16

LoRA Config:
- r=16 (rank)
- alpha=32
- dropout=0.05
- bias="none"
```

**Output**: Fine-tuned model adapters in `model_output/`

### Phase 3: Active Learning (`phase3_active_learning.ipynb`)

**Purpose**: Identify uncertain predictions for expert review to improve model quality.

**Key Steps**:
1. **Load fine-tuned model** with adapters
2. **Generate predictions** on validation/test data
3. **Calculate uncertainty scores**:
   - Entropy-based uncertainty
   - Logit margin analysis
   - Confidence thresholds
4. **Identify low-confidence samples** (uncertainty > 0.25)
5. **Create expert review pipeline**:
   - Interactive annotation interface
   - Rating system (1-5 scale)
   - Correction/improvement tracking
6. **Process expert feedback**
7. **Generate statistics and reports**

**Uncertainty Metrics**:
- **Entropy**: Measures prediction uncertainty
- **Confidence**: Model's certainty in its output
- **Logit Margin**: Difference between top predictions

**Expert Review Process**:
```
Sample Review Rate: 37.5% (6 out of 16 samples)
Average Uncertainty: 0.228
Average Confidence: 94.7%
Retraining Rate: 83.3% (5 out of 6 reviewed)
```

**Output**: 
- Annotated samples in `expert_feedback/`
- Quality metrics and statistics
- Samples for model retraining

### Phase 4: Dynamic Prompting (`phase4_dynamic_prompting.ipynb`)

**Purpose**: Adapt response tone and detail level based on patient context using few-shot prompting.

**Key Steps**:
1. **Build Few-Shot Example Bank**:
   - Multiple tone variations (professional, empathetic, simplified)
   - Different detail levels (concise, moderate, detailed)
   - Context-specific examples
2. **Implement Context Detection**:
   - **Urgency detection**: Emergency keywords, symptom severity
   - **Anxiety detection**: Emotional language, worry indicators
   - **Knowledge level**: Medical terminology usage, question complexity
   - **Cultural sensitivity**: Language patterns, formality level
3. **Dynamic Prompt Generation**:
   - Select relevant few-shot examples
   - Construct context-aware system prompts
   - Modulate tone and detail parameters
4. **Integration with Model**:
   - Load fine-tuned model + adapters
   - Apply dynamic prompts at inference
   - Generate adaptive responses

**Context Detection Examples**:
```python
High Urgency: "severe chest pain", "can't breathe", "emergency"
High Anxiety: "really worried", "scared", "terrified"
Low Knowledge: Simple language, basic questions
High Knowledge: Medical terminology, specific questions
```

**Tone & Detail Levels**:
- **Professional**: Clinical, formal, evidence-based
- **Empathetic**: Supportive, reassuring, compassionate
- **Simplified**: Easy-to-understand, non-technical
- **Technical**: Detailed medical terminology, in-depth

**Example Output**:
```
Input: "I'm really scared about my chest pain. Is this serious?"

Context Analysis:
- Urgency: HIGH (chest pain)
- Anxiety: HIGH ("really scared")
- Knowledge: MEDIUM

Generated Response (Empathetic + Urgent):
"I understand your concern, and chest pain should always be taken seriously. 
While there are various causes ranging from muscle strain to more serious 
cardiac issues, it's important to seek immediate medical attention..."
```

## 📖 Usage Guide

### Running the Complete Pipeline

#### 1. Preprocess Data
```bash
jupyter notebook data_preprocessing.ipynb
# Run all cells to process raw data
```

#### 2. Train the Model
```bash
jupyter notebook train_mistral7b.ipynb
# Run all cells to fine-tune Mistral 7B
# This will take several hours on a GPU
```

#### 3. (Optional) Active Learning
```bash
jupyter notebook phase3_active_learning.ipynb
# Generate uncertainty scores
# Review low-confidence samples
# Provide expert annotations
```

#### 4. Use Dynamic Prompting
```bash
jupyter notebook phase4_dynamic_prompting.ipynb
# Test context-aware responses
# Experiment with different patient scenarios
```

### Quick Inference Example

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Load model with adapters
base_model = "mistralai/Mistral-7B-Instruct-v0.3"
adapter_path = "./model_output"

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Generate response
prompt = """[INST] You are a medical assistant. Respond professionally.

Patient: I've been having persistent headaches for a week. What could cause this?
[/INST]"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

## 🤖 Model Details

### Base Model: Mistral 7B Instruct v0.3
- **Parameters**: 7 billion
- **Architecture**: Decoder-only transformer
- **Context Length**: 8,192 tokens (extended from 4,096)
- **Quantization**: 4-bit NF4 format
- **Vocabulary Size**: 32,000 tokens

### Fine-tuning Strategy: QLoRA
**QLoRA (Quantized Low-Rank Adaptation)** enables efficient fine-tuning by:
- **4-bit quantization** of base model weights (reduces memory by ~75%)
- **LoRA adapters** as trainable parameters (only ~1% of total parameters)
- **Double quantization** for further memory savings
- **paged optimizers** for handling memory spikes

**Benefits**:
- Train 7B models on consumer GPUs (16GB VRAM)
- Maintain close-to-full-precision performance
- Fast adaptation to domain-specific tasks
- Easy model sharing (adapters are small ~50MB)

### Training Statistics
```
Dataset Size:
- Training: 482 dialogues
- Validation: 60 dialogues

Training Configuration:
- Epochs: 5
- Batch Size: 1 (effective 8 with gradient accumulation)
- Learning Rate: 2e-4
- Sequence Length: 256 tokens
- Training Time: ~2-4 hours on A100 GPU
```

## 📊 Results and Performance

### Active Learning Statistics (Phase 3)
```
Pipeline Performance:
- Total Samples Analyzed: 16
- High Confidence: 10 (62.5%)
- Low Confidence (Reviewed): 6 (37.5%)
- Average Confidence: 94.7%
- Average Uncertainty: 0.228

Expert Review Results:
- Total Annotations: 6
- Samples for Retraining: 5 (83.3%)
- Average Expert Rating: 2.83/5
- Rating Distribution:
  * 1 star: 0
  * 2 stars: 2
  * 3 stars: 3
  * 4 stars: 1
  * 5 stars: 0
```

### Model Capabilities
✅ **Strengths**:
- Contextual medical information retrieval
- Professional and empathetic tone
- Appropriate medical terminology usage
- Safety-conscious responses
- Adaptive detail levels

⚠️ **Limitations**:
- Not a replacement for professional medical advice
- May have biases from training data
- Requires continuous monitoring and updates
- Should always include medical disclaimer

## 🔮 Future Enhancements

### Planned Features
1. **Multi-turn Conversation Management**
   - Dialogue state tracking
   - Context retention across sessions
   - Follow-up question handling

2. **Enhanced Safety Mechanisms**
   - Medical disclaimer injection
   - Harmful content filtering
   - Emergency detection and routing

3. **Multilingual Support**
   - Translation integration
   - Language-specific fine-tuning
   - Cultural adaptation

4. **Integration Features**
   - REST API for deployment
   - Web/mobile interface
   - EHR system integration
   - Telemedicine platform compatibility

5. **Advanced Active Learning**
   - Automated expert review scheduling
   - Batch annotation workflows
   - Continuous model updates
   - A/B testing framework

6. **Performance Optimization**
   - Model distillation for faster inference
   - Caching mechanisms
   - Streaming responses
   - Quantization optimization

7. **Evaluation Metrics**
   - Medical accuracy benchmarks
   - Patient satisfaction surveys
   - Expert validation scores
   - Safety incident tracking

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide for Python code
- Add docstrings to functions and classes
- Include comments for complex logic
- Test changes before submitting PR
- Update documentation as needed

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Mistral AI** for the Mistral 7B model
- **Hugging Face** for transformers and PEFT libraries
- **Medical dialogue dataset providers**
- **Open-source AI/ML community**

## 👥 Contributors

- **Karthikeya B** ([@Karthikeya-B19](https://github.com/Karthikeya-B19)) - Project Creator & Maintainer
- **Satya Karthikeya** ([@satyakarthikeya](https://github.com/satyakarthikeya)) - Contributor

## ⚠️ Disclaimer

**IMPORTANT**: This chatbot is for educational and research purposes only. It is **NOT** a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

---

## 📞 Contact

**Project Maintainer**: Karthikeya B
- GitHub: [@Karthikeya-B19](https://github.com/Karthikeya-B19)
- Repository: [medchat](https://github.com/Karthikeya-B19/medchat)

---

<div align="center">
Made with ❤️ and 🤖 by Karthikeya B
</div>
