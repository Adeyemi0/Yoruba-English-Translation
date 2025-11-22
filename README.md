# 🌍 Yoruba ⇄ English Bidirectional Translation Model

A production-ready neural machine translation system for Yoruba-English bidirectional translation, built by fine-tuning AfriTeVa V2 on 101,906 translation pairs.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/adeyemi001/yoruba-english-translation-model)
[![Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/adeyemi001/yoruba-english-translation-model)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)


![Translation Demo](yoruba%20-%20english%20translation.gif)

**Live Demo:** [Try it on Hugging Face Spaces](https://huggingface.co/spaces/adeyemi001/yoruba-english-translation-model)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance](#-model-performance)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Web Interface](#-web-interface)
- [Training Details](#-training-details)
- [Dataset](#-dataset)
- [Technical Implementation](#-technical-implementation)
- [Deployment](#-deployment)
- [Known Limitations](#-known-limitations)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [Citation](#-citation)
- [License](#-license)
- [Connect](#-connect)

---

## 🎯 Overview

This project implements a state-of-the-art neural machine translation system for Yoruba-English language pairs. The model supports:

- **Bidirectional Translation**: Seamless English → Yoruba and Yoruba → English translation
- **Automatic Language Detection**: Intelligently detects input language using character patterns and linguistic features
- **Multiple Translation Variants**: Generates 4 different translations with quality scoring to give users options
- **Smart Text Chunking**: Handles long texts through intelligent sentence-based segmentation
- **Production-Ready API**: RESTful Flask API with CORS support and comprehensive error handling
- **Modern Web Interface**: Beautiful, responsive UI with real-time translation and diacritical mark support

Built on **AfriTeVa V2**, a transformer-based model specifically designed for African languages, and fine-tuned on over 100K curated translation pairs.

---

## ✨ Key Features

### 🔄 Intelligent Translation Pipeline

- **Multi-Output Generation**: Produces 4 translation variants using different decoding strategies (beam search, sampling, etc.)
- **Quality Scoring**: Automatically ranks translations based on length ratios, vocabulary diversity, and coherence
- **Hallucination Detection**: Identifies and flags potentially unreliable translations
- **Context-Aware Chunking**: Preserves sentence boundaries when processing long texts (80 tokens per chunk)

### 🌐 Language Detection

- **Diacritical Mark Recognition**: Detects Yoruba-specific characters (ẹ, ọ, ṣ, á, à, é, è, etc.)
- **Lexical Analysis**: Identifies common Yoruba words and patterns
- **Hybrid Approach**: Combines character-level and word-level features for robust detection

### 🎨 User Experience

- **Real-Time Preview**: Character counter and live text validation
- **Direction Toggle**: Easy switching between EN→YO, YO→EN, and auto-detect modes
- **Responsive Design**: Mobile-friendly interface with gradient aesthetics
- **Educational Tooltips**: Guidance on using Yoruba diacritical marks correctly

### 🚀 Performance Optimization

- **GPU Support**: Automatic CUDA detection and utilization
- **Token Counting**: Accurate token estimation for optimal chunking
- **Caching-Ready**: Decorator support for future LRU caching implementation
- **Efficient Batching**: Configurable batch sizes for throughput optimization

---

## 📊 Model Performance

### Evaluation Metrics (Test Set: 10,190 samples)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **BLEU** | 16.30 | Measures n-gram overlap (0-100, higher is better) |
| **BLEU-1** | 48.57 | Unigram precision |
| **BLEU-4** | 6.64 | 4-gram precision |
| **METEOR** | 46.26 | Considers synonyms and paraphrasing |
| **ROUGE-1** | 58.83 | Unigram recall |
| **ROUGE-2** | 34.34 | Bigram recall |
| **ROUGE-L** | 51.73 | Longest common subsequence |
| **chrF** | 39.28 | Character n-gram F-score |
| **TER** | 68.52 | Translation edit rate (lower is better) |

### Performance Context

These metrics reflect the challenges of translating between a high-resource language (English) and a low-resource language (Yoruba):

- **BLEU scores** are typical for morphologically rich languages like Yoruba
- **METEOR and ROUGE scores** indicate strong semantic preservation
- **chrF score** demonstrates good character-level alignment
- The model excels at capturing meaning despite exact phrase mismatches

---

## 🏗️ Architecture

### Base Model: AfriTeVa V2

- **Type**: Sequence-to-Sequence Transformer (T5 architecture)
- **Parameters**: ~300M (large variant)
- **Pretraining**: Multilingual African language corpus
- **Tokenizer**: SentencePiece with 250K vocabulary

### Fine-Tuning Configuration

```python
{
  "max_sequence_length": 128,
  "batch_size": 32,
  "gradient_accumulation_steps": 2,
  "effective_batch_size": 64,
  "learning_rate": 3e-5,
  "warmup_steps": 300,
  "num_epochs": 18,
  "early_stopping_patience": 3,
  "optimizer": "AdamW",
  "scheduler": "Linear with warmup"
}
```

### Training Infrastructure

- **Duration**: 5.36 hours (19,311 seconds)
- **Hardware**: NVIDIA GPU with CUDA support
- **Framework**: PyTorch + Hugging Face Transformers
- **Checkpointing**: Every 6,000 steps with validation

---

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) NVIDIA GPU with CUDA 11.0+ for acceleration

### Clone Repository

```bash
git clone https://github.com/yourusername/yoruba-english-translation.git
cd yoruba-english-translation
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
transformers>=4.30.0
flask>=2.3.0
flask-cors>=4.0.0
sentencepiece>=0.1.99
sacrebleu>=2.3.0
rouge-score>=0.1.2
```

### Environment Variables (Optional)

Create a `.env` file:

```bash
MODEL_PATH=adeyemi001/yoruba-english-translation-model
MAX_LENGTH=128
CACHE_SIZE=100
PORT=7860
```

---

## 🚀 Quick Start

### Run the Flask App

```bash
python app.py
```

The application will:
1. Download the model from Hugging Face (first run only)
2. Load model weights and tokenizer
3. Start the Flask server on `http://localhost:7860`

### Access Web Interface

Open your browser and navigate to:
```
http://localhost:7860
```

### Translate via Python

```python
from app import translate_text

# English to Yoruba
result = translate_text(
    text="Hello, how are you?",
    direction="en2yo",
    num_outputs=4
)
print(result['best_translation'])
# Output: Báwo ni, báwo ni o ṣe wà?

# Auto-detect language
result = translate_text(
    text="Ẹ káàsán",
    direction="auto",
    num_outputs=2
)
print(result['best_translation'])
# Output: Good afternoon
```

---

## 📡 API Reference

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_source": "adeyemi001/yoruba-english-translation-model",
  "device": "cuda",
  "max_length": 128
}
```

---

### Language Detection

**Endpoint:** `POST /detect-language`

**Request:**
```json
{
  "text": "Mo fẹ́ràn rẹ"
}
```

**Response:**
```json
{
  "detected_language": "yo",
  "language_name": "Yoruba"
}
```

---

### Translation

**Endpoint:** `POST /translate`

**Request:**
```json
{
  "text": "I love learning new languages",
  "direction": "en2yo",
  "num_outputs": 4
}
```

**Parameters:**
- `text` (required): Text to translate
- `direction` (optional): `"en2yo"`, `"yo2en"`, or `"auto"` (default: `"auto"`)
- `num_outputs` (optional): Number of variants (1-4, default: 4)

**Response:**
```json
{
  "success": true,
  "input": "I love learning new languages",
  "output": "Mo nífẹ̀ẹ́ kíkọ́ èdè tuntun",
  "all_translations": [
    {
      "text": "Mo nífẹ̀ẹ́ kíkọ́ èdè tuntun",
      "config": "Balanced (Wide Beam)",
      "quality_score": 0.92,
      "hallucination_detected": false,
      "length": 28
    },
    {
      "text": "Mo fẹ́ràn kíkọ́ èdè mìíràn tuntun",
      "config": "Conservative (Beam Search)",
      "quality_score": 0.88,
      "hallucination_detected": false,
      "length": 33
    }
  ],
  "metadata": {
    "direction": "en2yo",
    "detected_language": "en",
    "chunks_used": 1,
    "method": "single_pass_multi_output",
    "input_length": 30,
    "num_variants": 4
  }
}
```

---

### Clear Cache

**Endpoint:** `POST /clear-cache`

**Response:**
```json
{
  "success": true,
  "message": "Cache cleared successfully"
}
```

---

## 🎨 Web Interface

### Features

1. **Direction Selector**
   - 🔄 Auto-detect (default)
   - 🇬🇧 → 🇳🇬 English to Yoruba
   - 🇳🇬 → 🇬🇧 Yoruba to English

2. **Translation Preview**
   - Real-time character counting
   - Live input validation
   - Multiple output display with quality scores

3. **Educational Tooltips**
   - Guidance on Yoruba diacritical marks
   - Explanation of multi-output approach
   - Best practices for accurate translations

4. **Responsive Design**
   - Mobile-optimized layout
   - Gradient aesthetic with blue-purple theme
   - Smooth animations and transitions

### Demo

![Translation Demo](yoruba%20-%20english%20translation.gif)

---

## 📚 Training Details

### Dataset Composition

| Split | Samples | Percentage |
|-------|---------|------------|
| **Train** | 81,526 | 80% |
| **Validation** | 10,190 | 10% |
| **Test** | 10,190 | 10% |
| **Total** | 101,906 | 100% |

### Training Process

1. **Data Preprocessing**
  
   - Normalized Yoruba diacritics
   - Tokenized with AfriTeVa tokenizer
   - Padded/truncated to 128 tokens

2. **Fine-Tuning Strategy**
   - Prefix-based formatting: `"translate English to Yoruba: {text}"`
   - Gradient accumulation for effective batch size of 64
   - Learning rate warm-up over 300 steps
   - Linear decay schedule

3. **Validation & Checkpointing**
   - Evaluated every 6,000 training steps
   - Saved best model based on validation BLEU
   - Early stopping with patience of 3 epochs

4. **Training Timeline**
   - **Training**: 5.36 hours
   - **Evaluation**: 1.22 hours
   - **Total**: 5.73 hours

### Hyperparameter Tuning

The final configuration was selected after experimenting with:
- Learning rates: [1e-5, 3e-5, 5e-5]
- Batch sizes: [16, 32, 64]
- Max lengths: [64, 128, 256]
- Warmup steps: [100, 300, 500]

---

## 💾 Dataset

### Sources

The training data combines multiple high-quality Yoruba-English parallel corpora:

1. **JW300**: Jehovah's Witnesses translations
2. **FFR**: Foundation for Endangered Languages
3. **Menyo-20k**: Community-contributed translations
4. **Custom Corpus**: Manually curated sentence pairs

### Data Quality Measures

- ✅ Removed duplicates and near-duplicates
- ✅ Filtered out sentences with extreme length ratios (>3:1)
- ✅ Validated Yoruba diacritical mark usage
- ✅ Checked for proper Unicode encoding
- ✅ Removed machine-translated contamination

### Sample Pairs

| English | Yoruba |
|---------|--------|
| Good morning | Ẹ káàrọ̀ |
| How are you? | Báwo ni o ṣe wà? |
| Thank you very much | O ṣeun púpọ̀ |
| I don't understand | Mi ò yé mi |

---

## 🔬 Technical Implementation

### Translation Decoding Strategies

The model generates multiple translations using different decoding approaches:

#### 1. Conservative (Beam Search)
```python
{
  'num_beams': 5,
  'temperature': 0.6,
  'early_stopping': True,
  'no_repeat_ngram_size': 3,
  'length_penalty': 1.0
}
```
- **Best for**: Formal text, technical content
- **Characteristics**: Safe, predictable, grammatically correct

#### 2. Balanced (Wide Beam)
```python
{
  'num_beams': 8,
  'temperature': 0.7,
  'length_penalty': 1.2,
  'no_repeat_ngram_size': 3
}
```
- **Best for**: General purpose translation
- **Characteristics**: Natural phrasing, balanced creativity

#### 3. Precise (Low Temperature)
```python
{
  'num_beams': 6,
  'temperature': 0.5,
  'repetition_penalty': 1.1,
  'no_repeat_ngram_size': 2
}
```
- **Best for**: Short sentences, idioms
- **Characteristics**: Literal, focused on accuracy

#### 4. Creative (Sampling)
```python
{
  'temperature': 0.8,
  'do_sample': True,
  'top_k': 50,
  'top_p': 0.92,
  'repetition_penalty': 1.2
}
```
- **Best for**: Casual conversation, creative text
- **Characteristics**: Diverse, natural, context-aware

### Quality Scoring Algorithm

The system scores each translation variant based on:

```python
def score_translation(input_text: str, translation: str) -> float:
    score = 1.0
    
    # Length ratio penalty (expect similar lengths)
    length_ratio = len(translation) / max(len(input_text), 1)
    if length_ratio > 2.0 or length_ratio < 0.3:
        score -= 0.3
    
    # Vocabulary diversity (penalize repetition)
    unique_ratio = unique_words / total_words
    if unique_ratio < 0.5:
        score -= 0.3
    
    # Minimum length check
    if output_too_short:
        score -= 0.4
    
    # Bonus for reasonable length
    if 0.6 <= length_ratio <= 1.8:
        score += 0.1
    
    return max(0.0, min(1.0, score))
```

### Hallucination Detection

Flags potentially unreliable translations using:

1. **Length Anomalies**: Output >2.5x input length
2. **Excessive Repetition**: Unique words <40% of total
3. **Truncation**: Output <30% of expected length

### Smart Chunking Strategy

For long texts exceeding 128 tokens:

```python
def smart_sentence_chunk(text: str, max_tokens: int = 80) -> List[str]:
    # Split by sentence boundaries (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Group sentences until token limit
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if current_chunk_tokens + sentence_tokens > max_tokens:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    
    return chunks
```

**Benefits:**
- Preserves context within sentences
- Avoids mid-sentence breaks
- Maintains translation coherence

---

## 🚢 Deployment

### Local Deployment

```bash
# Clone repository
git clone https://github.com/Adeyemi0/yoruba-english-translation.git
cd yoruba-english-translation

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 7860

# Run application
CMD ["python", "app.py"]
```

**Build and run:**
```bash
docker build -t yoruba-translator .
docker run -p 7860:7860 yoruba-translator
```

### Hugging Face Spaces

The model is already deployed on Hugging Face Spaces:
[https://huggingface.co/spaces/adeyemi001/yoruba-english-translation-model](https://huggingface.co/spaces/adeyemi001/yoruba-english-translation-model)

To deploy your own Space:

1. Create a new Space on Hugging Face
2. Upload `app.py` and `requirements.txt`
3. Set SDK to "Gradio" or "Streamlit"
4. Configure environment variables
5. Space will auto-build and deploy

### Production Considerations

- **Scaling**: Use Gunicorn with multiple workers
- **Caching**: Implement Redis for translation caching
- **Monitoring**: Add logging to track usage and errors
- **Rate Limiting**: Implement API rate limits to prevent abuse
- **GPU Optimization**: Use TensorRT or ONNX for faster inference

---

## ⚠️ Known Limitations

### Model Limitations

1. **Idiomatic Expressions**: May translate literally rather than capturing cultural nuance
2. **Code-Switching**: Limited support for mixed Yoruba-English sentences
3. **Domain Specificity**: Performance may vary on technical/specialized vocabulary
4. **Diacritical Marks**: Requires proper input marks for best Yoruba results

### Technical Constraints

1. **Token Limit**: Maximum 128 tokens per chunk (handled via automatic chunking)
2. **GPU Memory**: Requires ~2GB VRAM for optimal performance
3. **Inference Speed**: ~0.5-2 seconds per translation depending on length
4. **Batch Processing**: Not optimized for large-scale batch translation

### Data Biases

- Training data skews toward formal/religious text
- Underrepresents colloquial and modern slang
- May reflect cultural biases present in source corpora

---

## 🔮 Future Improvements

### Short-Term (Next 3 Months)

- [ ] Implement caching with Redis/LRU to speed up repeated translations
- [ ] Add support for document translation (PDF, DOCX)
- [ ] Create mobile app (React Native)
- [ ] Expand API with batch translation endpoint
- [ ] Add pronunciation guide (IPA) for Yoruba outputs
- [ ] Fine-tune on domain-specific corpora (medical, legal, technical)
- [ ] Implement back-translation for quality assessment
- [ ] Add speech-to-text and text-to-speech integration
- [ ] Create browser extension for inline translation
- [ ] Build translation memory system for consistency
- [ ] Expand to other Nigerian languages (Igbo, Hausa)
- [ ] Develop context-aware translation using dialogue history
- [ ] Implement active learning with user feedback
- [ ] Create specialized models for code-switching scenarios
- [ ] Publish academic paper on low-resource language NMT

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

### Ways to Contribute

1. **Report Bugs**: Open an issue with detailed reproduction steps
2. **Suggest Features**: Propose new functionality via GitHub Discussions
3. **Improve Documentation**: Fix typos, add examples, clarify instructions
4. **Submit Code**: Fork the repo, create a feature branch, and open a PR
5. **Provide Data**: Share high-quality Yoruba-English sentence pairs

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/yoruba-english-translation.git
cd yoruba-english-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black app.py
flake8 app.py
```

### Contribution Guidelines

- Follow PEP 8 style guide
- Write unit tests for new features
- Update documentation for API changes
- Keep commit messages descriptive
- Squash commits before merging

---

## 📖 Citation

If you use this model in your research or project, please cite:

```bibtex
@software{adeyemi2024yoruba_translation,
  author = {Adediran, Adeyemi},
  title = {Yoruba-English Bidirectional Translation Model},
  year = {2024},
  publisher = {Hugging Face},
  url = {https://huggingface.co/adeyemi001/yoruba-english-translation-model},
  note = {Fine-tuned AfriTeVa V2 model for Yoruba-English translation}
}
```

### Acknowledgments

- **Base Model**: AfriTeVa V2 by Masakhane NLP
- **Training Infrastructure**: Google Colab Pro
- **Dataset Sources**: JW300, FFR, Menyo-20k, and community contributors
- **Framework**: Hugging Face Transformers

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### Key Points:

- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ⚠️ Must include copyright notice
- ⚠️ Must state changes made
- ⚠️ Liability and warranty disclaimer

---

## 🔗 Connect

**Developer**: Adediran Adeyemi

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/adediran-adeyemi-17103b114/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Profile-yellow?style=for-the-badge)](https://huggingface.co/adeyemi001)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=for-the-badge&logo=github)](https://github.com/Adeyemi0)

### Get in Touch

- 💼 **LinkedIn**: [Connect with me](https://www.linkedin.com/in/adediran-adeyemi-17103b114/)
- 📧 **Email**: adediran.yemite@yahoo.com
- 🌐 **Portfolio**: https://adeyemitheanalyst.pages.dev/


---

## 🙏 Support the Project

If you find this project helpful:

- ⭐ **Star** this repository on GitHub
- 🐛 **Report** bugs and suggest features
- 📢 **Share** with others working on African language NLP
- ☕ **Sponsor** development via GitHub Sponsors
- 📝 **Write** a blog post or tutorial about using the model

---

<div align="center">

**Made by Adediran Adeyemi**

🇳🇬 Preserving and promoting Nigerian languages through technology 🌍

</div>
