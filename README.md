# Machine Translation with Seq2Seq Neural Networks

An English-to-Spanish Neural Machine Translation (NMT) system built using TensorFlow, Keras, NumPy, and a Seq2Seq Encoder-Decoder architecture with LSTM networks.

This project began as the Codecademy Machine Translation capstone project and was extended significantly with additional functionality, performance improvements, a modern Keras workflow, an interactive command-line interface, and custom sentence translation capabilities.

---

## Features

### Neural Machine Translation

Translate English text into Spanish using:

- Seq2Seq architecture
- Encoder-Decoder neural networks
- LSTM layers
- Teacher forcing
- TensorFlow and Keras

---

### Interactive Translator

Translate custom English sentences directly from the command line.

Example:

```text
English: Who?
Spanish: ¿Quién?
```

```text
English: Run!
Spanish: ¡Órale!
```

---

### Dataset Translation Testing

Test translations against examples from the training dataset.

Menu option:

```text
2 - Test dataset examples
```

---

### Model Statistics Dashboard

View model metadata directly from the application.

Example:

```text
==================================================
MODEL STATISTICS
==================================================

Encoder vocabulary : 2516
Decoder vocabulary : 5302

Maximum input tokens : 8
Maximum output tokens: 17

Latent dimension : 256

Model size (MB): 75.52
```

---

### Unknown Vocabulary Detection

The translator warns when words are not present in the training vocabulary.

Example:

```text
English: My name is Wayne

Warning: 'Wayne' not found in training vocabulary.
```

---

### Improved Tokenization

Custom input now uses the same regex tokenization strategy used during preprocessing.

Example:

```text
Who?
```

becomes:

```python
['Who', '?']
```

This significantly improved custom translation accuracy.

---

## Project Structure

```text
machine-translation/
│
├── data/
│   └── spa.txt
│
├── models/
│   └── training_model.keras
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── translate.py
│   └── utils.py
│
├── tests/
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## Dataset

Source:

ManyThings.org Language Dataset

Language Pair:

```text
English → Spanish
```

Training Data Used:

```text
10,000 sentence pairs
```

Vocabulary Learned:

```text
Input Vocabulary  : 2516 tokens
Output Vocabulary : 5302 tokens
```

---

## Model Architecture

### Encoder

```text
Input Sequence
      ↓
LSTM (256 units)
      ↓
Hidden State
Cell State
```

### Decoder

```text
Previous Token
      ↓
LSTM (256 units)
      ↓
Dense Softmax Layer
      ↓
Next Token Prediction
```

### Total Parameters

```text
9,894,582
```

---

## Training Configuration

```python
latent_dim = 256
batch_size = 50
epochs = 100
```

Training includes:

```python
EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)
```

which automatically stops training when validation performance stops improving.

---

## Technologies Used

- Python 3
- TensorFlow
- Keras
- NumPy
- Neural Networks
- LSTM Networks
- Seq2Seq Models
- Natural Language Processing (NLP)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/WayneFranz77/machine-translation.git
```

Navigate into the project:

```bash
cd machine-translation
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment:

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Train Model

```bash
python src/train.py
```

The trained model will be saved as:

```text
models/training_model.keras
```

---

### Run Translator

```bash
python src/translate.py
```

Application menu:

```text
==================================================
 MACHINE TRANSLATION SYSTEM
==================================================

1 - Translate custom text
2 - Test dataset examples
3 - Show model statistics
4 - Exit
```

---

## Example Results

### Custom Translation

Input:

```text
Who?
```

Output:

```text
¿Quién?
```

---

Input:

```text
Run!
```

Output:

```text
¡Órale!
```

---

Input:

```text
How are you?
```

Output:

```text
¿Estabas aquí?
```

---

## Challenges Encountered

During development the following challenges were addressed:

- Increasing training data from 500 to 10,000 sentence pairs
- Managing memory constraints when attempting 20,000+ sentence pairs
- Refactoring the application to separate training and inference
- Migrating from HDF5 (`.h5`) to modern Keras (`.keras`) model storage
- Improving tokenization consistency
- Handling vocabulary mismatches and unknown words
- Adding an interactive translation workflow

---

## Future Improvements

Potential future enhancements:

- Word embeddings
- Attention mechanisms
- Transformer architecture
- Better handling of unknown words
- Confidence scores for translations
- Web interface using Streamlit
- Larger training datasets
- BLEU score evaluation metrics

---

## Learning Outcomes

This project demonstrates:

- Deep learning fundamentals
- Neural machine translation
- Seq2Seq architectures
- Encoder-decoder networks
- TensorFlow and Keras workflows
- Model training and persistence
- Data preprocessing
- NLP tokenization techniques
- Git and GitHub project management

---

## Topics

Machine Learning • Deep Learning • Natural Language Processing • NLP • Machine Translation • Neural Machine Translation • Seq2Seq • LSTM • Encoder Decoder • TensorFlow • Keras • Python • Artificial Intelligence

---

## License

This project is released under the MIT License.

See the LICENSE file for details.

---

## Author

Wayne Franz

Created as part of the Codecademy Machine Translation project and expanded into a standalone machine learning portfolio project.