# 🔢 Multi-Token Handwritten Recognition: Digits, Letters & Alphanumerics

> 🧠 Developed by **Dr. Poulami Nandi** — Data Scientist & ML Researcher with a PhD in Quantum Physics from IIT Kanpur and Postdoc from the University of Pennsylvania.

<p align="center">
  <img src="assets/pipeline.png" width="700"/>
</p>

---

## 🗂️ Project Summary

This project extends classic MNIST digit recognition to a more general task: recognizing sequences of **1 to 10 handwritten tokens** – digits, letters, and alphanumeric characters – placed side by side. It simulates real-world OCR scenarios and explores how classification accuracy degrades with sequence length.

### 🏆 Key Features

- Supports:
  - **Digit-only tokens (0–9)** from MNIST
  - **Letter tokens (A–Z)** from EMNIST (letters split)
  - **Alphanumeric tokens (A–Z, a–z, 0–9)** from EMNIST (byclass)
- Flexible CNN model with multiple softmax heads – one per token
- Dynamic data generator to create composite sequences of variable length
- In-depth **EDA, training visualization, and error analysis**
- Evaluates **sequence-level accuracy** and per-token metrics
- Modular architecture suitable for curriculum learning or OCR pretraining

---

## 🧠 Architecture Overview

<p align="center">
  <img src="assets/loss_curves.png" width="500"/>
</p>

- **Input:** Grayscale images of shape (28 × *n*, 28), where *n* is number of tokens
- **Backbone:** 2 Convolutional + MaxPooling → Dense(128)
- **Heads:** Each token is predicted via an independent `Dense(10|26|62)` layer
- **Loss:** Categorical Crossentropy per head
- **Metrics:** Per-token accuracy, Full-sequence accuracy

---

## 🧩 Function Breakdown

### 🔹 `build_dataset(n_digits=3, mode="digit", sample_limit=50000)`

- Dynamically builds composite sequences from MNIST/EMNIST.
- Tokens are randomly drawn and `np.hstack`ed to simulate sequences.
- Returns `train`, `val`, and `test` splits with token-wise one-hot labels.
- Modes: `digit`, `letter`, `alphanumeric`

### 🔹 `eda_summary(x, ys, n_digits, title_prefix)`

- Plots:
  - **Pixel intensity distribution** across samples
  - **Class balance** for first token
- Helps detect skewed training data or bias in sampling

### 🔹 `build_model(n_digits, vocab_size)`

- Builds a Keras model with:
  - Shared CNN feature extractor
  - Multiple softmax heads depending on token count
  - Output shape: `[(None, vocab_size)] * n_digits`
- Supports:
  - `vocab_size = 10` (digits), `26` (letters), `62` (alphanum)

### 🔹 `train_and_evaluate(n_digits, mode, sample_size)`

- Trains and evaluates the model on sequences of `n_digits` tokens
- Returns:
  - `digit-wise accuracy`
  - `sequence accuracy`
  - training `history`
  - validation `loss/accuracy` plots
  - confusion matrix for top-k predictions

### 🔹 `compare_performance_across_lengths()`

- Runs `train_and_evaluate` for 1 to 10 token lengths
- Plots sequence accuracy vs. number of tokens
- Generates summary table

---

## 📊 Exploratory Data Analysis (EDA)

<p align="center">
  <img src="assets/sample_composite.png" width="450"/>
</p>

- Shows how pixel intensity is distributed
- Tracks class imbalance across token types
- Samples shown per token position

---

## 🧪 Results

| Tokens | Sequence Accuracy | Sample Size | Top-1 Token Accuracy |
|--------|-------------------|-------------|----------------------|
| 1-digit   | 0.992           | 10k         | 0.992                |
| 2-digit   | 0.983           | 50k         | 0.993, 0.991         |
| 3-digit   | 0.973           | 60k         | 0.990, 0.988, 0.987  |
| ...       | ...             | ...         | ...                  |
| 10-digit  | ~0.800          | 60k         | ~0.92–0.94 range     |

> 📌 As the number of tokens increases, **sequence accuracy declines faster** than per-token accuracy due to compounding error.

---

## 🔠 Supported Modes

| Mode | Dataset Used | Characters Covered | Output Heads |
|------|--------------|--------------------|---------------|
| `digit` | MNIST        | 0–9                | 10            |
| `letter` | EMNIST/letters | A–Z              | 26            |
| `alphanumeric` | EMNIST/byclass | A–Z, a–z, 0–9 | 62          |

---

## 📁 Directory Structure

```
composite_token_classifier/
├── README.md
├── requirements.txt
├── assets/                   ← banners, loss plots, confusion heatmaps
├── notebooks/
│   └── digit_letter_alpha_classifier.ipynb
├── src/
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── viz_utils.py
│   └── train.py
└── examples/
    └── quick_demo.py
```

---

## 📦 Installation

```bash
git clone https://github.com/Poulami-Nandi/composite_token_classifier.git
cd composite_token_classifier
pip install -r requirements.txt
```

---

## 🚀 Try It Yourself

Run 3-digit digit model with 50K samples:
```python
from train import train_and_evaluate
train_and_evaluate(n_digits=3, mode="digit", sample_size=50000)
```

Compare digit vs. letter vs. alphanumeric accuracy:
```python
for mode in ["digit", "letter", "alphanumeric"]:
    for n in range(1, 6):
        train_and_evaluate(n_digits=n, mode=mode, sample_size=60000)
```

---

## 🔍 Use Cases

- Handwriting OCR with variable-length input
- Document parsing from scanned forms
- Pretraining for language-to-vision token models
- Sequence recognition in low-resource languages

---

## 👩‍💻 About Me

**Dr. Poulami Nandi**  
PhD in Quantum Physics – IIT Kanpur  
Postdoc – University of Pennsylvania  

💼 *Currently a Data Scientist applying ML across Healthcare, Finance & Physics*  
🔗 [LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/)  
📚 [Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- `tensorflow.keras.datasets.mnist`
- `tensorflow_datasets.emnist`
- Academic inspiration from quantum pattern recognition and digit tokenization

