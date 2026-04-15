# 🔤 Word2Vec Models — CBOW vs Skip-Gram

A hands-on implementation of **Word2Vec** using Gensim, comparing the two core architectures: **Continuous Bag of Words (CBOW)** and **Skip-Gram**. The models are trained on a machine learning text corpus and used to explore semantic word relationships through dense vector representations.

---

## 📌 Overview

Word2Vec is a shallow neural network-based technique that learns word embeddings — dense numerical representations of words that capture semantic meaning. This notebook:

- Preprocesses raw text (tokenization, stop word removal, lowercasing)
- Trains both CBOW and Skip-Gram Word2Vec models
- Compares word vectors and semantic similarity results across the two architectures

---

## 🧠 Models Implemented

| Model | Gensim Parameter | Description |
|---|---|---|
| **CBOW** | `sg=0` | Predicts a target word from its surrounding context words |
| **Skip-Gram** | `sg=1` | Predicts surrounding context words from a single target word |

Both models are trained with:
- `vector_size = 100`
- `window = 5`
- `min_count = 1`

---

## 📁 Notebook Structure

```
Word2Vec_models_.ipynb
│
├── 1. Install Dependencies         # pip install gensim, PyPDF2
├── 2. Import Libraries             # pandas, nltk, gensim, re
├── 3. Text Extraction (Optional)   # PDF ingestion via PyPDF2
├── 4. Preprocessing                # Lowercase, remove punctuation, stop words
├── 5. Tokenization & Windowing     # Sliding window sentence creation
├── 6. Train CBOW Model             # Word2Vec with sg=0
├── 7. Train Skip-Gram Model        # Word2Vec with sg=1
├── 8. Inspect Vocabulary           # Vocabulary size for both models
├── 9. Display Word Vectors         # First 10 dimensions for sample words
└── 10. Semantic Similarity         # Most similar words for a target word
```

---

## 🛠️ Setup & Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Install Dependencies

```bash
pip install gensim nltk PyPDF2
```

Or run directly in the notebook:

```python
!pip install gensim PyPDF2
```

---

## ▶️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/suneetpaul/Deep_learning.git
   cd Deep_learning/Topic_modling
   ```

2. Launch the notebook:
   ```bash
   jupyter notebook Word2Vec_models_.ipynb
   ```

3. Run all cells in order. The notebook uses a built-in machine learning text corpus by default — no external dataset required.

> **Optional:** To use your own PDF, uncomment the PDF extraction block and provide a file path:
> ```python
> pdf = '/content/your_document.pdf'
> text = extract_text_from_pdf(pdf)
> ```

---

## 🔍 Key Features

### Text Preprocessing
```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return tokens
```
Removes punctuation, converts to lowercase, and filters English stop words using NLTK.

### Word Similarity Example
```python
target_word = "machine"

# CBOW
cbow_model.wv.most_similar(target_word, topn=5)

# Skip-Gram
skipgram_model.wv.most_similar(target_word, topn=5)
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `gensim` | Word2Vec model training |
| `nltk` | Stop words corpus |
| `PyPDF2` | Optional PDF text extraction |
| `pandas` | Data handling |
| `re` | Regex-based text cleaning |

---

## 📚 References

- [Mikolov et al. (2013) — Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
- [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
- [NLTK Stop Words](https://www.nltk.org/book/ch02.html)

---

*Part of the [Deep Learning](https://github.com/suneetpaul/Deep_learning) repository by [@suneetpaul](https://github.com/suneetpaul)*
