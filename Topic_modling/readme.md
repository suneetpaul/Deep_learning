# Topic Modeling Analysis: LSA, LDA, and NMF

This Jupyter notebook implements three different topic modeling techniques to analyze PDF documents. The notebook demonstrates how to extract, preprocess, and model topics from text using Latent Semantic Analysis (LSA), Latent Dirichlet Allocation (LDA), and Non-negative Matrix Factorization (NMF).

## Overview

The notebook performs topic modeling on a PDF document about "Legal Aspects of Mergers and Acquisitions in India" using three different unsupervised learning techniques to discover hidden thematic structures in the text.

## Features

- **PDF Text Extraction**: Extracts text content from PDF files using PyPDF2
- **Text Preprocessing**: Cleans and normalizes text data (lowercasing, removing special characters, stopword removal)
- **LSA (Latent Semantic Analysis)**: Uses TF-IDF vectorization and TruncatedSVD for topic extraction
- **LDA (Latent Dirichlet Allocation)**: Probabilistic topic modeling using count-based vectorization
- **NMF (Non-negative Matrix Factorization)**: Matrix factorization technique for topic discovery

## Requirements

### Python Libraries

```bash
pip install scikit-learn gensim nltk PyPDF2
```

### Dependencies

- **scikit-learn**: For vectorization and topic modeling algorithms (TruncatedSVD, LatentDirichletAllocation, NMF)
- **gensim**: Additional NLP and topic modeling capabilities
- **nltk**: Natural Language Toolkit for text preprocessing (stopwords)
- **PyPDF2**: PDF parsing and text extraction
- **pandas**: Data manipulation (imported but not actively used in current version)

## Usage

### 1. Upload PDF Document

The notebook is designed to run in Google Colab and includes a file upload widget:

```python
from google.colab import files
uploaded = files.upload()
```

Alternatively, you can specify a direct path to your PDF:

```python
pdf = '/content/your_document.pdf'
```

### 2. Run the Notebook

Execute cells sequentially to:
1. Install dependencies
2. Extract text from PDF
3. Preprocess the text
4. Apply LSA modeling
5. Apply LDA modeling
6. Apply NMF modeling

### 3. View Results

Each model outputs discovered topics with their top 10 most relevant terms.

## Methodology

### Text Preprocessing

The preprocessing pipeline includes:
- Converting text to lowercase
- Removing non-alphabetic characters using regex
- Tokenization (splitting text into words)
- Stopword removal using NLTK's English stopwords list

```python
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens)
```

### LSA (Latent Semantic Analysis)

- **Vectorization**: TF-IDF with max 5000 features
- **Dimensionality Reduction**: TruncatedSVD with 10 components
- **Output**: 10 topics with top 10 terms each

### LDA (Latent Dirichlet Allocation)

- **Vectorization**: Count-based (Bag of Words) with max 5000 features
- **Model**: LDA with 10 topics
- **Output**: 10 topics with top 10 terms each

### NMF (Non-negative Matrix Factorization)

- **Vectorization**: TF-IDF (reuses LSA vectorizer)
- **Model**: NMF with 10 components
- **Output**: 10 topics with top 10 terms each

## Sample Output

The notebook extracts topics related to the legal domain, particularly focusing on:
- SEBI regulations and insider trading
- Banking and RBI regulations
- Corporate mergers and acquisitions (Tata Steel, HDFC, Vodafone-Idea, Zomato-Blinkit)
- FEMA and cross-border transactions
- Competition law and market integration

Example topic from LSA:
```
Topic 0: ['sebi', 'section', 'act', 'regulations', 'india', 'mergers', 'company', 'rbi', 'trading', 'companies']
```

## Key Parameters

### Configurable Settings

- **Number of topics**: Set `n_components=10` in all three models (can be adjusted)
- **Maximum features**: Set `max_features=5000` in vectorizers (can be adjusted)
- **Random state**: Set to `42` for reproducibility

### Customization

You can modify these parameters based on your needs:
- Increase/decrease number of topics
- Adjust feature count for vocabulary size
- Change n-gram range in vectorizers
- Modify minimum/maximum document frequency thresholds

## File Structure

```
.
├── LSA_LDA_NMF___AS1.ipynb    # Main notebook
└── README.md                   # This file
```

## Notes

- The notebook is optimized for Google Colab but can be adapted for local Jupyter environments
- Make sure to download NLTK stopwords on first run: `nltk.download('stopwords')`
- PDF extraction quality depends on the source PDF structure and formatting
- All three models use `random_state=42` for reproducible results

## Comparison of Techniques

| Technique | Vectorization | Approach | Best For |
|-----------|--------------|----------|----------|
| **LSA** | TF-IDF | SVD decomposition | Finding semantic relationships, smaller datasets |
| **LDA** | Count-based | Probabilistic generative model | Discovering latent topics, interpretability |
| **NMF** | TF-IDF | Matrix factorization | Non-negative data, sparse representations |

## Future Enhancements

Potential improvements to the notebook:
- Add visualization of topics (word clouds, t-SNE plots)
- Implement topic coherence metrics for model evaluation
- Add document-topic distribution visualization
- Include perplexity scores for LDA
- Experiment with optimal number of topics using elbow method
- Add support for multiple PDF files
- Export results to CSV or JSON

## Author

Assignment 1 - Topic Modeling Analysis

## License

This notebook is provided for educational purposes.