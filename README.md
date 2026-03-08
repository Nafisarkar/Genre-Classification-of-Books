# Genre Classification of Books

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

Modern, concise toolkit for classifying book genres from text using classical and deep-learning models.

## Table of contents

- [Genre Classification of Books](#genre-classification-of-books)
  - [Table of contents](#table-of-contents)
  - [About](#about)
    - [Project summary](#project-summary)
    - [Key Features](#key-features)
    - [Tech stack](#tech-stack)
    - [Performance benchmark (example)](#performance-benchmark-example)
    - [System architecture (high level)](#system-architecture-high-level)
    - [Impact \& sustainability](#impact--sustainability)
    - [Project team](#project-team)
  - [Highlights](#highlights)
  - [Quickstart](#quickstart)
  - [Usage examples](#usage-examples)
  - [Datasets \& models](#datasets--models)
  - [Development \& contributing](#development--contributing)

## About

Genre Classification of Books — a sophisticated automated system designed to classify literary works into genres by analyzing textual patterns from book excerpts. The project goes beyond surface metadata and focuses on the actual text, addressing limitations of manual categorization.

### Project summary

- Contextual intelligence that analyzes the first and last ~20 pages of a book to capture thematic introductions and conclusions.
- Hybrid data sourcing supporting born-digital PDFs and physical books via OCR integration.
- Multi-model architecture ranging from classical SVMs and Random Forests to Transformer-based models (BERT).
- End-to-end pipeline: OCR → cleaning → tokenization → TF‑IDF/embeddings → classification.

### Key Features

| Feature | Description |
|---|---|
| Contextual Intelligence | Analyzes the first and last 20 pages of a book to capture essential thematic introduction and conclusion. |
| Hybrid Data Sourcing | Processes born-digital PDFs and physical books via custom OCR integration (image capture → OCR → cleanup). |
| Multi-Model Architecture | Evaluates algorithms from traditional SVMs and Random Forests up to Transformer-based models (BERT). |
| End-to-End Pipeline | Automated text cleaning (noise reduction, stopword removal, normalization) and feature extraction using TF‑IDF or embeddings. |
| Web-Ready Interface | Flask-based dashboard for realtime PDF uploads and genre prediction (see `Webapplication/`). |

### Tech stack

| Category | Tools & Technologies |
|---|---|
| Languages | Python 3.10 |
| ML / DL Frameworks | TensorFlow, Keras, Scikit-learn, PyTorch |
| Preprocessing | OpenCV, Fitz (PyMuPDF), NLTK |
| Web Backend | Flask |
| Infrastructure | Google Colab (TPU Runtime), MiniConda |

### Performance benchmark (example)

Trained on ~600 books across 6 genres (Horror, Mystery, Thriller, Comedy, Romance, Fiction):

| Model | Accuracy | Notes |
|---|---:|---|
| BERT (Transformer) | 75% | Best balance of performance and robustness |
| SVM | 73% | Stable with smaller feature sets |
| Random Forest | 72% | Good baseline performance |
| Gradient Boosting (GBM) | 70% | Competitive with RF, slower to train |
| RNN | 57% | Struggles with long-range dependencies and large corpora |

These are experimental results — re-train and validate on your data for reproducible metrics.

### System architecture (high level)

1. Data acquisition: extract text from PDFs or capture images of physical pages.
2. OCR processing: convert images to searchable, cleaned English text using the OCR helper utilities.
3. Preprocessing: lowercase, noise removal, tokenization, stopword filtering, normalization.
4. Feature extraction: TF‑IDF, embeddings, or BERT token encodings.
5. Classification: run chosen model(s) to predict genre; return label + confidence score.

### Impact & sustainability

- Preservation: enables indexing of historical and rare books that lack modern metadata.
- Eco-friendly: supports digital libraries and reduces printing/shipping carbon footprint.
- User empowerment: improves recommendation systems and discovery of stylistically aligned works.

### Project team

| Name | Role / Affiliation |
|---|---|
| Shaon An Nafi | DIU — CSE Graduate |
| Peya Pandit | DIU — CSE Graduate |
| Department | Computer Science and Engineering, Daffodil International University |

## Highlights

- Multiple model implementations: classical ML (Random Forest, SVM, GBM, Naive Bayes, kNN) and neural nets (CNN, RNN).
- Lightweight Flask demo to test inference locally.
- Notebook-driven exploration in `Multiclass_Text_Classification.ipynb`.



## Quickstart

1. (Optional) Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies (create `requirements.txt` first if not present):

```powershell
python -m pip install -r requirements.txt
```

3. Train a model (example):

```powershell
python Random_Forest.py
```

4. Run the demo webapp:

```powershell
cd Webapplication
python app.py
```

Then open http://127.0.0.1:5000 in your browser.

## Usage examples

- Predict with a saved model (example CLI pattern — adapt for your script):

```powershell
python prediction.py --model Models/random_forest.pkl --text "A young wizard's adventure"
```

- Run the notebook for EDA and model comparisons:

```powershell
jupyter notebook Multiclass_Text_Classification.ipynb
```

## Datasets & models

- The CSVs `datasetV4.csv` and `datasetV5.csv` contain labeled examples used for training and evaluation. Inspect and preprocess them before training.
- Save trained models to the `Models/` folder so the `Webapplication` and `prediction.py` can load them.

## Development & contributing

- Add a `requirements.txt` with pinned versions and consider using `pip-tools` or `poetry` for reproducible environments.
- Provide small README snippets or docstrings inside each model script describing expected inputs/outputs and CLI flags.
- If you add model artifacts, include a small JSON file describing model metadata (algorithm, training date, input preprocessing).
