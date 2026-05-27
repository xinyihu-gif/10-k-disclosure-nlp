# What Drives Market Reactions? Evidence from NLP Analysis of 10-K Disclosures

This repository contains the code for an NLP project that analyzes textual disclosures in SEC 10-K filings and evaluates whether document-level text representations can help predict short-term market reactions.

## Project Overview

Financial disclosures such as 10-K filings contain rich information about firms' risks, operations, and future outlook. However, these documents are long, noisy, and often contain weak signals distributed across many sections.

This project compares several NLP-based modeling approaches to evaluate how different text representations capture information from long financial disclosure documents.

The main approaches include:

- TF-IDF + Logistic Regression
- Word2Vec document embeddings + Logistic Regression
- FinBERT embeddings + Logistic Regression
- Hierarchical Attention Network (HAN) exploratory experiments
- Lexical cue analysis for model interpretation

## Repository Structure

```text
10-k-disclosure-nlp/
├── scripts/              # Main experiment scripts
├── src/                  # Reusable preprocessing, feature, evaluation, and visualization code
├── notebooks/            # Experimental notebooks
├── outputs/
│   ├── figures/          # Confusion matrices and result figures
│   ├── reports/          # JSON result summaries
│   └── tables/           # Validation results and lexical cue tables
├── config.py             # Project path configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## Main Experiment Scripts

- TF-IDF baseline: `scripts/09_train_tfidf_logistic_large_lite.py`
- Word2Vec baseline: `scripts/05_word2vec_logistic_large_lite.py`
- Word2Vec tuning: `scripts/06_word2vec_logistic_large_lite_tuning.py`
- FinBERT embeddings: `scripts/14_finbert_embedding_large_lite_v1.py`
- Lexical cue analysis: `scripts/11_lexical_cue_analysis.py`

## Data

The original 10-K filing data and processed datasets are not included in this repository due to size and data-use constraints.

The project code is organized around project-root-relative paths configured in `config.py`.

## Outputs

Selected model results, validation tables, lexical cue outputs, and figures are included under `outputs/`.

Example output files include:

- Confusion matrices for selected models
- Validation result tables
- JSON summaries of model performance
- Lexical cue tables for positive and negative prediction classes

## Skills Demonstrated

- Python project organization
- Financial text preprocessing
- Long-document NLP pipeline design
- TF-IDF feature engineering
- Word2Vec embedding construction
- FinBERT-based document representation
- Logistic regression modeling
- Threshold tuning and model comparison
- Model evaluation and error analysis
- Lexical cue analysis and interpretation
- Reproducible experiment scripting

## Notes

This repository is intended to showcase the modeling pipeline, experiment scripts, and selected results. Large raw data files, processed datasets, and trained model objects are excluded from version control.
