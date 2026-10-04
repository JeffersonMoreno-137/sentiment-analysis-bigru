# IMDb Sentiment Analysis with Bidirectional GRU & GloVe Embeddings

End-to-end NLP binary classification pipeline for movie review sentiment analysis using PyTorch. The project benchmarks a baseline Bidirectional GRU against an architecture initialized with pre-trained GloVe embeddings.

## Key Performance Metrics

- **Dataset**: 40,000 raw reviews (balanced binary classification: Positive / Negative).
- **Test Accuracy**: ~89% (Macro F1-Score: 0.89).
- **Architectures**: Custom 2-Layer Bidirectional GRU vs. Pretrained GloVe (100d) Bi-GRU.
- **Key finding**: GloVe initialization converged faster (Epoch 3 vs Epoch 4) with lower validation loss (0.2692 vs 0.2974).

## 1. Problem Context

Movie reviews and ratings provide a rich source of unstructured textual data suited for identifying underlying user sentiment. This project implements an end-to-end binary sentiment classification pipeline on IMDb movie reviews, categorizing each text as either **positive** or **negative**.

Recurrent Neural Networks (RNNs) are well-suited for sequence processing and modeling temporal dependencies across textual tokens. Specifically, gated architectures such as `GRU` and `LSTM` capture long-term contextual relationships essential for estimating sequence polarity.

* **Dataset:** [IMDb Movie Ratings Sentiment Analysis (Kaggle)](https://www.kaggle.com/datasets/yasserh/imdb-movie-ratings-sentiment-analysis/data)

## 2. Objective

Design, train, and evaluate a recurrent neural network architecture capable of classifying IMDb reviews into binary sentiment categories (**positive** vs. **negative**).

The technical workflow encompasses:
- Text cleaning, regex-based normalization, and noise reduction.
- Vocabulary construction isolated strictly to the training split to prevent data leakage.
- Tokenization, fixed-length sequence padding, and truncation.
- Architectural implementation and training of a **Bidirectional GRU (Bi-GRU)** model.
- Quantitative evaluation utilizing Accuracy, Precision, Recall, F1-Score, and Confusion Matrix analysis.

## Model Architecture

```mermaid
graph LR
    subgraph P1["1. Data Pipeline"]
        Raw["Raw Text Review"] --> Clean["Regex & Lowercasing"]
        Clean --> Tok["Vocab Mapping (10k, pad=230)"]
    end

    subgraph P2["2. Recurrent Network"]
        Tok --> Emb["Embedding (Custom / GloVe 100d)"]
        Emb --> D1["Dropout (0.3)"]
        D1 --> GRU["2-Layer Bi-GRU (hidden=128)"]
    end

    subgraph P3["3. Classification Head"]
        GRU --> Concat["Concat Hidden (256d)"]
        Concat --> D2["Dropout (0.3)"]
        D2 --> FC["Linear (256 → 1)"]
        FC --> Loss["BCEWithLogits / Sigmoid"]
    end
```
