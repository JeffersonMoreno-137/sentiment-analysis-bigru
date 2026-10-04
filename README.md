# IMDb Sentiment Analysis with Bidirectional GRU & GloVe Embeddings

End-to-end NLP binary classification pipeline for movie review sentiment analysis using PyTorch. The project benchmarks a baseline Bidirectional GRU against an architecture initialized with pre-trained GloVe embeddings.

## Key Performance Metrics

- **Dataset**: 40,000 raw reviews (balanced binary classification: Positive / Negative).
- **Test Accuracy**: ~89% (Macro F1-Score: 0.89).
- **Architectures**: Custom 2-Layer Bidirectional GRU vs. Pretrained GloVe (100d) Bi-GRU.
- **Key finding**: GloVe initialization converged faster (Epoch 3 vs Epoch 4) with lower validation loss (0.2692 vs 0.2974).

## Model Architecture

```mermaid
graph TD
    RawText[Raw Text Review] --> Clean[Regex Cleaning & Lowercasing]
    Clean --> Tokenizer[Vocab Mapping max_words=10k, pad_len=230]
    Tokenizer --> Embedding[Embedding Layer: Custom or GloVe 100d]
    Embedding --> Dropout1[Dropout p=0.3]
    Dropout1 --> BiGRU[2-Layer Bidirectional GRU hidden_dim=128]
    BiGRU --> Concat[Concat Forward + Backward Hidden States: 256d]
    Concat --> Dropout2[Dropout p=0.3]
    Dropout2 --> Linear[Linear Layer: 256 -> 1]
    Linear --> BCE[BCEWithLogitsLoss / Sigmoid]
