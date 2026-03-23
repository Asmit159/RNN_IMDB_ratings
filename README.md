# 🎬 IMDB Sentiment Analysis using RNN + TF-IDF

A complete end-to-end **sentiment analysis pipeline** built on the IMDB 50K movie reviews dataset.
This project classifies reviews as **Positive** or **Negative** using a hybrid approach of **TF-IDF feature extraction** and a neural network (RNN-based architecture).

---

##  Overview

This project demonstrates:

* Text preprocessing and cleaning for NLP
* Feature extraction using **TF-IDF**
* Training a neural network for classification
* Model evaluation using multiple metrics
* Manual inference on custom user input

---

##  Dataset

* **IMDB Dataset** (50,000 movie reviews)
* Balanced dataset:

  * 25,000 positive
  * 25,000 negative

---

##  Model Architecture

Although structured as an RNN, the model effectively behaves like a **dense classifier over TF-IDF features**.

```
Input (TF-IDF Vector - 5000 features)
        ↓
RNN Layer (sequence length = 1)
        ↓
Fully Connected Layer
        ↓
Sigmoid Output
```

---

##  Preprocessing Pipeline

The following steps were applied:

1. Lowercasing text
2. Removing URLs
3. Removing HTML tags
4. Removing punctuation
5. Stopword removal (NLTK)
6. Stemming (Porter Stemmer)

---

##  Tech Stack

* Python
* PyTorch
* Scikit-learn
* NLTK
* Pandas
* NumPy

---

##  Results

| Metric    | Score       |
| --------- | ----------- |
| Accuracy  | **~85.3%**  |
| AUC-ROC   | **0.94 🔥** |
| Precision | ~85.4%      |
| Recall    | ~87.5%      |

---

##  Confusion Matrix

```
            Pred 0     Pred 1
True 0      4115        753
True 1       631       4418
```

### Interpretation:

* Strong class separation (high AUC)
* Balanced performance across both classes
* Errors mainly due to:

  * sarcasm
  * negation ("not good")
  * mixed sentiment

---

##  Evaluation Metrics

* Confusion Matrix
* Classification Report
* ROC Curve (AUC = 0.94)

---

##  Key Insights

* TF-IDF captures strong sentiment signals effectively
* Model performs well on clear sentiment cases
* Limitations arise due to:

  * lack of word order understanding
  * inability to capture context

---

##  Example Predictions

```python
predict_review("This movie was absolutely amazing!")
# Output: Positive (Confidence: ~95%)

predict_review("Worst movie ever, total waste of time")
# Output: Negative (Confidence: ~98%)
```

---

##  Limitations

* RNN does not fully utilize sequence learning due to TF-IDF input
* Cannot handle:

  * negation properly
  * sarcasm
  * contextual meaning

---

##  Future Improvements

* Replace TF-IDF with **word embeddings**
* Upgrade to:

  * LSTM / GRU
  * Bidirectional models
* Use **Transformer-based models (BERT)**
* Add attention mechanism
* Deploy as a web app

---

##  How to Run

1. Upload dataset to Google Drive
2. Mount drive in Colab
3. Run notebook cells sequentially
4. Use:

```python
predict_review("Your review here")
```

---

##  Key Takeaway

This project highlights how **classical NLP (TF-IDF)** combined with deep learning can achieve strong baseline performance, while also showing the importance of **context-aware models** for deeper understanding.

---

## ⭐ If you liked this project

Give it a ⭐ on GitHub and feel free to contribute!

---
