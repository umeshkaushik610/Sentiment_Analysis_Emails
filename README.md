# 📧 Spam Detection using BERT — Transfer Learning for NLP

![TensorFlow](https://img.shields.io/badge/TensorFlow-2.6-orange) ![BERT](https://img.shields.io/badge/Model-BERT-blue) ![Accuracy](https://img.shields.io/badge/Accuracy-88%25-green)

Not your average Naive Bayes spam filter. This project fine-tunes Google's BERT transformer for binary spam classification — using transfer learning so that only **769 parameters** need to be trained on top of a 109M-parameter language model.

---

## 🤔 Why BERT?

Most spam classifiers use TF-IDF + Naive Bayes or Logistic Regression. These treat words as independent frequency counts with no contextual understanding. Consider:

> "You have a chance to **win a scholarship**" vs "You have a chance to **win $5000 — claim NOW**"

A bag-of-words model sees both as near-identical. BERT reads the entire sentence bidirectionally and understands the difference.

| Approach | How it works | Context-aware? | Accuracy* |
|---|---|---|---|
| Naive Bayes (TF-IDF) | Word frequency counts | ❌ No | ~95% |
| Logistic Regression | Bag-of-words features | ❌ No | ~96% |
| LSTM | Sequential word patterns | ⚠️ Partially | ~97% |
| **BERT (this project)** | Bidirectional transformer | ✅ Full context | **88%** |

> **Note on accuracy:** Traditional ML scores higher here because spam often contains obvious keywords ("win", "$5000"). BERT's real advantage is on ambiguous, context-dependent sentences. This model also trains on only ~1,500 balanced samples vs ~5,500 for traditional models.

---

## 📊 Dataset

**Source:** [Kaggle — SMS Spam Dataset](https://www.kaggle.com/code/sid321axn/sms-spam-classifier-naive-bayes-ml-algo/data)

| Split | Ham (0) | Spam (1) | Total |
|---|---|---|---|
| Original | 4,825 | 747 | 5,572 (imbalanced) |
| After undersampling | 747 | 747 | 1,494 (balanced) |

Labels: `ham → 0`, `spam → 1`

---

## 🧠 Model Architecture

> **Transfer learning key insight:** BERT's 109 million parameters are completely frozen — pre-trained on Wikipedia + BooksCorpus. Only the 769-parameter classification head is trained. The model leverages massive language understanding from just ~1,500 training samples.

**BERT components (TensorFlow Hub):**
- Preprocessor: [bert_en_uncased_preprocess/3](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3)
- Encoder: [bert_en_uncased_L-12_H-768_A-12/4](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)

| Layer | Output Shape | Params | Trainable? |
|---|---|---|---|
| BERT encoder (pooled) | (None, 768) | 109,482,241 | No (frozen) |
| Dropout | (None, 768) | 0 | — |
| Dense (sigmoid) | (None, 1) | 769 | **Yes** |

Total trainable params: **769**

---

## ⚙️ Setup
```bash
pip install tensorflow==2.6
pip install tensorflow-hub
pip install tensorflow-text
pip install scikit-learn pandas numpy
```

> ⚠️ TF, TF Hub, and TF Text versions must be compatible. See the [official guide](https://www.tensorflow.org/text/tutorials/classify_text_with_bert).

---

## 📈 Results

| Metric | Value |
|---|---|
| Overall accuracy | **88%** |
| Spam recall | **0.92** |
| Test samples | 374 |

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Ham (0) | 0.91 | 0.84 | 0.87 |
| Spam (1) | 0.85 | 0.92 | 0.88 |

Spam recall of 0.92 means the model correctly catches 92% of spam — the more important metric here.

---

## 🔮 Inference Example
```python
reviews = [
    'Enter a chance to win $5000, hurry up, offer valid until march 31',
    'Hey Sam, are you coming for a cricket game tomorrow',
]
predictions = model.predict(reviews)
# Apply threshold: probability > 0.5 → Spam
```

| Message | Probability | Prediction |
|---|---|---|
| "Enter a chance to win $5000..." | 0.606 | 🔴 Spam |
| "Hey Sam, cricket game tomorrow" | 0.486 | 🟢 Ham |

---

## 🔧 Possible Improvements

- [ ] Unfreeze top N BERT encoder layers for full fine-tuning
- [ ] Use full imbalanced dataset with class weights instead of undersampling
- [ ] Try smaller BERT variant (`bert_en_uncased_L-4_H-512_A-8`) for faster inference
- [ ] Deploy as a Flask / FastAPI endpoint
- [ ] Add a Streamlit UI for live predictions

---

## 📚 References

- [BERT fine-tuning with TensorFlow — official tutorial](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- [BERT encoder — TF Hub](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4)
- [Dropout layers explained](https://towardsdatascience.com/machine-learning-part-20-dropout-keras-layers-explained-8c9f6dc4c9ab)
- [Dataset — Kaggle](https://www.kaggle.com/code/sid321axn/sms-spam-classifier-naive-bayes-ml-algo/data)
