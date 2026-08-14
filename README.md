# 🎫 Auto Email / Ticket Categorizer

A lightweight NLP-based support ticket classification system built with **Python, Scikit-learn, and Streamlit**.

The application automatically classifies incoming support tickets into:

- Billing
- Technical
- HR
- General

It also provides:

- TF-IDF text representation
- Logistic Regression classification
- Accuracy, Precision, Recall and F1
- Confusion matrix
- Real-time ticket classification
- Confidence score
- Human-review threshold
- Urgent/Normal priority tagging
- Five sample ticket predictions
- Streamlit live demo

## Project Architecture

```text
Ticket/Email
     |
     v
Text Cleaning
     |
     v
TF-IDF Vectorization
     |
     v
Logistic Regression
     |
     +----> Category
     |
     +----> Confidence
     |
     +----> Priority
     |
     +----> Human Review Decision
```

## Project Structure

```text
auto-ticket-categorizer/
│
├── app.py
├── train_model.py
├── generate_dataset.py
├── requirements.txt
├── README.md
│
├── data/
│   └── tickets.csv
│
├── models/
│   ├── ticket_classifier.pkl
│   └── tfidf_vectorizer.pkl

```

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Generate Dataset

The assessment requires an original dummy dataset.

Run:

```bash
python generate_dataset.py
```

This creates:

```text
data/tickets.csv
```

## Train the Model

Run:

```bash
python train_model.py
```

The trained model and TF-IDF vectorizer are saved under:

```text
models/
```

## Run Streamlit

Run:

```bash
streamlit run app.py
```

The application opens at:

```text
http://localhost:8501
```

## Model Choice

Logistic Regression was selected because the dataset is a small multi-class text classification problem and TF-IDF produces a high-dimensional sparse representation. Logistic Regression is efficient for this setting and exposes class probabilities through `predict_proba()`, which is useful for confidence-based human review.

Naive Bayes would also be a suitable baseline for this problem.

## Human Review

The default confidence threshold is 60%.

If:

```text
confidence < 60%
```

the application displays:

```text
Needs Human Review
```

Otherwise the ticket is considered suitable for automatic routing.

The threshold can be changed from the Streamlit sidebar.

## Priority Tagging

Simple keyword rules identify potentially urgent tickets.

Examples:

- urgent
- critical
- down
- not working
- blocked
- failed
- failure
- emergency

A matching ticket receives:

```text
URGENT
```

otherwise:

```text
NORMAL
```

## Sample Tickets

### Billing

```text
I was charged twice for my monthly subscription.
```

### Technical

```text
The application keeps crashing whenever I try to login.
```

### HR

```text
How many vacation days can I take this year?
```

### General

```text
Can you tell me where I can find the user guide?
```

### Urgent Technical

```text
URGENT! The production server is completely down and our team cannot access the application.
```

## Dataset Limitation

The dataset is intentionally self-created for demonstration purposes. Because it is relatively small and synthetic, its evaluation metrics may be higher than performance on real-world support tickets.

For production use, the model should be trained using a larger, continuously collected and human-reviewed dataset.

## Future Improvements

With more data and time, I would:

1. Compare Logistic Regression, Naive Bayes and Linear SVM.
2. Use cross-validation for more robust evaluation.
3. Add class imbalance handling if needed.
4. Add confidence calibration.
5. Add model/data drift monitoring.
6. Store human-review corrections for retraining.
7. Add authentication and an API layer for enterprise deployment.
8. Add multilingual ticket classification.
9. Add explainability showing important words behind a prediction.
10. Deploy the Streamlit application using a cloud platform.

## Assessment Coverage

| Requirement | Implementation |
|---|---|
| Own dummy dataset | `generate_dataset.py` |
| Text preprocessing | `clean_text()` |
| TF-IDF | `TfidfVectorizer` |
| Classifier | Logistic Regression |
| Accuracy | Streamlit KPI |
| Precision | Streamlit KPI |
| Recall | Streamlit KPI |
| F1 | Streamlit KPI |
| Confusion matrix | Evaluation tab |
| New ticket prediction | Live Classifier |
| 5+ sample tickets | Test Examples |
| Confidence | `predict_proba()` |
| Human review | 60% threshold |
| Priority | Keyword rules |
| Live demo | Streamlit |
| Reflection | Included |
