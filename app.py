import re
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(
    page_title="Auto Ticket Categorizer",
    page_icon="🎫",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0px;
}
.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model():
    model = joblib.load("models/ticket_classifier.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vectorizer

@st.cache_data
def load_dataset():
    return pd.read_csv("data/tickets.csv")

model, vectorizer = load_model()
df = load_dataset()

st.markdown(
    '<div class="main-title">🎫 Auto Email / Ticket Categorizer</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered support ticket routing using TF-IDF + Logistic Regression</div>',
    unsafe_allow_html=True
)

st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider(
    "Human Review Threshold",
    min_value=0.50,
    max_value=0.90,
    value=0.60,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.write("### Categories")
for category in sorted(df["category"].unique()):
    st.sidebar.write(f"• {category}")

X = df["text"].apply(clean_text)
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy * 100:.1f}%")
col2.metric("Precision", f"{precision * 100:.1f}%")
col3.metric("Recall", f"{recall * 100:.1f}%")
col4.metric("F1 Score", f"{f1 * 100:.1f}%")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🚀 Live Classifier", "📊 Model Evaluation", "🧪 Test Examples"]
)

with tab1:
    st.subheader("Classify a New Support Ticket")

    ticket_text = st.text_area(
        "Enter ticket / email text",
        height=180,
        placeholder="Example: My payment was charged twice and I need a refund."
    )

    classify_button = st.button(
        "🔍 Classify Ticket",
        type="primary",
        use_container_width=True
    )

    if classify_button:
        if not ticket_text.strip():
            st.warning("Please enter a ticket before classification.")
        else:
            cleaned = clean_text(ticket_text)
            vectorized = vectorizer.transform([cleaned])
            probabilities = model.predict_proba(vectorized)[0]

            predicted_index = np.argmax(probabilities)
            predicted_category = model.classes_[predicted_index]
            confidence = probabilities[predicted_index]

            urgent_keywords = [
                "urgent", "asap", "immediately", "down", "not working", "broken",
                "crash", "crashing", "locked out", "can't work", "cannot work",
                "emergency", "critical", "blocked", "right now", "as soon as possible"
            ]

            text_lower = ticket_text.lower()
            is_urgent = any(
                keyword in text_lower for keyword in urgent_keywords
            )
            priority = "URGENT" if is_urgent else "NORMAL"
            needs_review = confidence < confidence_threshold

            st.markdown("---")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.metric("Predicted Category", predicted_category)
                st.metric("Confidence", f"{confidence * 100:.2f}%")

            with result_col2:
                st.metric("Priority", priority)
                if needs_review:
                    st.error("⚠️ Needs Human Review")
                else:
                    st.success("✅ Safe for Automatic Routing")

            st.subheader("Category Probability")

            probability_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities * 100
            })

            st.bar_chart(probability_df.set_index("Category"))

with tab2:
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Classification Report")

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(3),
            use_container_width=True
        )

    with col2:
        st.write("### Confusion Matrix")

        cm = confusion_matrix(
            y_test, y_pred, labels=model.classes_
        )

        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    st.markdown("---")
    st.write("### Dataset Distribution")
    st.bar_chart(df["category"].value_counts())

with tab3:
    st.subheader("Five New Sample Tickets")

    sample_tickets = [
        "I was charged twice for my monthly subscription.",
        "The application keeps crashing whenever I try to login.",
        "How many vacation days can I take this year?",
        "Can you tell me where I can find the user guide?",
        "The server is completely down and our team cannot access the system."
    ]

    for i, ticket in enumerate(sample_tickets, start=1):
        vectorized = vectorizer.transform([clean_text(ticket)])
        probabilities = model.predict_proba(vectorized)[0]
        index = np.argmax(probabilities)

        category = model.classes_[index]
        confidence = probabilities[index]

        urgent_words = [
            "urgent", "critical", "down",
            "not working", "blocked", "failed"
        ]

        priority = (
            "URGENT"
            if any(word in ticket.lower() for word in urgent_words)
            else "NORMAL"
        )

        needs_review = confidence < confidence_threshold

        with st.expander(f"Ticket {i}: {ticket}"):
            col1, col2, col3 = st.columns(3)
            col1.write(f"**Category:** {category}")
            col2.write(f"**Confidence:** {confidence * 100:.2f}%")
            col3.write(f"**Priority:** {priority}")

            if needs_review:
                st.warning("Needs Human Review")
            else:
                st.success("Automatically routable")

st.markdown("---")
st.subheader("💡 Reflection")
st.write("""
With more labeled data, I would improve the model's ability to handle
ambiguous tickets and reduce false classifications. I would compare
Logistic Regression with Naive Bayes and Linear SVM using cross-validation.
For production, I would monitor confidence, class imbalance, data drift,
misclassifications, and periodically retrain using newly human-reviewed tickets.
""")
