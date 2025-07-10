import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="SMS Spam Classifier", layout="centered")
st.title("📩 SMS Spam Classifier (ML-Based)")
st.write("Upload your dataset, train models, and classify SMS messages as **Spam** or **Ham**.")

# -------- Clean Text Function --------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -------- Load and Prepare Data --------
@st.cache_data
def load_data(file):
    df = pd.read_csv(file, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df.dropna(inplace=True)
    df['message'] = df['message'].apply(clean_text)
    return df

uploaded_file = st.file_uploader("Upload spam.csv", type="csv")

if uploaded_file:
    try:
        df = load_data(uploaded_file)

        st.subheader("📊 Class Distribution (Before Oversampling)")
        st.bar_chart(df['label'].value_counts().rename(index={0: 'Ham', 1: 'Spam'}))

        X = df['message']
        y = df['label']

        vectorizer = TfidfVectorizer(stop_words='english')
        X_vec = vectorizer.fit_transform(X)

        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X_vec, y)

        st.subheader("🔁 Class Distribution (After Oversampling)")
        st.bar_chart(pd.Series(y_resampled).value_counts().rename(index={0: 'Ham', 1: 'Spam'}))

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
            'SVM': SVC(kernel='linear', probability=True, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced')
        }

        st.subheader("📈 Model Performance")
        trained_models = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            trained_models[name] = model

            st.markdown(f"### 🔍 {name} — Accuracy: {acc:.4f}")
            st.text(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], ax=ax)
            ax.set_title(f'Confusion Matrix - {name}')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

        st.subheader("🧪 Try It Yourself")
        user_input = st.text_area("Enter your SMS message here:")
        if user_input:
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])
            st.markdown("### 🤖 Model Predictions:")
            for name, model in trained_models.items():
                pred = model.predict(vec)[0]
                label = "🚫 Spam" if pred == 1 else "✅ Ham"
                st.write(f"- **{name}**: {label}")
    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
