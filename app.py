
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import pickle
import os
# from imblearn.over_sampling import RandomOverSampler

# Set page config
st.set_page_config(
    page_title="Spam/Ham Email Classifier",
    page_icon="📧",
    layout="wide"
)

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load and train models (optimized)
@st.cache_resource
def load_and_train_models():
    try:
        # Load the dataset
        df = pd.read_csv('spam.csv', encoding='latin-1')
        
        # Data cleaning
        df = df[['v1', 'v2']]
        df.columns = ['label', 'message']
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df = df.dropna()
        
        # Take smaller sample for faster processing
        df = df.sample(n=min(1000, len(df)), random_state=42)
        
        df['message'] = df['message'].apply(clean_text)
        
        # Preprocessing
        X = df['message']
        y = df['label']
        
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        X_vec = vectorizer.fit_transform(X)
        
        # Simple train/test split without oversampling
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
        
        # Reduced models for faster training
        models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=100, class_weight='balanced', solver='liblinear')
        }
        
        # Train models
        trained_models = {}
        model_accuracies = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            trained_models[name] = model
            model_accuracies[name] = accuracy
        
        return trained_models, vectorizer, model_accuracies, df
        
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

# Prediction function
def predict_message(message, models, vectorizer):
    message_clean = clean_text(message)
    message_vec = vectorizer.transform([message_clean])
    predictions = {}
    probabilities = {}
    
    for name, model in models.items():
        pred = model.predict(message_vec)[0]
        prob = model.predict_proba(message_vec)[0][1]  # Probability of spam
        predictions[name] = 'Spam' if pred == 1 else 'Ham'
        probabilities[name] = prob
    
    return predictions, probabilities

# Main app
def main():
    st.title("📧 Spam/Ham Email Classifier")
    st.markdown("---")
    
    # Load models with error handling
    models, vectorizer, accuracies, df = load_and_train_models()
    
    if models is None:
        st.error("Failed to load models. Please check your dataset.")
        return
    
    # Sidebar
    st.sidebar.title("📊 Model Performance")
    for model_name, accuracy in accuracies.items():
        st.sidebar.metric(model_name, f"{accuracy:.3f}")
    
    # Data overview
    st.sidebar.title("📈 Dataset Overview")
    st.sidebar.write(f"Total messages: {len(df)}")
    st.sidebar.write(f"Ham messages: {len(df[df['label'] == 0])}")
    st.sidebar.write(f"Spam messages: {len(df[df['label'] == 1])}")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🔍 Classify Your Message")
        
        # Text input
        user_message = st.text_area(
            "Enter your message here:",
            height=150,
            placeholder="Type your email message here to check if it's spam or ham..."
        )
        
        if st.button("Classify Message", type="primary"):
            if user_message.strip():
                predictions, probabilities = predict_message(user_message, models, vectorizer)
                
                st.subheader("📋 Classification Results")
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Model': list(predictions.keys()),
                    'Prediction': list(predictions.values()),
                    'Spam Probability': [f"{prob:.3f}" for prob in probabilities.values()]
                })
                
                st.dataframe(results_df, use_container_width=True)
                
                # Consensus prediction
                spam_count = sum(1 for pred in predictions.values() if pred == 'Spam')
                consensus = "Spam" if spam_count >= 2 else "Ham"
                
                if consensus == "Spam":
                    st.error(f"🚨 **Consensus: SPAM** ({spam_count}/4 models agree)")
                else:
                    st.success(f"✅ **Consensus: HAM** ({4-spam_count}/4 models agree)")
                
                # Probability chart
                st.subheader("📊 Spam Probability by Model")
                prob_df = pd.DataFrame({
                    'Model': list(probabilities.keys()),
                    'Spam Probability': list(probabilities.values())
                })
                
                fig, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(prob_df['Model'], prob_df['Spam Probability'], 
                             color=['red' if p > 0.5 else 'green' for p in prob_df['Spam Probability']])
                ax.set_ylabel('Spam Probability')
                ax.set_title('Spam Probability by Model')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("Please enter a message to classify.")
    
    with col2:
        st.header("ℹ️ About")
        st.write("""
        This app uses 4 different machine learning models to classify emails as spam or ham:
        
        - **Naive Bayes**: Probabilistic classifier
        - **Logistic Regression**: Linear classifier
        - **SVM**: Support Vector Machine
        - **Random Forest**: Ensemble method
        
        The app shows individual predictions and a consensus result.
        """)
        
        st.header("🛠️ Features")
        st.write("""
        - Real-time classification
        - Multiple model comparison
        - Probability visualization
        - Consensus prediction
        - Interactive interface
        """)

if __name__ == "__main__":
    main()
