
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
                with st.spinner("Classifying..."):
                    try:
                        predictions, probabilities = predict_message(user_message, models, vectorizer)
                        
                        st.subheader("📋 Classification Results")
                        
                        # Show results immediately
                        for model_name, prediction in predictions.items():
                            if prediction == "Spam":
                                st.error(f"🚨 **{model_name}**: {prediction} (Probability: {probabilities[model_name]:.3f})")
                            else:
                                st.success(f"✅ **{model_name}**: {prediction} (Probability: {probabilities[model_name]:.3f})")
                        
                        # Quick consensus without complex charts
                        spam_count = sum(1 for pred in predictions.values() if pred == 'Spam')
                        consensus = "Spam" if spam_count >= 1 else "Ham"  # Changed from >= 2 to >= 1
                        
                        st.markdown("---")
                        if consensus == "Spam":
                            st.error(f"🚨 **Final Result: SPAM** ({spam_count}/{len(predictions)} models agree)")
                        else:
                            st.success(f"✅ **Final Result: HAM** ({len(predictions)-spam_count}/{len(predictions)} models agree)")
                            
                    except Exception as e:
                        st.error(f"Error during classification: {str(e)}")
                        
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
