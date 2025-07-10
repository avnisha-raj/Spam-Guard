from google.colab import files
uploaded = files.upload()

# Import libraries
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

# For oversampling
from imblearn.over_sampling import RandomOverSampler

# Load the dataset (assumes you uploaded spam.csv)
df = pd.read_csv('spam.csv', encoding='latin-1')

# Data cleaning: Keep only the necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Map labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Remove missing values
df = df.dropna()

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['message'] = df['message'].apply(clean_text)

# EDA: Show class distribution
print("Label distribution before balancing:")
print(df['label'].value_counts())

# Plot class distribution
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham (Before Balancing)')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()

# Preprocessing: TF-IDF Vectorization
X = df['message']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

# Oversample the minority class
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_vec, y)

# Show new class distribution
print("Label distribution after balancing:")
print(pd.Series(y_resampled).value_counts())

# Plot new class distribution
sns.countplot(x=y_resampled)
plt.title('Spam vs Ham (After Oversampling)')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.show()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Model building (using class_weight='balanced' where supported)
models = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
    'SVM': SVC(kernel='linear', probability=True, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced')
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\nModel: {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# User-interactive prediction
print("\n--- Spam/Ham Message Classifier ---")
print("Type your message and press Enter to classify. Type 'exit' to quit.\n")

def predict_message(message):
    message_clean = clean_text(message)
    message_vec = vectorizer.transform([message_clean])
    predictions = {}
    for name, model in models.items():
        pred = model.predict(message_vec)[0]
        predictions[name] = 'Spam' if pred == 1 else 'Ham'
    return predictions

while True:
    user_input = input("Enter a message: ")
    if user_input.lower() == 'exit':
        print("Exiting classifier. Goodbye!")
        break
    preds = predict_message(user_input)
    print("Predictions:")
    for model_name, pred in preds.items():
        print(f"  {model_name}: {pred}")
    print()



