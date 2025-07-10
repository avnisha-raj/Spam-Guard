# SMS Spam Classifier (Machine Learning Project)

A machine learning-based text classification system that classifies SMS messages as Spam or Ham using multiple ML models. This project is built using Python and executed in Google Colab.

---
## Features

- Cleaned and preprocessed real-world SMS dataset (`spam.csv`)
- Text vectorization using **TF-IDF**
-  Handled class imbalance using **RandomOverSampler**
- Trained and compared **4 classifiers**:
  - Naive Bayes
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
- Evaluated models using:
  - Accuracy
  - Classification report
  - Confusion matrix
-  Interactive CLI input to test real-time message predictions

---

## Dataset
- Source: [UCI SMS Spam Collection Dataset]([https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- File: `spam.csv`
- Contains ~5,574 SMS labeled as `ham` or `spam`

---
##  Tech Stack
- Python
- Pandas, NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib, Seaborn
- Google Colab (development)

---

##  How to Run the Project
1. Clone this repository or download the files
2. Install dependencies:

```bash
pip install -r requirements.txt

Run the script:
app.py
Enter your message to test real-time predictions
(Type exit to quit the CLI)

