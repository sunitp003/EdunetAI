Email Spam Detection with Machine Learning
Overview
This project aims to build a spam email detection system using machine learning techniques in Python. The goal is to classify emails as spam or legitimate (ham) by training a machine learning model. The dataset used for this project contains a collection of SMS messages labeled as spam or ham.

Steps Involved
1. Import Libraries
We use various Python libraries like pandas, sklearn, and nltk for data manipulation, model training, and evaluation.

python
Copy code
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
2. Load Dataset
The dataset is loaded into a pandas DataFrame. We remove unnecessary columns and handle any duplicate or missing data.

python
Copy code
df = pd.read_csv("/path/to/spam.csv", encoding="ISO-8859-1")
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)  # Drop unnecessary columns
df = df.drop_duplicates()  # Remove duplicates
3. Preprocessing
The "Category" column is converted to numeric values: 1 for ham and 0 for spam.

python
Copy code
df["Category"] = df["v1"].map({"ham": 1, "spam": 0})
X = df['v2']  # Email content
y = df['Category']  # Labels
4. Feature Extraction
The text data is transformed into numerical features using TF-IDF vectorization.

python
Copy code
feature_extraction = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)
5. Model Training
A Logistic Regression model is trained using the training data.

python
Copy code
model = LogisticRegression()
model.fit(X_train_features, y_train)
6. Model Evaluation
The model's performance is evaluated using accuracy, confusion matrix, and classification report.

python
Copy code
accuracy_on_test_data = accuracy_score(y_test, model.predict(X_test_features))
conf_matrix = confusion_matrix(y_test, model.predict(X_test_features))
classification_rep = classification_report(y_test, model.predict(X_test_features))
7. Make Predictions
Finally, we use the trained model to classify new email inputs as spam or ham.

python
Copy code
input_mail = "Congratulations! You've won a prize!"
input_data_features = feature_extraction.transform([input_mail])
prediction = model.predict(input_data_features)
Results
Accuracy on test data: 96%
Precision: 95.8%
Recall: 99.8%
F1-Score: 97.5%
Example Prediction
python
Copy code
Spam Mail
Requirements
Python 3.x
Libraries: numpy, pandas, sklearn, matplotlib, seaborn, nltk
You can install the required libraries using:

bash
Copy code
pip install -r requirements.txt
Conclusion
This project demonstrates the use of machine learning techniques to effectively classify emails as spam or ham. With an accuracy of 96%, the model performs well in detecting spam emails.
