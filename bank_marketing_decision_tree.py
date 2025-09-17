import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import urllib.request

# Step 1: Download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
zip_path = "bank-additional.zip"
data_folder = "bank-additional"

# Download zip file
urllib.request.urlretrieve(url, zip_path)

import zipfile
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_folder)

# Load the dataset (using bank-additional/bank-additional-full.csv)
data_path = f"{data_folder}/bank-additional/bank-additional-full.csv"
df = pd.read_csv(data_path, sep=';')

print("Dataset loaded. Sample:")
print(df.head())

# Step 2: Preprocessing
# Encode target variable 'y'
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Encode categorical columns using LabelEncoder for simplicity
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('y', axis=1)
y = df['y']

# Split into train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Train Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Save the model
import joblib
joblib.dump(clf, "decision_tree_bank_marketing_model.joblib")

print("Model saved as 'decision_tree_bank_marketing_model.joblib'")
