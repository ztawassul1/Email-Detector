import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Data
df = pd.read_csv("mail_data.csv", encoding="latin1")
# Features and Target
X = df["Message"]
y = df["Category"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert Text
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("📧 Spam Email Detector")
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

message = st.text_area("Enter your message")

if st.button("Predict"):
    if message == "":
        st.warning("Please enter a message.")
    else:
        data = vectorizer.transform([message])
        result = model.predict(data)

        if result[0] == "spam":
            st.error("🚨 Spam Message")
        else:
            st.success("✅ Not Spam")

