import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ------------------------------
# Title
# ------------------------------
st.title("📧 Spam Email Detection")
st.write("Enter a message below to check whether it is Spam or Not Spam.")

# ------------------------------
# Load Dataset
# ------------------------------
df = pd.read_csv(
    r"C:\Users\hp\Desktop\Data Science course\mail_data.csv",
    encoding="latin1"
)

# ------------------------------
# Features and Target
# ------------------------------
X = df[['Message']]
y = df['Category']

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------------------
# Convert Text into Numbers
# ------------------------------
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train['Message'])
X_test = vectorizer.transform(X_test['Message'])

# ------------------------------
# Train Model
# ------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ------------------------------
# Model Accuracy
# ------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"{accuracy*100:.2f}%")

# ------------------------------
# User Input
# ------------------------------
st.subheader("Check Your Message")

user_message = st.text_area(
    "Type your email or SMS here:",
    height=150
)

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict"):

    if user_message.strip() == "":
        st.warning("Please enter a message.")
    else:

        input_data = vectorizer.transform([user_message])

        prediction = model.predict(input_data)

        if prediction[0] == "spam":
            st.error("🚨 This message is SPAM.")
        else:
            st.success("✅ This message is NOT SPAM.")