import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Spam Detector", page_icon="📧")

st.title("📧 Spam Email Detector")
st.write("This app predicts whether a message is **Spam** or **Not Spam**.")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only the first two columns
df = df.iloc[:, :2]
df.columns = ["Category", "Message"]

# -----------------------------
# FEATURES & TARGET
# -----------------------------
X = df["Message"]
y = df["Category"].map({"ham": 0, "spam": 1})

# -----------------------------
# TEXT VECTORIZATION
# -----------------------------
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# -----------------------------
# TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# MODEL ACCURACY
# -----------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("📊 Model Accuracy")
st.success(f"Accuracy: {accuracy * 100:.2f}%")

# -----------------------------
# VIEW DATASET
# -----------------------------
with st.expander("📂 View Dataset"):
    st.dataframe(df)

# -----------------------------
# SPAM DETECTOR
# -----------------------------
st.subheader("✉️ Check Your Message")

message = st.text_area(
    "Enter your message:",
    key="message_input"
)

if st.button("Predict"):

    if message.strip() == "":
        st.warning("Please enter a message.")

    else:
        message_vector = vectorizer.transform([message])

        prediction = model.predict(message_vector)[0]
        confidence = model.predict_proba(message_vector).max() * 100

        if prediction == 1:
            st.error(f"🚨 SPAM\n\nConfidence: {confidence:.2f}%")
        else:
            st.success(f"✅ NOT SPAM\n\nConfidence: {confidence:.2f}%")

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
with st.expander("📈 Classification Report"):
    report = classification_report(y_test, y_pred)
    st.text(report)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
with st.expander("📉 Confusion Matrix"):
    matrix = confusion_matrix(y_test, y_pred)
    st.write(matrix)