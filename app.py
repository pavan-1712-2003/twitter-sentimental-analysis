import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Text cleaning setup
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("Twitter Sentiment Classifier")

st.write("Type a tweet below and the model will classify it as Positive, Negative, or Neutral.")

tweet = st.text_area("Tweet")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        cleaned = clean_text(tweet)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]

        st.subheader("Sentiment")
        st.success(result)
