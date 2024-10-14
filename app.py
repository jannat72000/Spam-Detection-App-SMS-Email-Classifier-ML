import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()
tokenizer = TreebankWordTokenizer()

def transform_text(text):
    text = text.lower()
    text = tokenizer.tokenize(text)

    y = [i for i in text if i.isalnum()]

    filtered_words = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]

    # Apply stemming
    stemmed_words = [ps.stem(i) for i in filtered_words]

    return " ".join(stemmed_words)

# Load the vectorizer and model
try:
    tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    st.write("Model and vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    st.stop()

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    try:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)
        st.write(f"Transformed SMS: {transformed_sms}")

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])
        st.write("Vectorization complete.")

        # 3. Predict
        result = model.predict(vector_input)[0]
        st.write("Prediction complete.")

        # 4. Display
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
