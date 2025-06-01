import streamlit as st
import pandas as pd
import numpy as np
import string
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
import pickle
import os

# Force download NLTK stopwords
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
        st.info("NLTK resource 'stopwords' found.")
    except LookupError:
        st.warning("Downloading NLTK resource: stopwords")
        try:
            nltk.download('stopwords', quiet=True, raise_on_error=True)
            st.success("Successfully downloaded stopwords")
        except Exception as e:
            st.error(f"Failed to download stopwords: {str(e)}. Please check your internet or delete C:\\Users\\DELL\\AppData\\Roaming\\nltk_data.")
            return False
    return True

# Call the download function
if not download_nltk_resources():
    st.error("Cannot proceed without NLTK stopwords. Try deleting C:\\Users\\DELL\\AppData\\Roaming\\nltk_data and re-running.")
    st.stop()

# Get stopwords
stop_words = stopwords.words('english')

# Function to clean text (from your code)
def clean_text(text):
    text = "".join([char.lower() for char in str(text) if char not in string.punctuation])
    tokens = re.split(r'\W+', text)
    text = [word for word in tokens if word not in stop_words]
    return text

# Function to load or train the model
def load_or_train_model():
    model_file = 'rf_spam_classifier_model.pkl'
    vectorizer_file = 'tfidf_vectorizer.pkl'

    # Check if model and vectorizer exist
    if os.path.exists(model_file) and os.path.exists(vectorizer_file):
        try:
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            with open(vectorizer_file, 'rb') as f:
                vectorizer = pickle.load(f)
            return model, vectorizer
        except Exception as e:
            st.warning(f"Failed to load model or vectorizer: {str(e)}. Retraining model...")

    # Balanced in-code dataset
    data = {
        'text': [
            "Win a free iPhone now!", "Claim your prize today!", "Free money for you!", "Exclusive offer, sign up now!",
            "Hey, how are you today?", "Meeting at 3 PM tomorrow", "Can we discuss the project?", "Lunch plans this weekend?"
        ],
        'label': ['spam', 'spam', 'spam', 'spam', 'ham', 'ham', 'ham', 'ham']
    }
    messages = pd.DataFrame(data)

    # Vectorize text using TF-IDF
    tfidf_vect = TfidfVectorizer(analyzer=clean_text, max_features=1000, ngram_range=(1, 2), min_df=1)
    X_tfidf = tfidf_vect.fit_transform(messages['text'])
    X_features = pd.DataFrame(X_tfidf.toarray())

    # Train Random Forest with balanced class weights
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_model = rf.fit(X_features, messages['label'])

    # Evaluate model on the same data (small dataset)
    y_pred = rf_model.predict(X_features)
    accuracy = accuracy_score(messages['label'], y_pred)
    precision = precision_score(messages['label'], y_pred, pos_label='spam', zero_division=0)
    recall = recall_score(messages['label'], y_pred, pos_label='spam', zero_division=0)
    f1 = f1_score(messages['label'], y_pred, pos_label='spam', zero_division=0)
    cm = confusion_matrix(messages['label'], y_pred, labels=['ham', 'spam'])

    st.write("**Model Performance on Training Data:**")
    st.write(f"- Accuracy: {accuracy:.3f}")
    st.write(f"- Precision (Spam): {precision:.3f}")
    st.write(f"- Recall (Spam): {recall:.3f}")
    st.write(f"- F1-Score (Spam): {f1:.3f}")
    st.write("**Confusion Matrix (Ham, Spam):**")
    st.write(cm)

    # Save model and vectorizer
    try:
        with open(model_file, 'wb') as f:
            pickle.dump(rf_model, f)
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(tfidf_vect, f)
    except Exception as e:
        st.warning(f"Failed to save model or vectorizer: {str(e)}")

    return rf_model, tfidf_vect

# Streamlit app
st.title("Spam/Ham Classifier")
st.write("Enter messages to classify as Spam or Ham. Uses Random Forest with balanced class weights.")

# Load or train model
model, vectorizer = load_or_train_model()

# Single message input
st.subheader("Classify a Single Message")
user_input = st.text_area("Enter a message:", height=100, value="call me as soon as you reach home")

if st.button("Classify Single Message"):
    if user_input:
        if model is None or vectorizer is None:
            st.error("Model could not be loaded or trained.")
        else:
            try:
                # Transform input
                text_tfidf = vectorizer.transform([user_input])
                # Predict
                prediction = model.predict(text_tfidf)[0]
                # Get probabilities
                prob = model.predict_proba(text_tfidf)[0]
                prob_spam = prob[model.classes_.tolist().index('spam')]
                prob_ham = prob[model.classes_.tolist().index('ham')]
                # Display result
                st.write(f"**Prediction: {prediction.upper()}**")
                st.write(f"**Probability (Spam): {prob_spam:.2%}**")
                st.write(f"**Probability (Ham): {prob_ham:.2%}**")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please enter a message to classify.")

# Manual testing section for multiple entries
st.subheader("Test Multiple Messages")
st.write("Enter multiple messages (one per line) to test predictions:")
test_inputs = st.text_area("Enter test messages (one per line):", height=150, value="call me as soon as you reach home\nWin a free iPhone now!\nMeeting at 3 PM tomorrow\nClaim your prize today!")

if st.button("Classify Test Messages"):
    if test_inputs:
        if model is None or vectorizer is None:
            st.error("Model could not be loaded or trained.")
        else:
            try:
                # Split inputs by line
                messages = [msg.strip() for msg in test_inputs.split('\n') if msg.strip()]
                if not messages:
                    st.error("No valid messages entered.")
                else:
                    # Transform inputs
                    text_tfidf = vectorizer.transform(messages)
                    # Predict
                    predictions = model.predict(text_tfidf)
                    # Get probabilities
                    probs = model.predict_proba(text_tfidf)
                    prob_spam = [prob[model.classes_.tolist().index('spam')] for prob in probs]
                    prob_ham = [prob[model.classes_.tolist().index('ham')] for prob in probs]

                    # Create results DataFrame
                    results = pd.DataFrame({
                        'Message': messages,
                        'Prediction': [pred.upper() for pred in predictions],
                        'Spam Probability': [f"{p:.2%}" for p in prob_spam],
                        'Ham Probability': [f"{p:.2%}" for p in prob_ham]
                    })

                    # Display results
                    st.write("**Test Results:**")
                    st.dataframe(results, use_container_width=True)
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please enter at least one message to classify.")