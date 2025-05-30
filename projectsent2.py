import streamlit as st
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app configuration
st.set_page_config(page_title="Sentiment Classifier", page_icon="😊")

# Title and description
st.title("Sentiment Classifier ")
st.write("Enter a piece of text to analyze its sentiment (positive, neutral, or negative).")

# Text input
user_input = st.text_area("Type your review:", height=150)

# Predict sentiment and display results
if st.button("Analyze Sentiment"):
    if user_input:
        # Get sentiment scores
        scores = analyzer.polarity_scores(user_input)
        
        # Determine dominant sentiment
        compound = scores['compound']
        if compound >= 0.05:
            sentiment = "Positive"
        elif compound <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        # Display results
        st.subheader("Results")
        st.write(f"**Dominant Sentiment**: {sentiment}")
        st.write(f"**Sentiment Scores**:")
        st.write(f"- Positive: {scores['pos']:.3f}")
        st.write(f"- Neutral: {scores['neu']:.3f}")
        st.write(f"- Negative: {scores['neg']:.3f}")
        st.write(f"- Compound: {scores['compound']:.3f}")
        
        # Create bar chart for sentiment scores
        fig, ax = plt.subplots()
        sentiments = ['Positive', 'Neutral', 'Negative']
        values = [scores['pos'], scores['neu'], scores['neg']]
        ax.bar(sentiments, values, color=['#36A2EB', '#FFCE56', '#FF6384'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Sentiment Distribution')
        st.pyplot(fig)
    else:
        st.error("Please enter some text to analyze.")
        