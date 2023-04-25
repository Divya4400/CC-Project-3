import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load the MusicCaps dataset
df = pd.read_csv('musiccaps.csv')

nltk.download('stopwords')
nltk.download('punkt')

# Define a function to preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stop words and punctuation
    stop_words = set(stopwords.words('english'))
    words = [word for word in tokens if word.isalnum() and word not in stop_words]
    # Join the words back into a string
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Preprocess the captions
df['caption'] = df['caption'].apply(preprocess_text)

# Define the vectorizer
vectorizer = TfidfVectorizer()

# Vectorize the captions
vectorized_captions = vectorizer.fit_transform(df['caption'])

# Save the preprocessed and vectorized data to file
with open('preprocessed_vectorized_data.pkl', 'wb') as f:
    pickle.dump((df, vectorizer, vectorized_captions), f)
