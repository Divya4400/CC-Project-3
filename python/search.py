from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import pickle
import os
from middleware import add_middleware

from preprocess_data import preprocess_text

app = FastAPI()

# add middleware
add_middleware(app)

class TextInput(BaseModel):
    text: str

# Load the preprocessed and vectorized data
with open('preprocessed_vectorized_data.pkl', 'rb') as f:
    data = pickle.load(f)

df = data[0]

def find_similar_caption(input_text):
    # Extract the vectorizer and the vectorized captions
    vectorizer = data[1]
    vectorized_captions = data[2]

    # Get the most similar caption
    most_similar_caption, similarity_score, most_similar_idx = get_most_similar_caption(input_text, vectorizer, vectorized_captions, df['caption'])
    
    # Return the most similar caption, similarity score, and ytid
    return most_similar_caption, similarity_score, df.loc[most_similar_idx, 'ytid']

def get_most_similar_caption(input_text, vectorizer, vectorized_data, captions):
    # Vectorize the input text
    vectorized_input = vectorizer.transform([input_text])
    
    # Calculate the cosine similarity between the input text and all captions
    similarity_scores = cosine_similarity(vectorized_input, vectorized_data)
    
    # Find the index of the most similar caption
    most_similar_idx = np.argmax(similarity_scores)
    
    # Get the most similar caption and its similarity score
    most_similar_caption = captions.iloc[most_similar_idx]
    similarity_score = similarity_scores[0, most_similar_idx]
    
    return most_similar_caption, similarity_score, most_similar_idx

def get_mp3_path(ytid, folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.startswith(ytid) and file_name.endswith('.mp3'):
            return os.path.abspath(os.path.join(folder_path, file_name))
    return None

""" input_text = "menacing grunts"
most_similar_caption, similarity_score, ytid = find_similar_caption(input_text)
print("Most similar caption:", most_similar_caption)
print("Cosine similarity score:", similarity_score)
print("YTID:", ytid)
print(get_mp3_path(ytid, "music_data")) """

@app.get("/recommendations")
def get_recommendations(text: str):
    most_similar_caption, similarity_score, ytid = find_similar_caption(text)
    mp3_path = get_mp3_path(ytid, "music_data")
    
    recommendations = [
        {
            "description": most_similar_caption,
            "music_url": mp3_path
        }
    ]
    
    return {"recommendations": recommendations}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)