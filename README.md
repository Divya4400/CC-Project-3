# CC-Project-3

## BACKEND

1. The first step is to run the download_musiccaps.py file:

The code loads the MusicCaps dataset using the load_dataset function from the datasets library, selects the desired number of examples if a limit is given, creates a directory to save the downloaded clips, and processes each example in the dataset using the process function. The process function downloads the audio clip snippet from YouTube and saves it in the directory, updates the example with the path to the saved clip, and returns the example, which is stored locally in MP3 format in a folder called music_data.

2. The next step is to run preprocess_data.py:

This code is responsible for preprocessing the MusicCaps dataset for use in the recommendation system. It starts by loading the dataset from a CSV file (musiccaps.csv) and downloading the necessary NLTK resources. Then, it defines a function to preprocess the text data by tokenizing the text, removing stop words and punctuation, and joining the remaining words back into a string. Next, it applies this preprocessing function to the captions in the dataset. After preprocessing, the code defines a TfidfVectorizer to convert the preprocessed captions into feature vectors. The vectorizer uses the inverse document frequency (IDF) weighting scheme to down-weight words that occur frequently in the corpus. It then applies the vectorizer to the preprocessed captions to obtain a sparse matrix of vectorized captions. Finally, the preprocessed and vectorized data are saved to a file (preprocessed_vectorized_data.pkl) using the pickle library for later use in the recommendation system. The resulting file contains the preprocessed dataset, the trained vectorizer, and the vectorized captions.

3. The main file that provides the API for the user input is search.py:

It loads preprocessed and vectorized data from the pickle file, and defines a function called 'get_most_similar_caption', which takes an input text from the user, a vectorizer, the vectorized data, and captions as input, and returns the most similar caption, similarity score, and index. With that information the system searches a local folder for the respective MP3 file, and returns its path if found. This functionality is provided to the frontend as an API endpoint using the FastAPI framework called '/recommendations', which accepts a GET request with a required parameter called 'text'and returns a JSON response with a recommended MP3 file, and caption.

Execute this file to run uvicorn and start a webserver in your localhost.

## FRONTEND

The frontend is designed using HTML and CSS. Javascript is used to make the API call and to render the music player based on the response obtained.

A form is created and an EventListener is added to it, to track when the button is clicked. The search text is extracted from the input field and the system calls the API `http://localhost:8000/recommendations?text=${searchText}`. Is the status code is 200, the code will get the json response and render the code accordingly.