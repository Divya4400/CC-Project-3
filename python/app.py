from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class Recommendation(BaseModel):
    description: str
    music_url: str

@app.get('/recommendations')
async def get_recommendations(text: str):
    # Your code to generate recommendations goes here
    recommendations = [
        Recommendation(description="A calm and mellow piano piece with a simple melody.", 
                       music_url="https://www2.cs.uic.edu/~i101/SoundFiles/CantinaBand60.wav"),
        Recommendation(description="An upbeat pop song with a catchy guitar riff and female vocals.", 
                       music_url="https://www2.cs.uic.edu/~i101/SoundFiles/ImperialMarch60.wav")
    ]
    return {"recommendations": recommendations}

@app.exception_handler(Exception)
async def error_handler(request, exc):
    return {"error": str(exc)}, 500