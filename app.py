from fastapi import FastAPI, UploadFile, File
from routes.image_analysis import analyze_image
import os

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Fish API"}

@app.post("/analyze-image")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_image(contents)
    return result
