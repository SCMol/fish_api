from fastapi import FastAPI, UploadFile, File
from routes.image_analysis import analyze_image
from routes.species_id import id_class
from routes.disease_id import analyze_disease

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to Fish API"}

@app.post("/analyze-image")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    result = analyze_image(contents)
    return result

@app.post("/id-species")
async def id(file: UploadFile = File(...)):
    contents = await file.read()
    species_id = id_class(contents)
    return {"species_id": species_id}

@app.post("/analyze-disease")
async def analyze_disease_route(file: UploadFile = File(...)):
    contents = await file.read()
    disease_result = analyze_disease(contents)
    return disease_result
