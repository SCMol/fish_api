from fastapi import FastAPI, UploadFile, File
from routes.image_analysis import analyze_image
from routes.species_id import id_species

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
    species_id = id_species(contents)
    return {"species_id": species_id}
