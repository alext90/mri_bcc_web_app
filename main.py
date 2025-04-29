import io

import torch
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image

import logging

from utils.utils import image_to_tensor, make_prediction
from utils.model import BCC_Model
from db.models.caseModel import CaseModel, Base
from fastapi import Depends
from sqlalchemy.orm import Session
from typing import List
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://postgres:password@db:5432/mri_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load model
model = BCC_Model(num_classes=3)
model.load_state_dict(torch.load("model/bcc_model.pth"))
model.eval()

# Mount static files (for CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/cases/", response_class=HTMLResponse)
def get_cases(request: Request, db: Session = Depends(get_db)):
    cases = db.query(CaseModel).all()
    return templates.TemplateResponse("cases.html", {"request": request, "cases": cases})

@app.post("/upload/")
async def upload_image(file: UploadFile, db: Session = Depends(get_db)):
    # Read the uploaded file
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        logger.info(f"File uploaded: {file.filename}")
    except Exception as e:
        logger.error(f"Error reading image file: {e}")
        return {"error": "Invalid image file"}
    
    input_tensor = image_to_tensor(image)
    predicted_class_name = make_prediction(input_tensor, model)
    logger.info(f"Predicted class: {predicted_class_name}")

    new_case = CaseModel(prediction=predicted_class_name)
    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    return {"predicted_class_name": predicted_class_name}
