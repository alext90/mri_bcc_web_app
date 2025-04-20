import io

import torch
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import models

import logging

from utils.utils import image_to_tensor, make_prediction
from utils.model import BCC_Model

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load model
base_model = models.mobilenet_v2(pretrained=False)
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


@app.post("/upload/")
async def upload_image(file: UploadFile):
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
    return {"predicted_class_name": predicted_class_name}
