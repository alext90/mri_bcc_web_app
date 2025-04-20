import io

import torch
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from torchvision import models

from utils.utils import image_to_tensor, make_prediction
from utils.model import BCC_Model

app = FastAPI()


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
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")  # Convert to RGB if needed
    input_tensor = image_to_tensor(image)

    predicted_class_name = make_prediction(input_tensor, model)
    return {"predicted_class_name": predicted_class_name}
