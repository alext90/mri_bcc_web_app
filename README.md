# MRI Brain Cancer Classification Web App

This is a FastAPI web application that allows users to upload MRI scans and classify them into one of three classes using a PyTorch model. The three classes are:
- **Glioma**
- **Meningioma**
- **Pituitary**

The app provides a simple interface to upload an image, displays the uploaded image, and shows the predicted class.

## Features
- Upload MRI images in `.jpg`, `.png`, or other common formats.
- Classify the image into one of three classes using a pre-trained PyTorch model.
- Display the uploaded image and the prediction result on the same page.

## Requirements
- Python 3.11 or higher
- FastAPI
- Uvicorn
- PyTorch
- torchvision
- Pillow

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mri-classification-webapp.git
   cd mri-classification-webapp```

2. Create a virtual environment and activate it
```uv run main.py```

3. Copy your pre-trained model into the model folder

## Running the App
1. Start the FastAPI server:
`uvicorn main:app --reload`

2. Open your browser and navigate to:
`http://127.0.0.1:8000`

3. Upload an MRI image and view the prediction result

## Project Structure
```
dl_web_app/
├── main.py                 # FastAPI application
├── utils/
│   ├── utils.py            # Helper functions (e.g., image transforms, predictions)
│   ├── model.py            # Custom PyTorch model definition
├── model/
│   └── bcc_model.pth       # Pre-trained PyTorch model
├── templates/
│   └── index.html          # HTML template for the web interface
├── static/
│   └── style.css           # CSS for styling the web interface
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```