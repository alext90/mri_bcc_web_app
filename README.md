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
- Python 3.12 or higher
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

## Running the App Locally
1. Start the FastAPI server:  
```
uvicorn main:app --reload
```

2. Open your browser and navigate to:  
```
http://127.0.0.1:8000
```

3. Upload an MRI image and view the prediction result

## Running the App with Docker
1. Build the Docker image:  
```
docker build -t mri-classification-app .
```
2. Run the Docker container:  
```
docker run -d -p 8000:8000 --name mri-classification-container mri-classification-app
```
3. Open your browser and navigate to:  
```
http://127.0.0.1:8000
```
4. To stop the container:  
```
docker stop mri-classification-container
```
5. To remove the container:  
```
docker rm mri-classification-container
```


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