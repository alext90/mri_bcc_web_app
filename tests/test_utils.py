import torch
from PIL import Image
from utils.utils import image_to_tensor

def test_image_to_tensor():
    image = Image.new("RGB", (224, 224))
    tensor = image_to_tensor(image)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)