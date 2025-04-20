from torchvision import transforms
import torch
from PIL import Image

IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN, NORMALIZATION_STD = [0.5], [0.5]

CLASS_MAP = {
    0: "Glioma",
    1: "Menin",
    2: "Tumor",
}

transform = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
    ]
)

def image_to_tensor(image: Image) -> torch.Tensor:
    """
    Convert a PIL image to a tensor and apply transformations.
    Args:
        image (PIL.Image): The input image.
    Returns:
        torch.Tensor: The transformed image tensor.
    """
    transformed_image = transform(image)
    return transformed_image.unsqueeze(0)


def make_prediction(input_tensor: torch.Tensor, model: torch.nn.Module) -> str:
    """
    Make a prediction using the model.
    Args:
        input_tensor (torch.Tensor): The input tensor representing the image.
        model (torch.nn.Module): The trained model.
    Returns:
        str: The predicted class name.
    """
    with torch.no_grad():
        output = model(input_tensor)
        predicted_class_idx = torch.argmax(output, dim=1).item()

    predicted_class_name = CLASS_MAP.get(predicted_class_idx, "Unknown")
    return predicted_class_name