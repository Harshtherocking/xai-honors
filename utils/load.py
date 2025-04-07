import torch
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_B6_Weights

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms.functional import to_pil_image



DATASET_PATH = "imagenet-1000"

# def load_vision_model(model_name: str):
#     """
#     Load a vision model from PyTorch Hub.

#     Args:
#         model_name (str): The name of the model to load.

#     Returns:
#         torch.nn.Module: The loaded model.
#     """
#     try:
#         model = torch.hub.load('pytorch/vision', model_name, pretrained=True)
#         model.eval()  # Set the model to evaluation mode
#         return model
#     except Exception as e:
#         raise ValueError(f"Error loading model '{model_name}': {e}")

def resnet_50_preprocess(images):
    """
    Preprocess images for ResNet-50.

    Args:
        images (PIL.Image.Image or list of PIL.Image.Image): The input image(s).

    Returns:
        torch.Tensor: The preprocessed image tensor(s).
    """
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    if isinstance(images, list):
        return torch.stack([preprocess(image) for image in images])
    else:
        return preprocess(images)



def tensor_to_pil_image(tensor):
    """
    Convert a tensor to a PIL RGB image.

    Args:
        tensor (torch.Tensor): The input tensor.

    Returns:
        PIL.Image.Image: The converted PIL image.
    """
    if tensor.ndimension() == 4:  # If batch dimension exists, take the first image
        tensor = tensor[0]
    return to_pil_image(tensor)



def efficientnet_b6_preprocess(images):
    """
    Preprocess an image for EfficientNet.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    weights = EfficientNet_B6_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(images)


def efficientnet_v2_s_preprocess(images):
    """
    Preprocess an image for EfficientNet V2.

    Args:
        image (PIL.Image.Image): The input image.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    weights = EfficientNet_V2_S_Weights.DEFAULT
    preprocess = weights.transforms()
    return preprocess(images)

def reverse_normalization(tensor):
    """
    Reverse the normalization applied to a tensor and convert it to a PIL image.

    Args:
        tensor (torch.Tensor): The normalized tensor in C, H, W format.

    Returns:
        PIL.Image.Image: The denormalized PIL image.
    """
    # Define the mean and std used for normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Denormalize the tensor
    return tensor * std + mean
    



def load_dataset (path = DATASET_PATH) -> Dataset:
    """
    Load the dataset from the specified path.

    Args:
        path (str): The path to the dataset directory.

    Returns:
        torch.utils.data.Dataset: The loaded dataset.
    """
    dataset = torchvision.datasets.ImageFolder(root=path, transform=preprocess_image)
    return dataset


def load_dataloader (dataset : Dataset) -> DataLoader : 
    """
    Load the DataLoader for the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to load.

    Returns:
        torch.utils.data.DataLoader: The DataLoader for the dataset.
    """
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader



