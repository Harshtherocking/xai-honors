"""
Image utility functions for reading, displaying, and processing images using PIL.
"""

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import torch
import torchvision.transforms.functional as TF
from typing import Union, List, Optional, Tuple


def read_image(image_path: str) -> Image.Image:
    """
    Read an image from file path using PIL.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        PIL.Image.Image: Loaded image
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If file is not a valid image
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    try:
        image = Image.open(image_path)
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Error loading image {image_path}: {e}")


def display_image(image: Union[Image.Image, str], 
                 title: Optional[str] = None, 
                 figsize: Tuple[int, int] = (8, 6),
                 show_axis: bool = False) -> None:
    """
    Display an image using matplotlib.
    
    Args:
        image: PIL Image object or path to image file
        title (str, optional): Title for the plot
        figsize (tuple): Figure size (width, height)
        show_axis (bool): Whether to show axis labels and ticks
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = read_image(image)
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold')
    
    if not show_axis:
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def display_images(images: List[Union[Image.Image, str]], 
                  titles: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (15, 5),
                  cols: int = 3,
                  show_axis: bool = False) -> None:
    """
    Display multiple images in a grid layout.
    
    Args:
        images: List of PIL Image objects or image file paths
        titles (list, optional): List of titles for each image
        figsize (tuple): Overall figure size (width, height)
        cols (int): Number of columns in the grid
        show_axis (bool): Whether to show axis labels and ticks
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (image, ax) in enumerate(zip(images, axes.flat)):
        # Load image if path is provided
        if isinstance(image, str):
            image = read_image(image)
        
        ax.imshow(image)
        
        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=12)
        
        if not show_axis:
            ax.axis('off')
    
    # Hide empty subplots
    for i in range(n_images, rows * cols):
        axes.flat[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_image_with_info(image: Union[Image.Image, str], 
                           title: Optional[str] = None,
                           show_info: bool = True) -> None:
    """
    Display image with detailed information.
    
    Args:
        image: PIL Image object or path to image file
        title (str, optional): Title for the plot
        show_info (bool): Whether to show image information
    """
    # Load image if path is provided
    if isinstance(image, str):
        image = read_image(image)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Display image
    ax1.imshow(image)
    if title:
        ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # Display image information
    if show_info:
        info_text = f"""Image Information:
Size: {image.size[0]} × {image.size[1]} pixels
Mode: {image.mode}
Format: {image.format or 'Unknown'}
"""
        ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')
    
    plt.tight_layout()
    plt.show()


def resize_image(image: Union[Image.Image, str], 
                size: Tuple[int, int],
                keep_aspect: bool = True) -> Image.Image:
    """
    Resize an image to specified dimensions.
    
    Args:
        image: PIL Image object or path to image file
        size (tuple): Target size (width, height)
        keep_aspect (bool): Whether to maintain aspect ratio
        
    Returns:
        PIL.Image.Image: Resized image
    """
    if isinstance(image, str):
        image = read_image(image)
    
    if keep_aspect:
        image.thumbnail(size, Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize(size, Image.Resampling.LANCZOS)


def get_image_info(image: Union[Image.Image, str]) -> dict:
    """
    Get detailed information about an image.
    
    Args:
        image: PIL Image object or path to image file
        
    Returns:
        dict: Dictionary containing image information
    """
    if isinstance(image, str):
        image = read_image(image)
    
    return {
        'size': image.size,
        'mode': image.mode,
        'format': image.format,
        'width': image.width,
        'height': image.height,
        'aspect_ratio': image.width / image.height
    }


def list_images_in_directory(directory: str, 
                           extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')) -> List[str]:
    """
    List all image files in a directory.
    
    Args:
        directory (str): Path to directory
        extensions (tuple): Valid image file extensions
        
    Returns:
        list: List of image file paths
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(extensions):
            image_files.append(os.path.join(directory, filename))
    
    return sorted(image_files)


def display_random_images(directory: str, 
                         n_images: int = 6,
                         figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Display random images from a directory.
    
    Args:
        directory (str): Path to directory containing images
        n_images (int): Number of images to display
        figsize (tuple): Figure size (width, height)
    """
    import random
    
    image_files = list_images_in_directory(directory)
    
    if not image_files:
        print(f"No image files found in {directory}")
        return
    
    # Select random images
    selected_images = random.sample(image_files, min(n_images, len(image_files)))
    
    # Display selected images
    display_images(selected_images, 
                  titles=[os.path.basename(f) for f in selected_images],
                  figsize=figsize)


def denormalize_image(image_tensor):
    """
    Denormalize an image tensor using ImageNet mean and std values.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor
        
    Returns:
        torch.Tensor: Denormalized image tensor
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image_tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image_tensor.device)
    denormalized_tensor = image_tensor * std + mean
    return denormalized_tensor


def visualize_heatmap_overlay(patch_embeddings, original_image_tensor):
    """
    Visualize heatmap overlay on original image.
    
    Args:
        patch_embeddings: Patch embeddings tensor
        original_image_tensor: Original image tensor (3, 384, 384)
    """
    assert original_image_tensor.shape == (3, 384, 384), "Image must be (3, 384, 384)"

    # Reshape to 24x24 grid
    heatmap = patch_embeddings.reshape(24, 24)

    # Convert to torch for interpolation
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0)  # shape (1,1,24,24)
    heatmap_resized = TF.resize(heatmap_tensor, size=[384, 384], interpolation=TF.InterpolationMode.BILINEAR).squeeze().numpy()

    # Convert image to HWC for plotting
    original_image_tensor = denormalize_image(original_image_tensor)
    image_np = original_image_tensor.permute(1, 2, 0).numpy()

    trial = np.expand_dims(heatmap_resized, axis=2) * image_np

    # Plotting
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.title("Original Image")
    plt.imshow(image_np)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Heatmap")
    plt.imshow(heatmap_resized, cmap="gray")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Overlay")
    plt.imshow(image_np)
    plt.imshow(heatmap_resized, cmap='jet', alpha=0.7)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Combined")
    plt.imshow(trial)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
