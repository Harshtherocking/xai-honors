import torch
from torchvision.models import (
    efficientnet_b6, 
    resnet50, 
    ResNet50_Weights, 
    efficientnet_v2_s, 
    EfficientNet_V2_S_Weights, 
    EfficientNet_B6_Weights, 
    vit_b_16,
    ViT_B_16_Weights
)

import captum 
from captum.attr import LRP, Lime, Occlusion, GuidedGradCam

from PIL import Image
from torchvision import transforms 

from utils.load import efficientnet_v2_s_preprocess, efficientnet_b6_preprocess, reverse_normalization, tensor_to_pil_image, resnet_50_preprocess
import numpy as np

import matplotlib.pyplot as plt

def plot_attr_as_heatmap(attr, original_image):
    """
    Plots the attribution map as a heatmap overlaid on the original image.

    Args:
        attr (torch.Tensor): The attribution map tensor.
        original_image (PIL.Image): The original image.
    """
    attr = attr.detach().cpu().numpy()
    attr = np.mean(attr, axis=0)  # Average across color channels if needed

    # Normalize the attribution map
    attr = (attr - np.min(attr)) / (np.max(attr) - np.min(attr) + 1e-8)

    # Convert original image to numpy array
    original_image_np = np.array(original_image)

    # If the image is in (C, H, W), transpose it to (H, W, C)
    if original_image_np.shape[0] == 3:  # Check if it's in (C, H, W)
        original_image_np = np.transpose(original_image_np, (1, 2, 0))

    # Plot the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(original_image_np)
    plt.imshow(attr, cmap='jet', alpha=0.5)  # Overlay heatmap with transparency
    plt.axis('off')
    plt.title("Attribution Heatmap")
    plt.show()


# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device : ", device)



def load_efficientnet_b6():
    # Load the EfficientNet V2 S model with pretrained weights
    model = efficientnet_b6(weights=EfficientNet_B6_Weights.IMAGENET1K_V1)
    model.eval()  # Set the model to evaluation mode
    return model

def load_resnet_50():
    # Load the ResNet-50 model with pretrained weights
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.eval()  # Set the model to evaluation mode
    return model

def load_efficientnet_v2_s():
    # Load the EfficientNet V2 S model with pretrained weights
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.eval()  # Set the model to evaluation mode
    return model


def load_image(image_path = "imagenet_val/00100/87469483327336.jpg") : 
    image = Image.open(image_path)
    # transformation 
    image_pre = efficientnet_b6_preprocess(image)
    # image_pre = resnet_50_preprocess(image)

    # temp = reverse_normalization(image_pre)
    # tensor_to_pil_image(temp).show()
    return image_pre


def main():
    # model = load_efficientnet_b6()
    model = load_resnet_50()
    # model = load_efficientnet_v2_s()

    # model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    # model.eval()
    # print(model)
    # exit()

    # for resnet50
    # layer = model.layer4[-1].conv3
    # layer = model.layer4[-1].conv2
    layer = model.avgpool

    # for efficientnetb6
    # layer = model.avgpool
    # layer = model.features[0][0]

    print("Layer: ", layer)

    # image loading 
    img = load_image()
    
    # get class number
    output = model(img.unsqueeze(0))
    max_idx = torch.argmax(output, dim=1).item()
    print("Class ID : ", max_idx)

    # grad cam 
    grad_cam_model = GuidedGradCam(model, layer)
    attr = grad_cam_model.attribute(img.unsqueeze(0), target= max_idx)
    attr = attr[0]



    tensor_to_pil_image(attr).show()
    tensor_to_pil_image(attr.clamp(min = 0, max = 255)).show()

    # plot_attr_as_heatmap(attr.clamp(min = 0, max = 255), reverse_normalization(img))

    # # LRP 
    # lrp = LRP(model)
    # attr = lrp.attribute(img.unsqueeze(0), target=max_idx)
    # attr = attr[0]
    # tensor_to_pil_image(attr).show()


    # # Lime
    # lime = Lime(model)
    # attr = lime.attribute(img.unsqueeze(0), target=max_idx)
    # attr = attr[0]
    # tensor_to_pil_image(attr).show()


    # print("attr : ", attr)
    # print("attr shape : ", attr.shape)


    pass

if __name__ == "__main__":
    main()