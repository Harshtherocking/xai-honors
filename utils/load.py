import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration, AutoProcessor
)



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
    





def load_blip() -> (AutoModelForImageTextToText, AutoProcessor):
    model_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_id, dtype ="auto", device_map="auto")
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    return model, processor


def load_paligemma() -> (AutoModelForImageTextToText, AutoProcessor):
    model_id = "google/paligemma2-3b-pt-224"
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    processor = PaliGemmaProcessor.from_pretrained(model_id)
    return model, processor


def load_qwenvl_v3() -> (AutoModelForImageTextToText, AutoProcessor) :
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor