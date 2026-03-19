import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
    AutoModelForImageTextToText,
    Qwen3VLForConditionalGeneration, AutoProcessor
)

import config
from config import HF_TOKEN
from peft import PeftModel


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
    





def load_blip(dir : str | None = None) -> (AutoModelForImageTextToText, AutoProcessor):
    model_id = "Salesforce/blip-image-captioning-large"
    processor = BlipProcessor.from_pretrained(model_id, dtype ="auto", token= config.HF_TOKEN)
    model = BlipForConditionalGeneration.from_pretrained(model_id, token=HF_TOKEN)

    if dir  :
        model_id = model_id if dir is None else dir
        model = PeftModel.from_pretrained(model, model_id)
    return model, processor


def load_paligemma(dir : str | None = None) -> (AutoModelForImageTextToText, AutoProcessor):
    model_id = "google/paligemma2-3b-pt-224"
    processor = PaliGemmaProcessor.from_pretrained(model_id, token = config.HF_TOKEN)
    model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, dtype=torch.bfloat16, token = config.HF_TOKEN)

    if dir  :
        model_id = model_id if dir is None else dir
        model = PeftModel.from_pretrained(model, model_id)
    return model, processor


def load_qwenvl_v3(dir : str | None = None) -> (AutoModelForImageTextToText, AutoProcessor) :
    model_id = "Qwen/Qwen3-VL-4B-Instruct"
    processor = AutoProcessor.from_pretrained(model_id, token= config.HF_TOKEN)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", token = config.HF_TOKEN
    )

    if dir  :
        model_id = model_id if dir is None else dir
        model = PeftModel.from_pretrained(model, model_id)
    return model, processor