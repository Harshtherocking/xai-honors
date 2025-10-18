from threading import local
import torch
from torchvision.models import EfficientNet_V2_S_Weights,EfficientNet_B6_Weights

from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms.functional import to_pil_image
from transformers import AutoProcessor, AutoModelForImageTextToText



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
    



def load_dataset (path = DATASET_PATH, preprocess_func=None) -> Dataset:
    """
    Load the dataset from the specified path.

    Args:
        path (str): The path to the dataset directory.
        preprocess_func: The preprocessing function to apply. 
                        Defaults to resnet_50_preprocess if None.

    Returns:
        torch.utils.data.Dataset: The loaded dataset.
    """
    if preprocess_func is None:
        preprocess_func = resnet_50_preprocess
    
    dataset = torchvision.datasets.ImageFolder(root=path, transform=preprocess_func)
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


def load_blip_model_and_processor(model_name: str = "Salesforce/blip-image-captioning-base"):
    """
    Load BLIP model and processor for image captioning.
    
    Args:
        model_name (str): The name of the BLIP model to load. 
                         Defaults to "Salesforce/blip-image-captioning-base"
    
    Returns:
        tuple: (processor, model) - The loaded processor and model
    """
    try:
        print(f"Loading BLIP model: {model_name}")
        
        # Check for local model cache first
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import os
        
        # Extract model name without organization prefix
        model_short_name = model_name.split('/')[-1] if '/' in model_name else model_name
        
        # Common local cache paths
        local_paths = [
            f"./models/{model_short_name}",
            f"./models/{model_name}",
            os.path.expanduser(f"~/.cache/huggingface/hub/{model_name}"),
            os.path.expanduser(f"~/.cache/huggingface/transformers/{model_name}")
        ]
        
        # Check if model exists locally
        local_model_path = None
        for path in local_paths:
            if os.path.exists(path):
                local_model_path = path
                print(f"Found local model at: {path}")
                break
        
        if local_model_path:
            print("Loading from local cache...")
            processor = AutoProcessor.from_pretrained(local_model_path)
            model = AutoModelForImageTextToText.from_pretrained(local_model_path)
        else:
            print("Downloading model from Hugging Face...")
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForImageTextToText.from_pretrained(model_name)
            
            # Save model locally for future use
            os.makedirs(f"./models/{model_short_name}", exist_ok=True)
            print(f"Saving model locally to: ./models/{model_short_name}")
            processor.save_pretrained(f"./models/{model_short_name}")
            model.save_pretrained(f"./models/{model_short_name}")
        
        print("BLIP model and processor loaded successfully!")
        
        return processor, model
        
    except Exception as e:
        raise ValueError(f"Error loading BLIP model '{model_name}': {e}")


def load_blip_captioning():
    """
    Convenience function to load the default BLIP image captioning model.
    
    Returns:
        tuple: (processor, model) - The loaded BLIP captioning processor and model
    """
    return load_blip_model_and_processor("Salesforce/blip-image-captioning-base")


def load_blip_vqa():
    """
    Convenience function to load BLIP VQA model.
    
    Returns:
        tuple: (processor, model) - The loaded BLIP VQA processor and model
    """
    return load_blip_model_and_processor("Salesforce/blip-vqa-base")


def generate_caption(model, processor, img, max_length=30):
    """
    Generate caption token by token using a text generation model.
    
    Args:
        model: Text generation model (e.g., BLIP, GPT, etc.)
        processor: Model processor/tokenizer
        img: Input image tensor
        max_length (int): Maximum number of tokens to generate
        
    Returns:
        torch.Tensor: Generated token IDs
    """
    generated_ids = torch.tensor([
        [processor.tokenizer.cls_token_id]
    ])

    for i in range(max_length):
        # zero all collected grads
        model.zero_grad()

        outputs = model(img, generated_ids)

        # batch, vocab
        next_token_logits = outputs.logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1)

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

        # end at sep token
        if next_token.item() == processor.tokenizer.sep_token_id:
            break

    return generated_ids


def calculate_bleu_score(generated_caption, reference_caption):
    """
    Calculate BLEU score for caption evaluation.
    
    Args:
        generated_caption (str): The generated caption
        reference_caption (str): The reference/ground truth caption
        
    Returns:
        float: BLEU score between 0 and 1
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    # Tokenize captions
    generated_tokens = generated_caption.lower().split()
    reference_tokens = reference_caption.lower().split()
    
    # Calculate BLEU score with smoothing
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
    
    return bleu_score


def calculate_chrf_score(generated_caption, reference_caption):
    """
    Calculate CHRF score for caption evaluation.
    
    Args:
        generated_caption (str): The generated caption
        reference_caption (str): The reference/ground truth caption
        
    Returns:
        float: CHRF score between 0 and 1
    """
    from sacrebleu import CHRF
    
    # Calculate CHRF score
    chrf = CHRF()
    score = chrf.sentence_score(generated_caption, [reference_caption])
    
    return score.score / 100.0  # Normalize to 0-1 range


def evaluate_caption(generated_caption, reference_caption):
    """
    Evaluate generated caption using multiple metrics.
    
    Args:
        generated_caption (str): The generated caption
        reference_caption (str): The reference/ground truth caption
        
    Returns:
        dict: Dictionary containing BLEU and CHRF scores
    """
    bleu_score = calculate_bleu_score(generated_caption, reference_caption)
    chrf_score = calculate_chrf_score(generated_caption, reference_caption)
    
    return {
        'bleu_score': bleu_score,
        'chrf_score': chrf_score
    }



