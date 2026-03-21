import os
from utils.load import load_blip, load_qwenvl_v3, load_paligemma
# from xai.hook import create_hook_functions, processed_grads
import torch
from utils.inference import generate_caption, DecodeConfig
from PIL import Image
from utils.dataset import load_dataset_from_hub
from utils.train import train, prepare_lora_model
import config
from utils.gen import generate_caption_logits
from xai.likelihood import register_grad_hooks, processed_grads, calculate_likelihood
from xai.posterior import calculate_posterior

from xai.prior import register_cross_attn_hooks, stack_cross_attn, calculate_prior
from utils.gen import generate_caption_logits
from xai.visual import visualize

DEVICE = config.DEVICE

# changing config
config.MODEL = "blip"
config.LOAD_PRETRAINED_MODEL = False


if __name__ == "__main__":
    if config.MODEL == "blip":
        if config.LOAD_PRETRAINED_MODEL:
            model, processor = load_blip(config.OUTPUT_DIR)
        model, processor = load_blip()
    elif config.MODEL == "qwenvl":
        if config.LOAD_PRETRAINED_MODEL:
            model, processor = load_qwenvl_v3(config.OUTPUT_DIR)
        model, processor = load_qwenvl_v3()
    elif config.MODEL == "paligemma":
        if config.LOAD_PRETRAINED_MODEL:
            model, processor = load_paligemma(config.OUTPUT_DIR)
        model, processor = load_paligemma()

    assert model is not None, f"{config.MODEL}: cant load model"
    assert processor is not None, f"{config.MODEL}: cant load processor"

    # print(model)


    ------------------------------------------------------------------------------------------
    dataset = load_dataset_from_hub(processor, split="train")
    val_dataset = load_dataset_from_hub(processor, split="validation")
    train(model, dataset, processor, val_dataset)
    ------------------------------------------------------------------------------------------

    print("Execution Completed")
    exit()




    # XAI part
    # -------------------------------------------------------------------------------------------------
    prod_image = Image.open("images/test/14.jpeg")

    # hook registration
    frwd_hooks, cross_attn_list = register_cross_attn_hooks(model)
    back_hooks, decoder_out, grad_dict= register_grad_hooks(model)

    # autoregressive decoding with greedy decoding
    out = generate_caption_logits(model, processor, prod_image)

    # calling bacward to collect gradients
    predictions = decoder_out["last_layer"]
    tokens = processed_grads(predictions, model, processor, verbose=True)

    # prior computation
    prior = calculate_prior(cross_attn_list)

    # likelihood computation
    likelihood = calculate_likelihood(grad_dict)

    print(f"prior : {prior.shape}")
    print(f"likelihood : {likelihood.shape}")

    posterior = calculate_posterior(prior, likelihood)

    # out = generate_caption(model, processor, prod_image)


    result = visualize(
        posterior,  # (1, T, H*W) — from calculate_prior() or compute_gradcam()
        tokens,  # list[str]  or  list[int]  or  1-D int tensor
        prod_image,  # PIL.Image (RGB)
        processor,  # AutoProcessor — only needed when tokens are integer ids
        out_dir="./vis_out",
        show=True,
        animated=True,
    )
    # -------------------------------------------------------------------------------------------------



    breakpoint()