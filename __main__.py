import os
from dotenv import load_dotenv
from utils.auth import setup_huggingface
from utils.load import load_blip_captioning, tensor_to_pil_image, generate_caption, evaluate_caption
from xai.hook import create_hook_functions, processed_grads
from utils.image_utils import read_image, denormalize_image, display_images, visualize_heatmap_overlay
import torch

load_dotenv()
setup_huggingface()


if __name__ == "__main__":
    processor, model = load_blip_captioning()
    
    prod_image = read_image("./images/black_suit.webp")
    inputs = processor(images=prod_image,  return_tensors="pt", do_rescale=True)
    img = inputs.pixel_values


    decoder_output = {}
    gradients = {}
    
    decoder_hook, gradient_hook = create_hook_functions(decoder_output, gradients)

    frwd_hook = model.text_decoder.cls.predictions.decoder.register_forward_hook(decoder_hook)
    back_hook = model.vision_model.encoder.layers[11].register_full_backward_hook(
        lambda module, grad_input, grad_output: gradient_hook(module, grad_input, grad_output, "vision_encoder_layer_11")
    )

    # Generate caption using the function from utils
    # generated_ids = generate_caption(model, processor, img)
    generated_ids = model.generate(
        img,
        max_length=30,
        do_sample=True,
        # top_k=5,
        top_p=0.8,
        temperature=0.8,
        num_beams=3
    )

    # Decode the generated caption
    caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Generated caption: {caption}")

    # Get reference caption from user input
    print("\n" + "="*50)
    print("CAPTION EVALUATION")
    print("="*50)
    reference_caption = input("Enter the reference/ground truth caption: ")
    
    if reference_caption.strip():
        # Calculate evaluation metrics
        scores = evaluate_caption(caption, reference_caption)
        
        print(f"\nEvaluation Results:")
        print(f"Generated: {caption}")
        print(f"Reference: {reference_caption}")
        print(f"BLEU Score: {scores['bleu_score']:.4f}")
        print(f"CHRF Score: {scores['chrf_score']:.4f}")
    else:
        print("No reference caption provided. Skipping evaluation.")

    # predictions
    predictions = decoder_output["last_layer"][0]

    # for grads in processed_grads(predictions, model, processor, gradients, layer_name = "vision_encoder_layer_11") :
    #    visualize_heatmap_overlay(grads[0], img[0]) 

    
