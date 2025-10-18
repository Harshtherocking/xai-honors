#!/usr/bin/env python3
"""
Hook module for XAI (Explainable AI) functionality.
This module provides hooks for model interpretation and analysis.
"""

from typing import Any, Dict, List
import numpy as np


def decoder_frwd_hook(module, input, output, decoder_output: Dict[str, Any]):
    """
    Forward hook for storing decoder layer output.
    
    Args:
        module: The module being hooked
        input: Input to the module
        output: Output from the module
        decoder_output: Dictionary to store decoder outputs
    """
    decoder_output["last_layer"] = output


def backward_hook(module, grad_input, grad_output, layer_name: str, gradients: Dict[str, List]):
    """
    Backward hook to collect gradients.
    
    Args:
        module: The module being hooked
        grad_input: Input gradients
        grad_output: Output gradients
        layer_name: Name of the layer for gradient storage
        gradients: Dictionary to store gradients for each layer
    """
    if layer_name not in gradients:
        gradients[layer_name] = []
    
    for g in grad_output:
        if g is not None:
            gradients[layer_name].append(g.detach().cpu().numpy())
        else:
            gradients[layer_name].append(None)


def create_hook_functions(decoder_output: Dict[str, Any], gradients: Dict[str, List]):
    """
    Create hook functions that use the provided dictionaries.
    
    Args:
        decoder_output: Dictionary to store decoder outputs
        gradients: Dictionary to store gradients
        
    Returns:
        Tuple: (decoder_hook, gradient_hook) functions
    """
    def decoder_hook(module, input, output):
        decoder_frwd_hook(module, input, output, decoder_output)
    
    def gradient_hook(module, grad_input, grad_output, layer_name: str):
        backward_hook(module, grad_input, grad_output, layer_name, gradients)
    
    return decoder_hook, gradient_hook


def processed_grads(predictions, model, processor, gradients, layer_name):
    """
    Process gradients to compute importance weights for each token.
    
    Args:
        predictions: Model predictions tensor
        model: The model to compute gradients from
        gradients: Dictionary containing collected gradients
        processor: Model processor for token decoding
        
    Yields:
        torch.Tensor: Weighted and normalized gradients for each token
    """
    for i in range(predictions.shape[0]):
        pred = predictions[i]

        y = pred.argmax()
        print(f"Token {i} : {processor.tokenizer.decode(y)} --------")

        model.zero_grad()
        pred[y].backward(retain_graph=True)

        grads = gradients[layer_name][-1]
        # print(f"Shape of Grads : {grads.shape}")[1,577, 768]

        # dropping class token
        grads = grads[:, 1:, :]

        # dropping negative grads
        grads = grads.clip(min=0)

        # for every channel k : {1... 768}
        # importance of a channel
        A_k = grads.sum(axis=1) / grads.shape[1]
        A_k = np.expand_dims(A_k, 1)
        # print(A_k.shape) # [1, 1, 768]

        # A_k * k Channel :
        weighted_grads = grads * A_k
        # print(weighted_grads.shape)  # [1, 576, 768]

        # sum up over channels k : {1 .... 768}
        weighted_grads = weighted_grads.sum(axis=2)
        # print(weighted_grads.shape) # [1, 576]

        # min max normalization
        g_min = weighted_grads.min()
        g_max = weighted_grads.max()

        weighted_grads = (weighted_grads - g_min) / (g_max - g_min)

        # print(weighted_grads.min()) # 0
        # print(weighted_grads.max()) # 1

        yield weighted_grads


if __name__ == "__main__":
    # Example usage
    print("XAI Hook module loaded successfully!")
    print("Use this module to hook into PyTorch models for XAI analysis.")
    print("Hook functions now accept dictionaries as parameters for external access.")
    
    # Example of how to use the new hook functions
    print("\nExample usage:")
    print("decoder_output = {}")
    print("gradients = {}")
    print("decoder_hook, gradient_hook = create_hook_functions(decoder_output, gradients)")
    print("# Now you can access decoder_output and gradients from outside the functions!")
    print("# Use lambda when registering backward hook: lambda m, gi, go: gradient_hook(m, gi, go, 'layer_name')")
