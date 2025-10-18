"""
Authentication utilities for Hugging Face and other services.
"""

import os
from huggingface_hub import login


def setup_huggingface():
    """Setup Hugging Face authentication"""
    # Check if HF_TOKEN environment variable is set
    if os.getenv('HF_TOKEN'):
        print("Using HF_TOKEN from environment variable")
        login(token=os.getenv('HF_TOKEN'))
    else:
        # Interactive login
        print("Please enter your Hugging Face token:")
        print("You can get your token from: https://huggingface.co/settings/tokens")
        token = input("Enter your token: ").strip()
        if token:
            login(token=token)
            print("Successfully logged in to Hugging Face!")
        else:
            print("No token provided. Some features may not work.")


def check_hf_token():
    """Check if HF_TOKEN environment variable is set"""
    return os.getenv('HF_TOKEN') is not None


def get_hf_token():
    """Get the HF_TOKEN from environment variable"""
    return os.getenv('HF_TOKEN')
