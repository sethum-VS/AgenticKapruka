"""NVIDIA NIM GenAI client helpers."""

from lib.genai.client import create_nvidia_client
from lib.genai.completions import generate_content

__all__ = ["create_nvidia_client", "generate_content"]
