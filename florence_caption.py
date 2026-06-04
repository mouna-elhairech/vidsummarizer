"""
Florence-2 image captioning module.

The model is loaded lazily on the first call to caption_image() and kept
in memory afterwards. This works for both CLI usage and long-running
processes (FastAPI), where the model stays warm across requests.
"""
import os
from unittest.mock import patch

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports


MODEL_NAME = "microsoft/Florence-2-base"  # 232M params, fastest variant

_model = None
_processor = None


def _fixed_get_imports(filename):
    """Strip flash_attn from imports; GPU-only, not needed on CPU."""
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_model():
    """Load Florence-2 model + processor (idempotent)."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    with patch("transformers.dynamic_module_utils.get_imports", _fixed_get_imports):
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    _processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    _model.eval()
    return _model, _processor


def caption_image(image, prompt_type="<CAPTION>", max_tokens=60):
    """
    Caption a single image.

    Args:
        image: PIL.Image or path to an image file.
        prompt_type: Florence-2 task prompt.
            "<CAPTION>"             -> short single sentence
            "<DETAILED_CAPTION>"    -> longer, more detailed
            "<MORE_DETAILED_CAPTION>" -> verbose paragraph
        max_tokens: cap on generated tokens.

    Returns:
        Caption string (stripped).
    """
    model, processor = load_model()

    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    inputs = processor(text=prompt_type, images=image, return_tensors="pt")

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=max_tokens,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]
    parsed = processor.post_process_generation(
        generated_text,
        task=prompt_type,
        image_size=(image.width, image.height),
    )
    return parsed[prompt_type].strip()


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        print("Usage: python florence_caption.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Captioning {image_path}...")

    start = time.time()
    caption = caption_image(image_path)
    elapsed = time.time() - start

    print(f"Caption ({elapsed:.1f}s): {caption}")