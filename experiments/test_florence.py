"""
Quick benchmark for Florence-2-base on the existing extracted frames.
Compares directly to Qwen2.5-VL test on the same images.
"""
import os
import time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from unittest.mock import patch
from transformers.dynamic_module_utils import get_imports

# Florence-2 model card lists flash_attn as a hard requirement, but it
# only matters on CUDA. This workaround removes the import so the model
# loads cleanly on CPU.
def fixed_get_imports(filename):
    if not str(filename).endswith("modeling_florence2.py"):
        return get_imports(filename)
    imports = get_imports(filename)
    if "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


MODEL_NAME = "microsoft/Florence-2-base"  # 230M params (fastest)
FRAMES_FOLDER = "frames"

# Florence-2 accepts task prompts. <CAPTION> = short, <DETAILED_CAPTION> = longer.
# Short is what we want for one-sentence video frame description.
TASK_PROMPT = "<CAPTION>"


def main():
    print(f"Loading {MODEL_NAME}...")
    load_start = time.time()

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            attn_implementation="sdpa",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s "
          f"(first run downloads ~500MB, then cached)\n")

    frames = sorted(
        os.path.join(FRAMES_FOLDER, f)
        for f in os.listdir(FRAMES_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not frames:
        print("No frames found.")
        return

    total = 0
    warm_total = 0
    for index, frame_path in enumerate(frames, start=1):
        print(f"[{index}/{len(frames)}] {os.path.basename(frame_path)}...")
        image = Image.open(frame_path).convert("RGB")

        start = time.time()
        inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=60,
                num_beams=3,
                do_sample=False,
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = processor.post_process_generation(
            generated_text,
            task=TASK_PROMPT,
            image_size=(image.width, image.height),
        )
        elapsed = time.time() - start

        caption = parsed[TASK_PROMPT]
        total += elapsed
        if index > 1:
            warm_total += elapsed

        label = "(cold)" if index == 1 else "(warm)"
        print(f"  Time:    {elapsed:.1f}s {label}")
        print(f"  Caption: {caption}\n")

    print(f"Total: {total:.1f}s for {len(frames)} frames")
    if len(frames) > 1:
        print(f"Warm avg: {warm_total/(len(frames)-1):.1f}s/frame")


if __name__ == "__main__":
    main()