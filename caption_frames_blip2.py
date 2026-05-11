import os
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

frames_folder = "frames"
files = os.listdir(frames_folder)
files.sort(key=lambda name: int(name.split("_")[1].split(".")[0]))

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float32
)

previous_caption = None
kept_count = 0
skipped_count = 0

for file_name in files:
    frame_path = os.path.join(frames_folder, file_name)
    image = Image.open(frame_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if caption == previous_caption:
        print("Frame ignorée :", file_name)
        print("Description répétée :", caption)
        print("-" * 40)
        skipped_count += 1
        continue

    print("Frame gardée :", file_name)
    print("Description :", caption)
    print("-" * 40)

    previous_caption = caption
    kept_count += 1

print("\nRésumé final :")
print("Frames gardées :", kept_count)
print("Frames ignorées car description répétée :", skipped_count)