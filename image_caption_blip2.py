from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

image = Image.open("test2.jpg").convert("RGB")

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float32
)

inputs = processor(images=image, return_tensors="pt")
generated_ids = model.generate(**inputs, max_new_tokens=50)
caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print("Description de l'image :", caption)