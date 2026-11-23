from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image = Image.open("test2.jpg").convert("RGB")

prompt = "Describe this image in detail"
inputs = processor(images=image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=80)

caption = processor.decode(output[0], skip_special_tokens=True)

print("Description de l'image :", caption)