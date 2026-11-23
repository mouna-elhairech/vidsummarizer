import requests
import base64
from PIL import Image
import io

image_path = "frames/frame_0000017.jpg"

image = Image.open(image_path).convert("RGB")
image.thumbnail((320, 320))

buffer = io.BytesIO()
image.save(buffer, format="JPEG", quality=85)
image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

payload = {
    "model": "qwen2.5vl:3b",
    "prompt": "Describe this image accurately in one short sentence. Mention only what is clearly visible. Do not invent objects.",
    "images": [image_base64],
    "stream": False
}

response = requests.post(
    "http://127.0.0.1:11434/api/generate",
    json=payload,
    timeout=300
)

print("Status code:", response.status_code)
print("Raw response:")
print(response.text)