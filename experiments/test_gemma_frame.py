import requests
import base64

image_path = "frames/frame_0.jpg"

with open(image_path, "rb") as image_file:
    image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

payload = {
    "model": "gemma3:4b",
    "prompt": "Describe this image accurately and briefly. Mention only what is clearly visible. Do not guess or invent objects.",
    "images": [image_base64],
    "stream": False
}

response = requests.post("http://127.0.0.1:11434/api/generate", json=payload)
result = response.json()

print("Réponse Gemma :")
print(result["response"])