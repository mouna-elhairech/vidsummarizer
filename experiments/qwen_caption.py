import requests
import base64
from PIL import Image
import io

def describe_image_with_qwen(image_path):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((320, 320))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    payload = {
        "model": "qwen2.5vl:3b",
        "prompt": (
            "Describe this image accurately in one short sentence. "
            "Mention only what is clearly visible. "
            "Do not invent objects."
        ),
        "images": [image_base64],
        "stream": False
    }

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json=payload,
        timeout=300
    )

    result = response.json()
    return result["response"].strip()