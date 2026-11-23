import requests
import base64
from PIL import Image
import io

def prepare_image_base64(image_path):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((320, 320))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def describe_scene_with_moondream(image_paths):
    encoded_images = []

    for image_path in image_paths:
        encoded_image = prepare_image_base64(image_path)
        encoded_images.append(encoded_image)

    payload = {
        "model": "moondream",
        "prompt": (
            "These images are frames from the same scene in a video. "
            "Describe what is happening across this scene in one short coherent paragraph. "
            "Focus on visible actions, objects, and important changes between frames. "
            "Do not invent objects that are not clearly visible."
        ),
        "images": encoded_images,
        "stream": False
    }

    response = requests.post(
        "http://127.0.0.1:11434/api/generate",
        json=payload,
        timeout=300
    )

    result = response.json()
    return result["response"].strip()