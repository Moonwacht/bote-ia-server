import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent"

class ImagePayload(BaseModel):
    image: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/modelos")
def listar_modelos():
    response = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    )
    return response.json()

@app.post("/classify")
def classify(payload: ImagePayload):
    clean_image = payload.image.replace("\n", "").replace("\r", "").replace(" ", "").strip()

    body = {
        "contents": [{
            "parts": [
                {
                    "text": "Look at this image. There is a solid colored background that you must ignore completely. Focus ONLY on identifying a physical waste object in the image. If you see a clear, distinct physical object (food, fruit, vegetable, paper, plastic, metal, glass, can, bottle, wrapper, etc), classify it. Reply 'organic' for food/plants/biodegradable. Reply 'inorganic' for plastic/metal/glass/paper/non-biodegradable. Reply 'empty' ONLY if there is absolutely no physical object visible at all. One word only, lowercase."
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": clean_image
                    }
                }
            ]
        }]
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=body,
            timeout=30
        )
        result = response.json()
        print("Gemini raw response:", result)

        text = result["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().lower()

        if "empty" in text:
            return {"type": "empty"}
        elif "organic" in text and "inorganic" not in text:
            return {"type": "organic"}
        elif "inorganic" in text:
            return {"type": "inorganic"}
        else:
            return {"type": "unknown", "raw": text}

    except Exception as e:
        print("Error completo:", str(e), "Response:", response.text if 'response' in locals() else "sin respuesta")
        return {"type": "error", "detail": str(e)}
