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
                    "text": "Look at this image. The background is a solid aqua/teal colored surface and must be completely ignored. Only classify the waste object placed ON the background. If there is no object visible and you only see the aqua background, reply with only the word 'empty'. If there is an object, reply with only 'organic' if it is food, plants, or biodegradable material, or only 'inorganic' if it is plastic, metal, glass, paper, or any non-biodegradable material. One word only, lowercase, no punctuation, no explanation."
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
