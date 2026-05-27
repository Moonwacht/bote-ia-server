import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

class ImagePayload(BaseModel):
    image: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/classify")
def classify(payload: ImagePayload):
    body = {
        "contents": [{
            "parts": [
                {
                    "text": "Look at this image. Classify the waste item as exactly one word, lowercase only. Reply with only 'organic' if it is food, plants, or biodegradable material. Reply with only 'inorganic' if it is plastic, metal, glass, paper, or any non-biodegradable material. No other words, no punctuation, no explanation."
                },
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": payload.image
                    }
                }
            ]
        }]
    }

    try:
        response = requests.post(
            f"{GEMINI_URL}?key={GEMINI_API_KEY}",
            json=body,
            timeout=15
        )
        result = response.json()
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        text = text.strip().lower()

        if "organic" in text and "inorganic" not in text:
            return {"type": "organic"}
        elif "inorganic" in text:
            return {"type": "inorganic"}
        else:
            return {"type": "unknown", "raw": text}

    except Exception as e:
        return {"type": "error", "detail": str(e)}
