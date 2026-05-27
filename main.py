import os
import base64
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse
from openai import OpenAI

app = FastAPI()

# ═══════════════════════════════════════
# OPENAI
# ═══════════════════════════════════════

api_key = os.environ.get("OPENAI_API_KEY")

print("API KEY DETECTADA:", api_key)

client = OpenAI(
    api_key=api_key
)

MODEL = "gpt-4o-mini"

# ═══════════════════════════════════════
# CODIGOS
# ═══════════════════════════════════════

ORGANIC_CODE   = "000101"
INORGANIC_CODE = "000110"
EMPTY_CODE     = "000000"
ERROR_CODE     = "111111"

# ═══════════════════════════════════════
# REQUEST MODEL
# ═══════════════════════════════════════

class ImagePayload(BaseModel):
    image: str

# ═══════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════

@app.get("/")
def health():
    return {"status": "ok"}

# ═══════════════════════════════════════
# CLASIFICACION
# ═══════════════════════════════════════

@app.post("/classify", response_class=PlainTextResponse)
def classify(payload: ImagePayload):

    try:

        clean_image = (
            payload.image
            .replace("\n", "")
            .replace("\r", "")
            .replace(" ", "")
            .strip()
        )

        image_url = f"data:image/jpeg;base64,{clean_image}"

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an embedded waste classification system.

Analyze ONLY the main object.

Ignore:
- background
- walls
- hands
- shadows
- tables

Return ONLY ONE exact code:

000101 = organic
000110 = inorganic
000000 = empty
111111 = error

Rules:
- food, fruit, vegetables, plants = 000101
- plastic, metal, glass, paper, cans, wrappers = 000110
- no visible object = 000000

DO NOT explain.
ONLY return the code.
"""
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=10
        )

        text = response.choices[0].message.content.strip()

        print("═══════════════════════════════")
        print("Codigo IA:", text)
        print("═══════════════════════════════")

        valid_codes = [
            ORGANIC_CODE,
            INORGANIC_CODE,
            EMPTY_CODE,
            ERROR_CODE
        ]

        if text not in valid_codes:
            print("Codigo invalido")
            return ERROR_CODE

        return text

    except Exception as e:

        print("ERROR:")
        print(str(e))

        return ERROR_CODE
