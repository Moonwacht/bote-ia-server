import os
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import PlainTextResponse

app = FastAPI()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)

class ImagePayload(BaseModel):
    image: str


# ═══════════════════════════════════════
# CODIGOS DEL SISTEMA
# ═══════════════════════════════════════

ORGANIC_CODE   = "000101"
INORGANIC_CODE = "000110"
EMPTY_CODE     = "000000"
ERROR_CODE     = "111111"


# ═══════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════

@app.get("/")
def health():
    return {"status": "ok"}


# ═══════════════════════════════════════
# LISTAR MODELOS
# ═══════════════════════════════════════

@app.get("/modelos")
def listar_modelos():
    response = requests.get(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
    )
    return response.json()


# ═══════════════════════════════════════
# CLASIFICACION
# ═══════════════════════════════════════

@app.post("/classify", response_class=PlainTextResponse)
def classify(payload: ImagePayload):

    # limpiar base64
    clean_image = (
        payload.image
        .replace("\n", "")
        .replace("\r", "")
        .replace(" ", "")
        .strip()
    )

    # prompt optimizado
    prompt = """
You are an embedded waste classification system.

Analyze ONLY the main physical object in the image.

Ignore:
- background
- walls
- hands
- shadows
- table
- empty colored surfaces

Return ONLY ONE of these exact codes:

000101 = organic waste
000110 = inorganic waste
000000 = empty / no object
111111 = error / uncertain

Rules:
- food, fruit, vegetables, plants = 000101
- plastic, metal, glass, paper, cans, wrappers = 000110
- absolutely no visible object = 000000

DO NOT explain.
DO NOT use JSON.
DO NOT add text.
ONLY return the exact 6-digit code.
"""

    body = {
        "contents": [{
            "parts": [
                {
                    "text": prompt
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

        print("═══════════════════════════════")
        print("Gemini RAW response:")
        print(result)
        print("═══════════════════════════════")

        # extraer texto
        text = result["candidates"][0]["content"]["parts"][0]["text"]

        text = text.strip()

        print("Codigo recibido:", text)

        # validar codigos permitidos
        valid_codes = [
            ORGANIC_CODE,
            INORGANIC_CODE,
            EMPTY_CODE,
            ERROR_CODE
        ]

        if text not in valid_codes:
            print("Codigo invalido detectado")
            return ERROR_CODE

        return text

    except Exception as e:

        print("ERROR COMPLETO:")
        print(str(e))

        if 'response' in locals():
            print(response.text)

        return ERROR_CODE
