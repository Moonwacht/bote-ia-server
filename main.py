```python
import os
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
# HEALTH CHECK
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

        # limpiar base64
        clean_image = (
            payload.image
            .replace("\n", "")
            .replace("\r", "")
            .replace(" ", "")
            .strip()
        )

        image_url = f"data:image/jpeg;base64,{clean_image}"

        # PROMPT MEJORADO
        prompt = """
You are a strict embedded waste classification system.

The image normally contains a plain blue background.

IMPORTANT:
DO NOT classify the blue background itself as an object.

ONLY classify if there is a CLEAR and DISTINCT physical waste object visible.

If you are not completely sure that an object exists, return:
000000

Return ONLY ONE exact code:

000101 = organic waste
000110 = inorganic waste
000000 = empty
111111 = error

Examples of organic:
- food
- fruit
- vegetables
- plants
- bread

Examples of inorganic:
- bottles
- cans
- plastic
- wrappers
- paper
- metal
- glass

Rules:
- blue background alone = 000000
- shadows alone = 000000
- lighting variations = 000000
- empty scene = 000000

DO NOT explain.
ONLY return the exact code.
"""

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": prompt
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

        # validar códigos
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
```
