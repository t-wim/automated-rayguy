import os
import json
import replicate
import requests
from dotenv import load_dotenv

# Lade die Umgebungsvariablen aus der .env Datei
load_dotenv()

# Lese den API-Token aus der Umgebungsvariable
api_token = os.getenv("REPLICATE_API_TOKEN")

if not api_token:
    raise ValueError("REPLICATE_API_TOKEN nicht gesetzt. Bitte in der .env Datei hinterlegen.")

# Setze den Token für die Replicate-API (je nach Replicate Lib evtl. so oder anders)
os.environ["REPLICATE_API_TOKEN"] = api_token

# 1️⃣ Bild einmal hochladen – returns eine temporäre URL
uploaded = replicate.files.upload("rayguy.png")  # Pfad zu deinem PNG
image_url = uploaded  # diese URL geben wir später mit

# 2️⃣ Prompts laden
with open("prompts_controlnet_all.json", encoding="utf-8") as f:
    prompts = json.load(f)

out_dir = "meme_set"
os.makedirs(out_dir, exist_ok=True)

for item in prompts:
    print("🔄", item["filename"])

    result = replicate.run(
        "lucataco/sdxl-controlnet:db2ffdbdc7f6cb4d6dab512434679ee3366ae7ab84f89750f8947d5594b79a47",
        input={
            "prompt": item["prompt"],
            "image": image_url,          # 👈 URL statt Base64
            "control_type": "canny",
            "seed": 42,
            "guidance_scale": 7.5,
            "num_inference_steps": 30
        },
        timeout=300                    # optional länger warten
    )

    file_url = result[0]
    img_data = requests.get(file_url).content
    with open(os.path.join(out_dir, item["filename"]), "wb") as f:
        f.write(img_data)
    print("✅  gespeichert:", item["filename"])