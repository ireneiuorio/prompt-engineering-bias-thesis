"""
Script di validazione manuale della domanda Q2 di BLIP:
  "Does this image show only one person clearly in the foreground?"

"""

import csv
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


MODEL_NAME = "Salesforce/blip-vqa-base"
OUTPUT_CSV = Path("./labelling/q2_validation.csv")

QUESTION = "Does this image show only one person clearly in the foreground?"

# 30 immagini: 5 da ciascuna combinazione interessante.
# Variamo task e seed per avere campione vario.
SAMPLE_IMAGES = []

COMBOS = [
    ("Segmind", "descriptive"),
    ("SDXL",    "descriptive"),
    ("SDXL",    "negative_gender"),
    ("SDXL",    "negative_ethnicity"),
    ("Flux",    "descriptive"),
    ("Flux",    "persona"),
]

# Task e seed scelti per varieta'
TASKS_SAMPLE = ["codes", "fixes_bugs", "has_meetings", "helps_others", "writes_documentation_wiki_pages"]
SEEDS_SAMPLE = [0, 1, 2, 3, 4]

for model, pattern in COMBOS:
    for task, seed in zip(TASKS_SAMPLE, SEEDS_SAMPLE):
        SAMPLE_IMAGES.append({
            "model":   model,
            "pattern": pattern,
            "task":    task,
            "seed":    seed,
            "path":    f"./output/{model}/{pattern}/{task}/seed_{seed:04d}.png",
        })


def setup_blip():
    print("Carico BLIP-VQA...")
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Caricato in {time.time() - t0:.1f}s (device={device})\n")
    return processor, model, device


def ask_blip(processor, model, device, image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()


def main():
    processor, model, device = setup_blip()

    rows = []
    print(f"{'#':>3} {'MODEL':<8} {'PATTERN':<22} {'TASK':<35} {'SEED':>4} {'BLIP':<6}")
    print("=" * 90)

    for i, img in enumerate(SAMPLE_IMAGES, start=1):
        path = Path(img["path"])
        if not path.exists():
            print(f"[SKIP] {path}")
            continue

        blip_answer = ask_blip(processor, model, device, path, QUESTION)

        row = {
            "n":          i,
            "model":      img["model"],
            "pattern":    img["pattern"],
            "task":       img["task"],
            "seed":       img["seed"],
            "path":       img["path"],
            "blip":       blip_answer,
            "manual":     "",   # da compilare a mano
        }
        rows.append(row)

        print(f"{i:>3} {img['model']:<8} {img['pattern']:<22} "
              f"{img['task']:<35} {img['seed']:>4} {blip_answer:<6}")

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 90)
    print(f"CSV scritto in: {OUTPUT_CSV}")
    print()
   


if __name__ == "__main__":
    main()
