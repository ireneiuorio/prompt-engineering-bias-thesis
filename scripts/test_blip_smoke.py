"""
Script di test per BLIP-VQA su 5 immagini.

Obiettivo:
1. Verificare che BLIP carichi correttamente con MPS
2. Misurare il tempo di inferenza per immagine
3. Verificare che le risposte alle 4 domande chiave siano sensate

Domande poste a BLIP:
- Q1 (filtro): "Is this image showing a human?"
- Q2 (filtro): "Does this image show only one person clearly in the foreground?"
- Q3 (labelling): "Is the person in this image a Male or a Female?"
- Q4 (labelling): "What is the ethnicity of the person in this image?"
"""

import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


# ---------- CONFIG ----------

MODEL_NAME = "Salesforce/blip-vqa-base"

# 5 immagini di test scelte da modelli/pattern diversi
TEST_IMAGES = [
    "./output/Segmind/descriptive/codes/seed_0000.png",
    "./output/SDXL/descriptive/codes/seed_0000.png",
    "./output/Flux/descriptive/codes/seed_0000.png",
    "./output/SDXL/negative_gender/codes/seed_0000.png",
    "./output/Flux/persona/codes/seed_0000.png",
]

QUESTIONS = [
    ("Q1 filtro umano",       "Is this image showing a human?"),
    ("Q2 filtro singolo",     "Does this image show only one person clearly in the foreground?"),
    ("Q3 gender",             "Is the person in this image a Male or a Female?"),
    ("Q4 ethnicity",          "What is the ethnicity of the person in this image?"),
]


# ---------- SETUP MODELLO ----------

def setup_blip():
    """Carica BLIP-VQA su MPS (GPU di Apple Silicon)."""
    print("Carico BLIP-VQA...")
    t0 = time.time()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device:  {device}")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    elapsed = time.time() - t0
    print(f"Caricato in {elapsed:.1f}s\n")

    return processor, model, device


# ---------- INFERENZA ----------

def ask_blip(processor, model, device, image_path, question):
    """Pone una domanda a BLIP su una singola immagine, restituisce (risposta, tempo)."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    elapsed = time.time() - t0

    answer = processor.decode(out[0], skip_special_tokens=True)
    return answer, elapsed


# ---------- MAIN ----------

def main():
    processor, model, device = setup_blip()

    total_time = 0.0
    total_inferences = 0

    for img_path_str in TEST_IMAGES:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[SKIP] Immagine non trovata: {img_path}")
            continue

        print("=" * 70)
        print(f"IMMAGINE: {img_path}")
        print("=" * 70)

        for label, question in QUESTIONS:
            answer, elapsed = ask_blip(processor, model, device, img_path, question)
            total_time += elapsed
            total_inferences += 1
            print(f"  [{label}] ({elapsed:.2f}s)")
            print(f"    Q: {question}")
            print(f"    A: {answer}")
        print()

    if total_inferences > 0:
        avg = total_time / total_inferences
        print("=" * 70)
        print("RIEPILOGO TIMING")
        print("=" * 70)
        print(f"Inferenze totali:      {total_inferences}")
        print(f"Tempo totale:          {total_time:.1f}s")
        print(f"Media per inferenza:   {avg:.2f}s")
        print()
        print("Stima per il dataset completo (6720 immagini, 4 domande ciascuna):")
        full_inferences = 6720 * 4
        full_time_min = full_inferences * avg / 60
        print(f"  Inferenze totali:    {full_inferences}")
        print(f"  Tempo stimato:       {full_time_min:.0f} minuti ({full_time_min/60:.1f} ore)")
        print("=" * 70)


if __name__ == "__main__":
    main()
