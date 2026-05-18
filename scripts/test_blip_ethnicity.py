"""
Script di test per verificare il mapping ethnicity di BLIP-VQA.
"""

import time
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering


MODEL_NAME = "Salesforce/blip-vqa-base"

TEST_IMAGES = [
    f"./output/Flux/negative_ethnicity/codes/seed_{i:04d}.png"
    for i in range(10)
]

QUESTIONS = [
    ("aperta",    "What is the ethnicity of the person in this image?"),
    ("vincolata", "Is the person in this image Arab, Asian, Black, or White?"),
]


def setup_blip():
    print("Carico BLIP-VQA...")
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Caricato in {time.time() - t0:.1f}s ({device})\n")
    return processor, model, device


def ask_blip(processor, model, device, image_path, question):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
    return processor.decode(out[0], skip_special_tokens=True)


def main():
    processor, model, device = setup_blip()

    print(f"{'IMMAGINE':<15} {'APERTA':<25} {'VINCOLATA':<25}")
    print("=" * 70)

    for img_path_str in TEST_IMAGES:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[SKIP] {img_path}")
            continue

        seed_name = img_path.stem

        risposte = []
        for _, question in QUESTIONS:
            answer = ask_blip(processor, model, device, img_path, question)
            risposte.append(answer)

        print(f"{seed_name:<15} {risposte[0]:<25} {risposte[1]:<25}")


if __name__ == "__main__":
    main()
