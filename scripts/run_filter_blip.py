"""
Script di filtro BLIP-VQA sull'intero dataset.
Una sola domanda: "Is this image showing a human?"
Coerente con d'Aloisio et al. e con la proposta dettagliata.
"""

import csv
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


MODEL_NAME = "Salesforce/blip-vqa-base"
OUTPUT_DIR = Path("./output")
RESULTS_CSV = Path("./labelling/filter_results.csv")
SUMMARY_TXT = Path("./labelling/filter_summary.txt")

MODELS = ["Segmind", "SDXL", "Flux"]
PATTERNS = ["persona", "descriptive", "negative_gender", "negative_ethnicity"]

QUESTION_HUMAN = "Is this image showing a human?"
SAVE_EVERY = 100


def setup_blip():
    print("Carico BLIP-VQA...")
    t0 = time.time()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    print(f"Caricato in {time.time() - t0:.1f}s (device={device})\n")
    return processor, model, device


def ask_blip(processor, model, device, image, question):
    inputs = processor(image, question, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=10)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()


def enumerate_images():
    images = []
    for model in MODELS:
        for pattern in PATTERNS:
            pattern_dir = OUTPUT_DIR / model / pattern
            if not pattern_dir.exists():
                print(f"[WARN] Cartella non trovata: {pattern_dir}")
                continue
            for task_dir in sorted(pattern_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                task = task_dir.name
                for img_path in sorted(task_dir.glob("seed_*.png")):
                    seed = int(img_path.stem.replace("seed_", ""))
                    images.append({
                        "model":      model,
                        "pattern":    pattern,
                        "task":       task,
                        "seed":       seed,
                        "image_path": str(img_path),
                    })
    return images


def load_already_processed():
    if not RESULTS_CSV.exists():
        return {}
    processed = {}
    with RESULTS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["pattern"], row["task"], int(row["seed"]))
            processed[key] = row
    return processed


def save_results(results):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "pattern", "task", "seed", "image_path",
                  "q1_answer", "passed"]
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_summary(results):
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    for r in results:
        key = (r["model"], r["pattern"])
        if key not in stats:
            stats[key] = {"total": 0, "passed": 0, "failed": 0}
        stats[key]["total"] += 1
        if str(r["passed"]) in ("True", "true"):
            stats[key]["passed"] += 1
        else:
            stats[key]["failed"] += 1

    lines = []
    lines.append("=" * 70)
    lines.append("FILTER SUMMARY (BLIP-VQA)")
    lines.append("=" * 70)
    lines.append(f"{'Model':<10} {'Pattern':<22} {'Total':>6} {'Pass':>6} "
                 f"{'Fail':>6} {'%pass':>7}")
    lines.append("-" * 70)
    for (model, pattern), s in sorted(stats.items()):
        pct = 100.0 * s["passed"] / s["total"] if s["total"] else 0
        lines.append(f"{model:<10} {pattern:<22} {s['total']:>6} "
                     f"{s['passed']:>6} {s['failed']:>6} {pct:>6.1f}%")
    lines.append("=" * 70)

    text = "\n".join(lines)
    print(text)
    SUMMARY_TXT.write_text(text + "\n")


def main():
    processor, model, device = setup_blip()

    all_images = enumerate_images()
    print(f"Immagini totali da processare: {len(all_images)}")

    already = load_already_processed()
    print(f"Gia' processate (resume):       {len(already)}\n")

    results = list(already.values())
    to_process = [
        img for img in all_images
        if (img["model"], img["pattern"], img["task"], img["seed"]) not in already
    ]
    print(f"Restano da processare:          {len(to_process)}\n")

    t_start = time.time()
    counter_since_save = 0

    try:
        for img in tqdm(to_process, desc="Filter Q1", unit="img"):
            try:
                image = Image.open(img["image_path"]).convert("RGB")
                q1 = ask_blip(processor, model, device, image, QUESTION_HUMAN)
                passed = (q1 == "yes")

                results.append({
                    "model":      img["model"],
                    "pattern":    img["pattern"],
                    "task":       img["task"],
                    "seed":       img["seed"],
                    "image_path": img["image_path"],
                    "q1_answer":  q1,
                    "passed":     passed,
                })

                counter_since_save += 1
                if counter_since_save >= SAVE_EVERY:
                    save_results(results)
                    counter_since_save = 0

            except Exception as e:
                print(f"\n[ERROR] {img['image_path']}: {type(e).__name__}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Salvo i risultati parziali...")

    save_results(results)
    write_summary(results)

    elapsed_total = time.time() - t_start
    print(f"\nTempo totale: {elapsed_total/60:.1f} minuti")
    print(f"Risultati: {RESULTS_CSV}")
    print(f"Summary:   {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
