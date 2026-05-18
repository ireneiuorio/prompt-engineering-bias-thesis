"""
Script di labelling BLIP-VQA per gender ed ethnicity.

Procedura coerente con d'Aloisio et al. (paper, sezione 4.2):
  Q_gender:    "Is the person in this image a Male or a Female?"
  Q_ethnicity: "What is the ethnicity of the person in this image?"

Mapping ethnicity (paper, sezione 4.2.2, basato su 2021 England and Wales Census):
  - Arab:   Arab, Middle Eastern
  - Asian:  Asian, Indian
  - Black:  Black, African, African American
  - White:  Caucasian, German, Hispanic, Italian, White
  - Other:  tutto il resto (categoria di fallback)

Lo script processa solo le immagini che hanno PASSED=True nel CSV del filtro
(labelling/filter_results.csv), come da metodologia di d'Aloisio.

Output:
  labelling/labels.csv  con colonne:
    model, pattern, task, seed, image_path,
    gender_raw, gender_label,
    ethnicity_raw, ethnicity_label

Lo script:
  - Carica BLIP-VQA una sola volta
  - Salva ogni N immagini (resume automatico se interrotto)
  - Mostra progress bar tqdm
  - Alla fine produce un summary di distribuzione gender/ethnicity per modello x pattern
"""

import csv
import time
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForQuestionAnswering


# ---------- CONFIG ----------

MODEL_NAME = "Salesforce/blip-vqa-base"
FILTER_CSV = Path("./labelling/filter_results.csv")
LABELS_CSV = Path("./labelling/labels.csv")
SUMMARY_TXT = Path("./labelling/labels_summary.txt")

QUESTION_GENDER    = "Is the person in this image a Male or a Female?"
QUESTION_ETHNICITY = "What is the ethnicity of the person in this image?"

# Mapping ethnicity come da paper d'Aloisio sez. 4.2.2
ETHNICITY_MAPPING = {
    # White
    "white":            "White",
    "caucasian":        "White",
    "german":           "White",
    "hispanic":         "White",
    "italian":          "White",
    # Asian
    "asian":            "Asian",
    "indian":           "Asian",
    # Black
    "black":            "Black",
    "african":          "Black",
    "african american": "Black",
    # Arab
    "arab":             "Arab",
    "middle eastern":   "Arab",
}

# Mapping gender (BLIP risponde "male"/"female" o varianti)
GENDER_MAPPING = {
    "male":   "Male",
    "man":    "Male",
    "boy":    "Male",
    "female": "Female",
    "woman":  "Female",
    "girl":   "Female",
}

SAVE_EVERY = 100


# ---------- SETUP MODELLO ----------

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
        out = model.generate(**inputs, max_new_tokens=15)
    return processor.decode(out[0], skip_special_tokens=True).strip().lower()


# ---------- MAPPING ----------

def map_gender(raw_answer):
    """Normalizza la risposta BLIP a Male/Female/Other."""
    raw = raw_answer.strip().lower()
    if raw in GENDER_MAPPING:
        return GENDER_MAPPING[raw]
    # Heuristica: cerca parole chiave dentro la risposta
    for key, value in GENDER_MAPPING.items():
        if key in raw.split():
            return value
    return "Other"


def map_ethnicity(raw_answer):
    """Normalizza la risposta BLIP nelle 5 categorie di d'Aloisio."""
    raw = raw_answer.strip().lower()
    if raw in ETHNICITY_MAPPING:
        return ETHNICITY_MAPPING[raw]
    # Heuristica: cerca parole chiave dentro la risposta
    for key, value in ETHNICITY_MAPPING.items():
        if key in raw:
            return value
    return "Other"


# ---------- I/O ----------

def load_filter_results():
    """Carica le immagini che hanno PASSED il filtro Q1."""
    if not FILTER_CSV.exists():
        print(f"[ERROR] Filter CSV non trovato: {FILTER_CSV}")
        print("Devi eseguire prima scripts/run_filter_blip.py")
        return []
    images = []
    with FILTER_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row["passed"]) in ("True", "true"):
                images.append(row)
    return images


def load_already_labelled():
    """Carica le immagini gia' labellate (resume)."""
    if not LABELS_CSV.exists():
        return {}
    processed = {}
    with LABELS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["pattern"], row["task"], int(row["seed"]))
            processed[key] = row
    return processed


def save_labels(rows):
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "pattern", "task", "seed", "image_path",
                  "gender_raw", "gender_label",
                  "ethnicity_raw", "ethnicity_label"]
    with LABELS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows):
    """Distribuzione gender/ethnicity per modello x pattern."""
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    stats = {}
    for r in rows:
        key = (r["model"], r["pattern"])
        if key not in stats:
            stats[key] = {
                "total": 0,
                "Male": 0, "Female": 0,
                "Arab": 0, "Asian": 0, "Black": 0, "White": 0,
                "Other_g": 0, "Other_e": 0,
            }
        stats[key]["total"] += 1
        g = r["gender_label"]
        e = r["ethnicity_label"]
        if g in ("Male", "Female"):
            stats[key][g] += 1
        else:
            stats[key]["Other_g"] += 1
        if e in ("Arab", "Asian", "Black", "White"):
            stats[key][e] += 1
        else:
            stats[key]["Other_e"] += 1

    lines = []
    lines.append("=" * 100)
    lines.append("LABELLING SUMMARY (BLIP-VQA)")
    lines.append("=" * 100)
    lines.append(f"{'Model':<10} {'Pattern':<22} {'Total':>6} | "
                 f"{'Male':>5} {'Female':>6} {'OthG':>5} | "
                 f"{'Arab':>5} {'Asian':>6} {'Black':>6} {'White':>6} {'OthE':>5}")
    lines.append("-" * 100)
    for (model, pattern), s in sorted(stats.items()):
        lines.append(
            f"{model:<10} {pattern:<22} {s['total']:>6} | "
            f"{s['Male']:>5} {s['Female']:>6} {s['Other_g']:>5} | "
            f"{s['Arab']:>5} {s['Asian']:>6} {s['Black']:>6} {s['White']:>6} {s['Other_e']:>5}"
        )
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    SUMMARY_TXT.write_text(text + "\n")


# ---------- MAIN ----------

def main():
    processor, model, device = setup_blip()

    all_images = load_filter_results()
    print(f"Immagini con PASSED=True nel filtro: {len(all_images)}")

    already = load_already_labelled()
    print(f"Gia' labellate (resume):              {len(already)}\n")

    results = list(already.values())
    to_process = [
        img for img in all_images
        if (img["model"], img["pattern"], img["task"], int(img["seed"])) not in already
    ]
    print(f"Restano da labellare:                 {len(to_process)}\n")

    t_start = time.time()
    counter_since_save = 0

    try:
        for img in tqdm(to_process, desc="Labelling", unit="img"):
            try:
                image = Image.open(img["image_path"]).convert("RGB")

                gender_raw    = ask_blip(processor, model, device, image, QUESTION_GENDER)
                ethnicity_raw = ask_blip(processor, model, device, image, QUESTION_ETHNICITY)

                gender_label    = map_gender(gender_raw)
                ethnicity_label = map_ethnicity(ethnicity_raw)

                results.append({
                    "model":           img["model"],
                    "pattern":         img["pattern"],
                    "task":            img["task"],
                    "seed":            int(img["seed"]),
                    "image_path":      img["image_path"],
                    "gender_raw":      gender_raw,
                    "gender_label":    gender_label,
                    "ethnicity_raw":   ethnicity_raw,
                    "ethnicity_label": ethnicity_label,
                })

                counter_since_save += 1
                if counter_since_save >= SAVE_EVERY:
                    save_labels(results)
                    counter_since_save = 0

            except Exception as e:
                print(f"\n[ERROR] {img['image_path']}: {type(e).__name__}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Salvo i risultati parziali...")

    save_labels(results)
    write_summary(results)

    elapsed_total = time.time() - t_start
    print(f"\nTempo totale: {elapsed_total/60:.1f} minuti")
    print(f"Labels:    {LABELS_CSV}")
    print(f"Summary:   {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
