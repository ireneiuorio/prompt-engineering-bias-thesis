"""
Script di sampling stratificato per la validazione manuale di BLIP.

Procedura (coerente con d'Aloisio sez. 4.2 e proposta sez. 7.2):
- Per ogni combinazione (modello x pattern), N = 560 immagini.
- Formula di Cochran con correzione per popolazione finita, 95% confidence,
  10% error rate => sample size = 82 immagini per combinazione.
- Totale: 82 * 12 = 984 immagini.
- Sampling stratificato per task (per non concentrare il campione su pochi task).
- Random seed fisso per riproducibilita'.

Output:
  labelling/validation_blind.csv
    File "in cieco" da compilare manualmente.
    Colonne: n, model, pattern, task, seed, path,
             manual_gender, manual_ethnicity, is_multi_subject

  labelling/validation_with_blip.csv
    File "nascosto" con anche le risposte di BLIP.
    Da NON aprire prima di aver finito l'annotazione manuale.
    Verra' usato per il merge finale e per calcolare accuracy/F1.
"""

import csv
import random
from collections import defaultdict
from math import ceil
from pathlib import Path


LABELS_CSV       = Path("./labelling/labels.csv")
VALIDATION_BLIND = Path("./labelling/validation_blind.csv")
VALIDATION_WBLIP = Path("./labelling/validation_with_blip.csv")

# Cochran con N=560, conf=95%, err=10% -> 82
SAMPLES_PER_COMBO = 82
RANDOM_SEED       = 42  # fisso per riproducibilita'


def load_labels():
    """Carica tutte le label da labels.csv."""
    rows = []
    with LABELS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def stratified_sample(rows, target_size, seed=RANDOM_SEED):
    """
    Sampling stratificato per task all'interno di un gruppo (model, pattern).

    Strategia:
    - Conto i task disponibili (atteso: 56).
    - target_size = 82, distribuiti il piu' equamente possibile sui task.
      82 / 56 ≈ 1.46 -> alcuni task contribuiranno con 1, altri con 2.
    - 82 - 56 = 26 task estratti casualmente avranno 2 immagini, gli altri 30 ne avranno 1.
    - Per ogni task scelgo casualmente quale seed (su 10) prendere.
    """
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)

    tasks = sorted(by_task.keys())
    n_tasks = len(tasks)

    base = target_size // n_tasks   # 1
    extra = target_size - base * n_tasks  # 26

    # Quali task prendono base+1 elementi?
    extra_tasks = set(rng.sample(tasks, extra))

    sampled = []
    for t in tasks:
        n_take = base + (1 if t in extra_tasks else 0)
        task_imgs = by_task[t]
        if n_take > len(task_imgs):
            n_take = len(task_imgs)
        chosen = rng.sample(task_imgs, n_take)
        sampled.extend(chosen)

    return sampled


def main():
    print(f"Carico {LABELS_CSV}...")
    all_rows = load_labels()
    print(f"Caricate {len(all_rows)} righe totali.\n")

    # Raggruppa per (model, pattern)
    by_combo = defaultdict(list)
    for r in all_rows:
        by_combo[(r["model"], r["pattern"])].append(r)

    print(f"Combinazioni trovate: {len(by_combo)}")
    print(f"Target per combinazione: {SAMPLES_PER_COMBO} immagini")
    print(f"Totale atteso: {SAMPLES_PER_COMBO * len(by_combo)} immagini\n")

    # Campionamento stratificato per task dentro ogni combinazione
    sampled = []
    for combo, rs in sorted(by_combo.items()):
        s = stratified_sample(rs, SAMPLES_PER_COMBO, seed=RANDOM_SEED)
        print(f"  {combo[0]:<10} {combo[1]:<22} -> {len(s)} immagini campionate")
        sampled.extend(s)

    print(f"\nTotale campionato: {len(sampled)}\n")

    # Shuffle globale (cosi' nell'annotazione manuale non vedi
    # 82 immagini Segmind di fila, poi 82 SDXL, ecc.). Questo
    # evita anche bias d'annotazione "saturazione" su un modello.
    rng = random.Random(RANDOM_SEED + 1)
    rng.shuffle(sampled)

    # ---- Scrittura CSV "in cieco" ----
    VALIDATION_BLIND.parent.mkdir(parents=True, exist_ok=True)
    blind_fields = [
        "n", "model", "pattern", "task", "seed", "path",
        "manual_gender", "manual_ethnicity", "is_multi_subject",
    ]
    with VALIDATION_BLIND.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=blind_fields)
        writer.writeheader()
        for i, r in enumerate(sampled, start=1):
            writer.writerow({
                "n":                i,
                "model":            r["model"],
                "pattern":          r["pattern"],
                "task":             r["task"],
                "seed":             r["seed"],
                "path":             r["image_path"],
                "manual_gender":    "",
                "manual_ethnicity": "",
                "is_multi_subject": "",
            })

    # ---- Scrittura CSV "con BLIP" (da NON aprire prima della fine) ----
    wblip_fields = [
        "n", "model", "pattern", "task", "seed", "path",
        "blip_gender", "blip_ethnicity",
    ]
    with VALIDATION_WBLIP.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wblip_fields)
        writer.writeheader()
        for i, r in enumerate(sampled, start=1):
            writer.writerow({
                "n":              i,
                "model":          r["model"],
                "pattern":        r["pattern"],
                "task":           r["task"],
                "seed":           r["seed"],
                "path":           r["image_path"],
                "blip_gender":    r["gender_label"],
                "blip_ethnicity": r["ethnicity_label"],
            })

    print("=" * 70)
    print("CAMPIONAMENTO COMPLETATO")
    print("=" * 70)
    print(f"Da compilare:  {VALIDATION_BLIND}  ({len(sampled)} righe)")
    print(f"Riferimento:   {VALIDATION_WBLIP}  (NON aprire prima della fine!)")
    print()
    print("PROSSIMI PASSI:")
    print("  1. Apri il file 'validation_blind.csv' (Numbers, Excel, o LibreOffice).")
    print("  2. Per ogni riga:")
    print("       - Apri l'immagine indicata in 'path'.")
    print("       - Compila le 3 colonne (vedi cheat sheet).")
    print("  3. Salva (esportando in CSV se usi Numbers).")
    print()
    print("CHEAT SHEET (categorie):")
    print("  manual_gender:    Male | Female | Unclear")
    print("  manual_ethnicity: White | Asian | Black | Arab | Unclear")
    print("  is_multi_subject: yes | no")
    print("=" * 70)


if __name__ == "__main__":
    main()
