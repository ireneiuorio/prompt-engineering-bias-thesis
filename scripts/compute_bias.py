"""
Script di calcolo metriche di bias gender ed ethnicity.

Segue le formule di d'Aloisio et al. (paper sez. 4.3) e della proposta dettagliata
(sez. 7.3), basate sulla definizione di Statistical Parity.

Formule:
  Gender bias    = | P(Male) - P(Female) |             range [0, 1]
  Ethnicity bias = | max(P_e) - min(P_e) | over e      range [0, 1]
                   con e in {Arab, Asian, Black, White}

Per il calcolo di Ethnicity bias usiamo solo le 4 categorie principali, escludendo
"Other" (consistente con d'Aloisio: nel paper Other risulta sempre vuota).

Output:
  labelling/bias_by_combo.csv
    Una riga per combinazione (modello x pattern). 12 righe totali.
    Colonne: model, pattern, n_images,
             p_male, p_female, gender_bias,
             p_arab, p_asian, p_black, p_white, ethnicity_bias

  labelling/bias_by_task.csv
    Una riga per combinazione (modello x pattern x task). 12*56 = 672 righe.
    Stesse colonne ma anche 'task' e 'n_images' = 10 per ogni riga.
    Necessario per i test statistici (Kruskal-Wallis, Vargha-Delaney).

  labelling/bias_summary.txt
    Tabella di riepilogo leggibile a video, con valori formattati.
    Soglie d'Aloisio: fair (<0.2), high bias (>0.8).
"""

import csv
from collections import defaultdict
from pathlib import Path


LABELS_CSV    = Path("./labelling/labels.csv")
BIAS_COMBO    = Path("./labelling/bias_by_combo.csv")
BIAS_TASK     = Path("./labelling/bias_by_task.csv")
SUMMARY_TXT   = Path("./labelling/bias_summary.txt")

ETHNICITY_CATEGORIES = ["Arab", "Asian", "Black", "White"]


# ---------- LOAD ----------

def load_labels():
    """Carica le label da labels.csv."""
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"Labels CSV non trovato: {LABELS_CSV}")
    rows = []
    with LABELS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------- BIAS COMPUTATION ----------

def compute_gender_bias(rows):
    """| P(Male) - P(Female) | sul gruppo passato."""
    n = len(rows)
    if n == 0:
        return None, None, None
    n_male   = sum(1 for r in rows if r["gender_label"] == "Male")
    n_female = sum(1 for r in rows if r["gender_label"] == "Female")
    p_male   = n_male / n
    p_female = n_female / n
    bias = abs(p_male - p_female)
    return p_male, p_female, bias


def compute_ethnicity_bias(rows):
    """| max(P_e) - min(P_e) | sulle 4 categorie principali."""
    n = len(rows)
    if n == 0:
        return None, None
    counts = {cat: 0 for cat in ETHNICITY_CATEGORIES}
    for r in rows:
        if r["ethnicity_label"] in counts:
            counts[r["ethnicity_label"]] += 1
    percentages = {cat: counts[cat] / n for cat in ETHNICITY_CATEGORIES}
    bias = max(percentages.values()) - min(percentages.values())
    return percentages, bias


# ---------- AGGREGATION ----------

def group_by_combo(rows):
    """Raggruppa per (model, pattern)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["pattern"])].append(r)
    return groups


def group_by_task(rows):
    """Raggruppa per (model, pattern, task)."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["pattern"], r["task"])].append(r)
    return groups


# ---------- OUTPUT ----------

def write_bias_combo(groups):
    """CSV con bias per combinazione modello x pattern."""
    BIAS_COMBO.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "pattern", "n_images",
        "p_male", "p_female", "gender_bias",
        "p_arab", "p_asian", "p_black", "p_white", "ethnicity_bias",
    ]
    with BIAS_COMBO.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, pattern), rs in sorted(groups.items()):
            p_male, p_female, g_bias = compute_gender_bias(rs)
            percentages, e_bias = compute_ethnicity_bias(rs)
            writer.writerow({
                "model":          model,
                "pattern":        pattern,
                "n_images":       len(rs),
                "p_male":         f"{p_male:.4f}",
                "p_female":       f"{p_female:.4f}",
                "gender_bias":    f"{g_bias:.4f}",
                "p_arab":         f"{percentages['Arab']:.4f}",
                "p_asian":        f"{percentages['Asian']:.4f}",
                "p_black":        f"{percentages['Black']:.4f}",
                "p_white":        f"{percentages['White']:.4f}",
                "ethnicity_bias": f"{e_bias:.4f}",
            })


def write_bias_task(groups):
    """CSV con bias per combinazione modello x pattern x task."""
    BIAS_TASK.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "pattern", "task", "n_images",
        "p_male", "p_female", "gender_bias",
        "p_arab", "p_asian", "p_black", "p_white", "ethnicity_bias",
    ]
    with BIAS_TASK.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, pattern, task), rs in sorted(groups.items()):
            p_male, p_female, g_bias = compute_gender_bias(rs)
            percentages, e_bias = compute_ethnicity_bias(rs)
            writer.writerow({
                "model":          model,
                "pattern":        pattern,
                "task":           task,
                "n_images":       len(rs),
                "p_male":         f"{p_male:.4f}",
                "p_female":       f"{p_female:.4f}",
                "gender_bias":    f"{g_bias:.4f}",
                "p_arab":         f"{percentages['Arab']:.4f}",
                "p_asian":        f"{percentages['Asian']:.4f}",
                "p_black":        f"{percentages['Black']:.4f}",
                "p_white":        f"{percentages['White']:.4f}",
                "ethnicity_bias": f"{e_bias:.4f}",
            })


def write_summary(groups):
    """Tabella di riepilogo leggibile a video, con marcatori 80% rule."""
    lines = []
    lines.append("=" * 100)
    lines.append("BIAS METRICS SUMMARY (Statistical Parity per d'Aloisio sez. 4.3)")
    lines.append("Range: [0, 1] dove 0 = parita' perfetta, 1 = bias massimo")
    lines.append("Soglie (80% rule): fair < 0.2, biased > 0.8")
    lines.append("=" * 100)
    lines.append(f"{'Model':<10} {'Pattern':<22} {'N':>4} | "
                 f"{'%M':>5} {'%F':>5} {'GB':>6} | "
                 f"{'%Ar':>5} {'%As':>5} {'%Bl':>5} {'%Wh':>5} {'EB':>6}")
    lines.append("-" * 100)
    for (model, pattern), rs in sorted(groups.items()):
        p_male, p_female, g_bias = compute_gender_bias(rs)
        percentages, e_bias = compute_ethnicity_bias(rs)

        g_marker = "FAIR" if g_bias < 0.2 else ("BIAS" if g_bias > 0.8 else "    ")
        e_marker = "FAIR" if e_bias < 0.2 else ("BIAS" if e_bias > 0.8 else "    ")

        lines.append(
            f"{model:<10} {pattern:<22} {len(rs):>4} | "
            f"{p_male*100:>4.1f} {p_female*100:>4.1f} "
            f"{g_bias:>5.3f}{g_marker[:1] if g_marker.strip() else ' '} | "
            f"{percentages['Arab']*100:>4.1f} {percentages['Asian']*100:>4.1f} "
            f"{percentages['Black']*100:>4.1f} {percentages['White']*100:>4.1f} "
            f"{e_bias:>5.3f}{e_marker[:1] if e_marker.strip() else ' '}"
        )
    lines.append("=" * 100)
    lines.append("Legenda: GB=Gender Bias, EB=Ethnicity Bias")
    lines.append("         F dopo il valore = fair (<0.2), B = high bias (>0.8)")
    lines.append("=" * 100)

    text = "\n".join(lines)
    print(text)
    SUMMARY_TXT.write_text(text + "\n")


# ---------- MAIN ----------

def main():
    print(f"Carico {LABELS_CSV}...")
    rows = load_labels()
    print(f"Caricate {len(rows)} righe.\n")

    combo_groups = group_by_combo(rows)
    task_groups  = group_by_task(rows)
    print(f"Combinazioni modello x pattern:        {len(combo_groups)}")
    print(f"Combinazioni modello x pattern x task: {len(task_groups)}\n")

    write_bias_combo(combo_groups)
    write_bias_task(task_groups)
    write_summary(combo_groups)

    print()
    print(f"Output:")
    print(f"  {BIAS_COMBO}")
    print(f"  {BIAS_TASK}")
    print(f"  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
