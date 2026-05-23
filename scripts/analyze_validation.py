"""
Script di analisi della validazione manuale di BLIP.

Coerente al 100% con il replication package di d'Aloisio et al.
(github.com/giordanoDaloisio/image-generation-bias, file
 analysis/Testing/Stats/analysis_blip_gender.py e analysis_blip_ethnicity.py).

Procedura:
- Merge tra validation_blind.csv (annotazioni manuali in cieco)
  e validation_with_blip.csv (risposte di BLIP).
- Per ogni combinazione (modello x pattern), calcolo:
    Accuracy   = accuracy_score(manual, blip)        [sklearn]
    Weighted F1 = f1_score(manual, blip, average='weighted')  [sklearn]
- Calcolo le stesse metriche anche aggregate per modello, per pattern,
  e overall (come fa d'Aloisio nel suo script).
- Per il threat to validity multi_subject:
    Calcolo accuracy stratificata (single vs multi).
    (Anche se nel campione i multi-subject sono solo 2, lo facciamo
     per coerenza con la Decisione 2.)

Nessuna esclusione di righe, nessun filtro. Pulizia identica a d'Aloisio.

Output:
  labelling/validation_results_gender.csv
    Accuracy + Weighted F1 per gender, per ogni combo + aggregati.

  labelling/validation_results_ethnicity.csv
    Accuracy + Weighted F1 per ethnicity, per ogni combo + aggregati.

  labelling/validation_summary.txt
    Tabella leggibile a video con tutti i risultati.

  labelling/validation_merged.csv
    Merge completo (manual + blip) per ulteriori analisi.
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score


VALIDATION_BLIND = Path("./labelling/validation_blind.csv")
VALIDATION_WBLIP = Path("./labelling/validation_with_blip.csv")
MERGED_CSV       = Path("./labelling/validation_merged.csv")
RESULTS_GENDER   = Path("./labelling/validation_results_gender.csv")
RESULTS_ETH      = Path("./labelling/validation_results_ethnicity.csv")
SUMMARY_TXT      = Path("./labelling/validation_summary.txt")


# ---------- LOAD & MERGE ----------

def read_csv_smart(path):
    """Legge un CSV provando ; e , come separatore."""
    with path.open("r", newline="", encoding="utf-8") as f:
        first = f.readline()
    sep = ";" if first.count(";") >= first.count(",") else ","
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=sep)
        rows = []
        for r in reader:
            rows.append({k: (v.strip() if v else "") for k, v in r.items()})
    return rows


def merge_data():
    """Merge tra annotazione manuale e label BLIP."""
    print(f"Carico {VALIDATION_BLIND}...")
    blind = read_csv_smart(VALIDATION_BLIND)
    print(f"  -> {len(blind)} righe")

    print(f"Carico {VALIDATION_WBLIP}...")
    wblip = read_csv_smart(VALIDATION_WBLIP)
    print(f"  -> {len(wblip)} righe")

    # Indicizzo wblip per (model, pattern, task, seed)
    blip_idx = {}
    for r in wblip:
        key = (r["model"], r["pattern"], r["task"], str(r["seed"]))
        blip_idx[key] = r

    merged = []
    missing = 0
    for r in blind:
        key = (r["model"], r["pattern"], r["task"], str(r["seed"]))
        if key not in blip_idx:
            missing += 1
            continue
        b = blip_idx[key]
        merged.append({
            "n":                r.get("n", ""),
            "model":            r["model"],
            "pattern":          r["pattern"],
            "task":             r["task"],
            "seed":             r["seed"],
            "path":             r.get("path", ""),
            "manual_gender":    r["manual_gender"],
            "blip_gender":      b["blip_gender"],
            "manual_ethnicity": r["manual_ethnicity"],
            "blip_ethnicity":   b["blip_ethnicity"],
            "is_multi_subject": r["is_multi_subject"],
        })

    if missing > 0:
        print(f"[WARN] {missing} righe blind non hanno match in wblip")

    print(f"Merge OK: {len(merged)} righe.\n")
    return merged


def save_merged(merged):
    MERGED_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = list(merged[0].keys())
    with MERGED_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(merged)


# ---------- CONTROLLO QUALITA' ----------

def check_data_quality(merged):
    """Stampa eventuali valori non standard nelle annotazioni manuali."""
    gender_ok    = {"Male", "Female"}
    ethnicity_ok = {"Arab", "Asian", "Black", "White"}

    bad_gender = [r for r in merged if r["manual_gender"] not in gender_ok]
    bad_eth    = [r for r in merged if r["manual_ethnicity"] not in ethnicity_ok]

    if bad_gender or bad_eth:
        print("=" * 70)
        print("ATTENZIONE: valori non standard nelle annotazioni manuali")
        print("=" * 70)
        if bad_gender:
            from collections import Counter
            c = Counter(r["manual_gender"] for r in bad_gender)
            print(f"  Gender non in {gender_ok}: {dict(c)}")
        if bad_eth:
            from collections import Counter
            c = Counter(r["manual_ethnicity"] for r in bad_eth)
            print(f"  Ethnicity non in {ethnicity_ok}: {dict(c)}")
        print()
        print("Per coerenza con d'Aloisio, i CSV manuali devono contenere")
        print("SOLO le categorie standard (Male/Female, White/Asian/Black/Arab).")
        print("Rivedi le immagini residue prima di proseguire.")
        print()
        return False
    return True


# ---------- METRICHE ----------

def compute_metrics(rows, manual_key, blip_key):
    """Compute accuracy + weighted F1 alla d'Aloisio."""
    if len(rows) == 0:
        return None, None, 0
    manual = [r[manual_key] for r in rows]
    blip   = [r[blip_key]   for r in rows]
    acc = accuracy_score(manual, blip)
    f1  = f1_score(manual, blip, average="weighted", zero_division=0)
    return acc, f1, len(rows)


# ---------- AGGREGAZIONI ----------

def group_by_combo(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["pattern"])].append(r)
    return g


def group_by_model(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["model"]].append(r)
    return g


def group_by_pattern(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["pattern"]].append(r)
    return g


# ---------- WRITE CSV RESULTS ----------

def write_results_csv(path, manual_key, blip_key, merged):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []

    # Per ogni combinazione modello x pattern
    for (model, pattern), rs in sorted(group_by_combo(merged).items()):
        acc, f1, n = compute_metrics(rs, manual_key, blip_key)
        rows_out.append({
            "type":     f"{model} / {pattern}",
            "n":        n,
            "accuracy": f"{acc:.4f}",
            "f1":       f"{f1:.4f}",
        })

    # Aggregato per modello
    for model, rs in sorted(group_by_model(merged).items()):
        acc, f1, n = compute_metrics(rs, manual_key, blip_key)
        rows_out.append({
            "type":     f"All {model}",
            "n":        n,
            "accuracy": f"{acc:.4f}",
            "f1":       f"{f1:.4f}",
        })

    # Aggregato per pattern
    for pattern, rs in sorted(group_by_pattern(merged).items()):
        acc, f1, n = compute_metrics(rs, manual_key, blip_key)
        rows_out.append({
            "type":     f"All {pattern}",
            "n":        n,
            "accuracy": f"{acc:.4f}",
            "f1":       f"{f1:.4f}",
        })

    # Overall
    acc, f1, n = compute_metrics(merged, manual_key, blip_key)
    rows_out.append({
        "type":     "OVERALL",
        "n":        n,
        "accuracy": f"{acc:.4f}",
        "f1":       f"{f1:.4f}",
    })

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["type", "n", "accuracy", "f1"])
        w.writeheader()
        w.writerows(rows_out)


# ---------- SUMMARY TXT ----------

def write_summary(merged):
    SUMMARY_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = []

    lines.append("=" * 90)
    lines.append("VALIDATION SUMMARY - BLIP vs Manual (d'Aloisio methodology)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("--- GENDER: Accuracy / Weighted F1 ---")
    lines.append(f"{'Type':<35} {'N':>4} {'Accuracy':>10} {'F1':>10}")
    lines.append("-" * 90)

    # combo
    for (model, pattern), rs in sorted(group_by_combo(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_gender", "blip_gender")
        lines.append(f"{model+' / '+pattern:<35} {n:>4} "
                     f"{acc:>10.4f} {f1:>10.4f}")
    lines.append("-" * 90)
    # by model
    for model, rs in sorted(group_by_model(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_gender", "blip_gender")
        lines.append(f"{'All '+model:<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")
    lines.append("-" * 90)
    # by pattern
    for pattern, rs in sorted(group_by_pattern(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_gender", "blip_gender")
        lines.append(f"{'All '+pattern:<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")
    # overall
    acc, f1, n = compute_metrics(merged, "manual_gender", "blip_gender")
    lines.append("-" * 90)
    lines.append(f"{'OVERALL GENDER':<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")

    lines.append("")
    lines.append("--- ETHNICITY: Accuracy / Weighted F1 ---")
    lines.append(f"{'Type':<35} {'N':>4} {'Accuracy':>10} {'F1':>10}")
    lines.append("-" * 90)
    for (model, pattern), rs in sorted(group_by_combo(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_ethnicity", "blip_ethnicity")
        lines.append(f"{model+' / '+pattern:<35} {n:>4} "
                     f"{acc:>10.4f} {f1:>10.4f}")
    lines.append("-" * 90)
    for model, rs in sorted(group_by_model(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_ethnicity", "blip_ethnicity")
        lines.append(f"{'All '+model:<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")
    lines.append("-" * 90)
    for pattern, rs in sorted(group_by_pattern(merged).items()):
        acc, f1, n = compute_metrics(rs, "manual_ethnicity", "blip_ethnicity")
        lines.append(f"{'All '+pattern:<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")
    acc, f1, n = compute_metrics(merged, "manual_ethnicity", "blip_ethnicity")
    lines.append("-" * 90)
    lines.append(f"{'OVERALL ETHNICITY':<35} {n:>4} {acc:>10.4f} {f1:>10.4f}")

    # Sezione multi-subject
    lines.append("")
    lines.append("--- THREAT TO VALIDITY: multi-subject stratification ---")
    multi_yes = [r for r in merged if r["is_multi_subject"].lower() == "yes"]
    multi_no  = [r for r in merged if r["is_multi_subject"].lower() == "no"]
    lines.append(f"Single-subject (No):  {len(multi_no)} immagini")
    lines.append(f"Multi-subject  (Yes): {len(multi_yes)} immagini")
    if multi_yes:
        a_g, f_g, _ = compute_metrics(multi_yes, "manual_gender", "blip_gender")
        a_e, f_e, _ = compute_metrics(multi_yes, "manual_ethnicity", "blip_ethnicity")
        lines.append(f"  Multi-subject  Gender    acc={a_g:.4f} F1={f_g:.4f}")
        lines.append(f"  Multi-subject  Ethnicity acc={a_e:.4f} F1={f_e:.4f}")
    if multi_no:
        a_g, f_g, _ = compute_metrics(multi_no, "manual_gender", "blip_gender")
        a_e, f_e, _ = compute_metrics(multi_no, "manual_ethnicity", "blip_ethnicity")
        lines.append(f"  Single-subject Gender    acc={a_g:.4f} F1={f_g:.4f}")
        lines.append(f"  Single-subject Ethnicity acc={a_e:.4f} F1={f_e:.4f}")

    lines.append("")
    lines.append("=" * 90)

    text = "\n".join(lines)
    print(text)
    SUMMARY_TXT.write_text(text + "\n")


# ---------- MAIN ----------

def main():
    merged = merge_data()
    save_merged(merged)

    if not check_data_quality(merged):
        print("Interrompo: prima sistema i valori non standard.")
        return

    write_results_csv(RESULTS_GENDER,
                      "manual_gender", "blip_gender", merged)
    write_results_csv(RESULTS_ETH,
                      "manual_ethnicity", "blip_ethnicity", merged)
    write_summary(merged)

    print()
    print("Output:")
    print(f"  {MERGED_CSV}")
    print(f"  {RESULTS_GENDER}")
    print(f"  {RESULTS_ETH}")
    print(f"  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
