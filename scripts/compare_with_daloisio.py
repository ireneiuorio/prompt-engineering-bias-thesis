"""
Script di confronto baseline d'Aloisio vs i tuoi 4 pattern.

Calcola i bias (gender + ethnicity) per le 3 baseline del paper
(General, SE, Fair) sui 3 modelli (Segmind, SDXL, Flux), usando
ESATTAMENTE le stesse formule applicate ai tuoi pattern:

    gender_bias    = |P(Male) - P(Female)|
    ethnicity_bias = |max(Pe) - min(Pe)|  over {Arab, Asian, Black, White}

Input:
  - /tmp/daloisio_csvs/G_{ethnicity,gender}_count_{segmind,xl,flux}.csv
  - /tmp/daloisio_csvs/SE_{ethnicity,gender}_count_{segmind,xl,flux}.csv
  - /tmp/daloisio_csvs/SE_{ethnicity,gender}_count_{segmind,xl,flux}_fair.csv
  - labelling/bias_by_combo.csv  (i tuoi 12 valori di bias)

Output:
  - labelling/baseline_daloisio.csv
        9 righe (3 modelli x 3 prompt baseline) con gender_bias ed
        ethnicity_bias di d'Aloisio.

  - labelling/comparison_table.csv
        21 righe = 3 modelli x 7 pattern (3 baseline + 4 tuoi).
        Tabella unificata per il confronto, con i delta vs SE.

  - labelling/comparison_summary.txt
        Tabella leggibile a video.
"""

import csv
from collections import defaultdict
from pathlib import Path


DALOISIO_DIR = Path("/tmp/daloisio_csvs")
YOUR_BIAS_CSV = Path("./labelling/bias_by_combo.csv")
OUT_BASELINE = Path("./labelling/baseline_daloisio.csv")
OUT_COMPARISON = Path("./labelling/comparison_table.csv")
OUT_SUMMARY = Path("./labelling/comparison_summary.txt")

# Mappa modello -> suffisso d'Aloisio
MODEL_SUFFIX = {
    "Segmind": "segmind",
    "SDXL":    "xl",
    "Flux":    "flux",
}

# I 3 prompt baseline di d'Aloisio
BASELINE_PROMPTS = ["General", "SE", "Fair"]

ETHNICITY_CATS = ["Arab", "Asian", "Black", "White"]


def get_filename(prompt, metric, model_suffix):
    """Costruisce il nome del file CSV di d'Aloisio."""
    if prompt == "General":
        return f"G_{metric}_count_{model_suffix}.csv"
    elif prompt == "SE":
        return f"SE_{metric}_count_{model_suffix}.csv"
    elif prompt == "Fair":
        return f"SE_{metric}_count_{model_suffix}_fair.csv"
    else:
        raise ValueError(f"Prompt sconosciuto: {prompt}")


def compute_bias_from_counts(csv_path, metric):
    """
    Carica un CSV count di d'Aloisio e calcola il bias aggregato
    sommando i conteggi su tutti i task.

    metric = 'gender' o 'ethnicity'
    """
    if not csv_path.exists():
        print(f"  [WARN] File non trovato: {csv_path}")
        return None

    totals = defaultdict(int)
    n_rows = 0
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k in ("", "Prompt"):  # colonne ignorate (id, Prompt text)
                    continue
                try:
                    totals[k] += int(v)
                except (ValueError, TypeError):
                    pass
            n_rows += 1

    if metric == "gender":
        n_male = totals.get("Male", 0)
        n_female = totals.get("Female", 0)
        total = n_male + n_female
        if total == 0:
            return None
        p_male = n_male / total
        p_female = n_female / total
        bias = abs(p_male - p_female)
        return {
            "n_rows":      n_rows,
            "n_total":     total,
            "p_male":      p_male,
            "p_female":    p_female,
            "gender_bias": bias,
        }

    elif metric == "ethnicity":
        total = sum(totals.get(c, 0) for c in ETHNICITY_CATS)
        if total == 0:
            return None
        ps = {c: totals.get(c, 0) / total for c in ETHNICITY_CATS}
        bias = max(ps.values()) - min(ps.values())
        return {
            "n_rows":         n_rows,
            "n_total":        total,
            "p_arab":         ps["Arab"],
            "p_asian":        ps["Asian"],
            "p_black":        ps["Black"],
            "p_white":        ps["White"],
            "ethnicity_bias": bias,
        }


def load_your_bias():
    """Carica i tuoi 12 valori di bias (4 pattern x 3 modelli)."""
    your_bias = {}
    with YOUR_BIAS_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = (r["model"], r["pattern"])
            your_bias[key] = {
                "n":              int(r["n_images"]),
                "p_male":         float(r["p_male"]),
                "p_female":       float(r["p_female"]),
                "gender_bias":    float(r["gender_bias"]),
                "p_arab":         float(r["p_arab"]),
                "p_asian":        float(r["p_asian"]),
                "p_black":        float(r["p_black"]),
                "p_white":        float(r["p_white"]),
                "ethnicity_bias": float(r["ethnicity_bias"]),
            }
    return your_bias


def main():
    # ---- Calcolo bias di d'Aloisio ----
    print("Calcolo bias di d'Aloisio dalle baseline (G/SE/Fair)...")
    baseline = {}  # (model, prompt) -> {gender_bias, ethnicity_bias, ...}
    for model, suffix in MODEL_SUFFIX.items():
        for prompt in BASELINE_PROMPTS:
            entry = {}
            for metric in ["gender", "ethnicity"]:
                f = DALOISIO_DIR / get_filename(prompt, metric, suffix)
                res = compute_bias_from_counts(f, metric)
                if res:
                    entry.update(res)
            if entry:
                baseline[(model, prompt)] = entry
                print(f"  OK {model}/{prompt}: "
                      f"gender_bias={entry.get('gender_bias', 'N/A'):.4f} "
                      f"ethnicity_bias={entry.get('ethnicity_bias', 'N/A'):.4f}")

    # ---- Scrivo baseline_daloisio.csv ----
    OUT_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "prompt", "n_total",
              "p_male", "p_female", "gender_bias",
              "p_arab", "p_asian", "p_black", "p_white", "ethnicity_bias"]
    with OUT_BASELINE.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (model, prompt), e in baseline.items():
            w.writerow({
                "model":          model,
                "prompt":         prompt,
                "n_total":        e.get("n_total", ""),
                "p_male":         f"{e.get('p_male', 0):.4f}" if 'p_male' in e else "",
                "p_female":       f"{e.get('p_female', 0):.4f}" if 'p_female' in e else "",
                "gender_bias":    f"{e.get('gender_bias', 0):.4f}" if 'gender_bias' in e else "",
                "p_arab":         f"{e.get('p_arab', 0):.4f}" if 'p_arab' in e else "",
                "p_asian":        f"{e.get('p_asian', 0):.4f}" if 'p_asian' in e else "",
                "p_black":        f"{e.get('p_black', 0):.4f}" if 'p_black' in e else "",
                "p_white":        f"{e.get('p_white', 0):.4f}" if 'p_white' in e else "",
                "ethnicity_bias": f"{e.get('ethnicity_bias', 0):.4f}" if 'ethnicity_bias' in e else "",
            })

    # ---- Carico i tuoi bias ----
    print("\nCarico i tuoi bias da labels.csv...")
    your_bias = load_your_bias()
    print(f"  Caricati {len(your_bias)} combinazioni tue.")

    # ---- Costruisco tabella di confronto ----
    print("\nCostruisco tabella di confronto...")
    YOUR_PATTERNS = ["persona", "descriptive", "negative_gender", "negative_ethnicity"]
    ALL_PATTERNS = BASELINE_PROMPTS + YOUR_PATTERNS  # 7 totali

    comparison_rows = []
    for model in ["Segmind", "SDXL", "Flux"]:
        # SE bias di riferimento per il delta
        se_g = baseline.get((model, "SE"), {}).get("gender_bias")
        se_e = baseline.get((model, "SE"), {}).get("ethnicity_bias")

        for pattern in ALL_PATTERNS:
            if pattern in BASELINE_PROMPTS:
                entry = baseline.get((model, pattern), {})
                src = "d'Aloisio"
            else:
                entry = your_bias.get((model, pattern), {})
                src = "thesis"

            if not entry:
                continue

            gb = entry.get("gender_bias", None)
            eb = entry.get("ethnicity_bias", None)

            # delta vs SE: negativo = miglioramento, positivo = peggioramento
            delta_g = (gb - se_g) if (gb is not None and se_g is not None) else None
            delta_e = (eb - se_e) if (eb is not None and se_e is not None) else None

            comparison_rows.append({
                "model":            model,
                "pattern":          pattern,
                "source":           src,
                "n":                entry.get("n_total") or entry.get("n", ""),
                "gender_bias":      f"{gb:.4f}" if gb is not None else "",
                "delta_gender_vs_se": f"{delta_g:+.4f}" if delta_g is not None else "",
                "ethnicity_bias":   f"{eb:.4f}" if eb is not None else "",
                "delta_ethnicity_vs_se": f"{delta_e:+.4f}" if delta_e is not None else "",
            })

    fields_comp = ["model", "pattern", "source", "n",
                   "gender_bias", "delta_gender_vs_se",
                   "ethnicity_bias", "delta_ethnicity_vs_se"]
    with OUT_COMPARISON.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_comp)
        w.writeheader()
        w.writerows(comparison_rows)

    # ---- Summary leggibile ----
    lines = []
    lines.append("=" * 100)
    lines.append("CONFRONTO BIAS: 3 baseline d'Aloisio (General/SE/Fair) + 4 nuovi pattern")
    lines.append("Delta vs SE: negativo = miglioramento, positivo = peggioramento")
    lines.append("=" * 100)
    lines.append(f"{'Model':<10} {'Pattern':<22} {'Src':<10} "
                 f"{'GB':>7} {'ΔGB/SE':>9} {'EB':>7} {'ΔEB/SE':>9}")
    lines.append("-" * 100)
    for r in comparison_rows:
        lines.append(
            f"{r['model']:<10} {r['pattern']:<22} {r['source']:<10} "
            f"{r['gender_bias']:>7} {r['delta_gender_vs_se']:>9} "
            f"{r['ethnicity_bias']:>7} {r['delta_ethnicity_vs_se']:>9}"
        )
        # separatore tra modelli
        if r is not comparison_rows[-1]:
            next_idx = comparison_rows.index(r) + 1
            if next_idx < len(comparison_rows) and comparison_rows[next_idx]["model"] != r["model"]:
                lines.append("-" * 100)
    lines.append("=" * 100)
    lines.append("\nLegenda:")
    lines.append("  Src='d'Aloisio' = baseline scaricata dal replication package del paper")
    lines.append("  Src='thesis'    = pattern nuovo proposto in questa tesi")
    lines.append("  GB  = gender bias  |P(Male) - P(Female)|")
    lines.append("  EB  = ethnicity bias |max(Pe) - min(Pe)| su {Arab,Asian,Black,White}")
    lines.append("  Δ = delta rispetto a SE (baseline di riferimento del paper)")

    text = "\n".join(lines)
    print()
    print(text)
    OUT_SUMMARY.write_text(text + "\n")

    print()
    print(f"Output:")
    print(f"  {OUT_BASELINE}")
    print(f"  {OUT_COMPARISON}")
    print(f"  {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
