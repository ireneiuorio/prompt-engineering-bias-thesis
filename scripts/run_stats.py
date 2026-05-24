"""
Test statistici sui bias per task (proposta dettagliata, sez. 8).

VERSIONE 2: parser dei prompt d'Aloisio reso robusto per gestire i
formati misti del replication package (underscore vs spazi, virgolette
extra, formulazioni diverse per xl/General).

Per ciascuna combinazione modello x metrica esegue 3 test:
  1. Kruskal-Wallis: test omnibus per verificare se esistono differenze
     significative tra i pattern (p < 0.05 indica almeno una differenza).
  2. Dunn con correzione Bonferroni: test post-hoc pairwise per
     identificare quali coppie di pattern differiscono significativamente.
  3. Vargha-Delaney A12: effect size per quantificare la magnitudo della
     differenza fra ogni coppia di pattern.

Output:
  labelling/stats_analysis_A_thesis_only.csv    Analisi A (solo tuoi pattern)
  labelling/stats_analysis_B_full.csv            Analisi B (con baseline)
  labelling/stats_summary.txt                    Summary leggibile a video
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal
import scikit_posthocs as sp


# ---- PATHS ----
BIAS_BY_TASK_CSV = Path("./labelling/bias_by_task.csv")
DALOISIO_DIR     = Path("./labelling/daloisio_baseline_data")

OUT_A   = Path("./labelling/stats_analysis_A_thesis_only.csv")
OUT_B   = Path("./labelling/stats_analysis_B_full.csv")
SUMMARY = Path("./labelling/stats_summary.txt")

MODELS   = ["Segmind", "SDXL", "Flux"]
METRICS  = ["gender_bias", "ethnicity_bias"]

THESIS_PATTERNS = ["persona", "descriptive", "negative_gender", "negative_ethnicity"]

MODEL_SUFFIX = {"Segmind": "segmind", "SDXL": "xl", "Flux": "flux"}

BASELINE_FILE = {
    "General": lambda metric, m: f"G_{metric.replace('_bias','')}_count_{m}.csv",
    "SE":      lambda metric, m: f"SE_{metric.replace('_bias','')}_count_{m}.csv",
    "Fair":    lambda metric, m: f"SE_{metric.replace('_bias','')}_count_{m}_fair.csv",
}
BASELINE_PATTERNS = ["General", "SE", "Fair"]

ETHNICITY_CATS = ["Arab", "Asian", "Black", "White"]


# ---------- PARSER ROBUSTO DEI PROMPT D'ALOISIO ----------

def normalize_task_name(s):
    """Normalizza un nome task: lowercase, spazi, no punteggiatura/virgolette."""
    if s is None:
        return None
    s = s.lower().replace("_", " ").strip()
    s = s.strip(".").strip()
    s = s.replace('"', '').replace("'", "").strip()
    return s


def extract_task_from_prompt(prompt_text):
    """Estrae il nome task dal prompt di d'Aloisio (gestisce tutti i formati misti)."""
    if not prompt_text:
        return None
    text = prompt_text.replace("_", " ").strip()
    text = text.lstrip('"').strip().lower()

    patterns = [
        r"photo portrait of a software engineer that (.+?)\.?$",
        r"photo portrait of a person that (.+?)\.?$",
        r"photo portrait of that person (.+?)\.?$",
        r"that (.+?)\.?$",
    ]
    for p in patterns:
        m = re.match(p, text)
        if m:
            return normalize_task_name(m.group(1))
    return None


# ---------- CARICAMENTO BIAS ----------

def load_your_bias_by_task():
    """Carica bias_by_task.csv (tuoi 4 pattern)."""
    rows = []
    with BIAS_BY_TASK_CSV.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "model":          r["model"],
                "pattern":        r["pattern"],
                "task":           normalize_task_name(r["task"]),
                "gender_bias":    float(r["gender_bias"]),
                "ethnicity_bias": float(r["ethnicity_bias"]),
            })
    return rows


def compute_daloisio_bias_by_task(csv_path, metric):
    """Calcola il bias per ogni task dal CSV count grezzo di d'Aloisio."""
    if not csv_path.exists():
        return None
    results = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_text = row.get("Prompt", "")
            task = extract_task_from_prompt(prompt_text)
            if not task:
                continue

            if metric == "gender_bias":
                male = int(row.get("Male", 0) or 0)
                female = int(row.get("Female", 0) or 0)
                total = male + female
                if total == 0:
                    continue
                bias = abs(male/total - female/total)
            elif metric == "ethnicity_bias":
                counts = {c: int(row.get(c, 0) or 0) for c in ETHNICITY_CATS}
                total = sum(counts.values())
                if total == 0:
                    continue
                ps = [counts[c]/total for c in ETHNICITY_CATS]
                bias = max(ps) - min(ps)
            else:
                continue
            results.append({"task": task, metric: bias})
    return results


def load_daloisio_bias_by_task():
    """Carica i bias per task delle 9 baseline d'Aloisio."""
    rows = []
    stats_loaded = defaultdict(int)
    for model, suffix in MODEL_SUFFIX.items():
        for pattern in BASELINE_PATTERNS:
            for metric in METRICS:
                fname = BASELINE_FILE[pattern](metric, suffix)
                fpath = DALOISIO_DIR / fname
                res = compute_daloisio_bias_by_task(fpath, metric)
                if res is None:
                    print(f"  [WARN] mancante: {fname}")
                    continue
                for r in res:
                    rows.append({
                        "model":   model,
                        "pattern": pattern,
                        "task":    r["task"],
                        metric:    r[metric],
                    })
                stats_loaded[(model, pattern, metric)] = len(res)

    # Diagnostica: quante righe caricate per ciascuna combo
    print("\nRighe caricate per combinazione (atteso 56 per ognuna):")
    for k, v in sorted(stats_loaded.items()):
        marker = "OK" if v == 56 else "WARN"
        print(f"  [{marker}] {k} -> {v} task")

    return rows


def merge_thesis_and_baseline(thesis_rows, baseline_rows):
    """Combina tuoi 4 pattern + 3 baseline d'Aloisio."""
    combined = defaultdict(lambda: defaultdict(dict))
    for r in thesis_rows:
        for metric in METRICS:
            combined[(r["model"], r["pattern"])][r["task"]][metric] = r[metric]
    for r in baseline_rows:
        metric_key = [m for m in METRICS if m in r][0]
        combined[(r["model"], r["pattern"])][r["task"]][metric_key] = r[metric_key]
    return combined


# ---------- VARGHA-DELANEY A12 ----------

def vargha_delaney_a12(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return float("nan")
    bigger = sum((xi > yj) for xi in x for yj in y)
    equal  = sum((xi == yj) for xi in x for yj in y)
    return (bigger + 0.5 * equal) / (n_x * n_y)


def a12_magnitude(a12):
    if np.isnan(a12):
        return "n/a"
    d = abs(a12 - 0.5)
    if d < 0.06: return "negligible"
    if d < 0.14: return "small"
    if d < 0.21: return "medium"
    return "large"


# ---------- TEST STATISTICI ----------

def run_tests_for_combo(combo_data, model, metric, patterns):
    groups = [combo_data[p] for p in patterns if len(combo_data.get(p, [])) > 0]
    valid_patterns = [p for p in patterns if len(combo_data.get(p, [])) > 0]
    if len(groups) < 2:
        return None

    try:
        kw_stat, kw_p = kruskal(*groups)
    except ValueError as e:
        # Caso: tutti i valori sono identici -> KW non applicabile
        return {
            "patterns": valid_patterns,
            "kw_stat": float("nan"),
            "kw_p": float("nan"),
            "kw_error": str(e),
            "dunn": None,
            "a12": {},
        }

    flat_vals = []
    flat_labels = []
    for p in valid_patterns:
        flat_vals.extend(combo_data[p])
        flat_labels.extend([p] * len(combo_data[p]))
    df = pd.DataFrame({"value": flat_vals, "group": flat_labels})
    dunn = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="bonferroni")

    a12_results = {}
    for i, p1 in enumerate(valid_patterns):
        for p2 in valid_patterns[i+1:]:
            a12 = vargha_delaney_a12(combo_data[p1], combo_data[p2])
            a12_results[(p1, p2)] = a12

    return {
        "patterns": valid_patterns,
        "kw_stat":  kw_stat,
        "kw_p":     kw_p,
        "dunn":     dunn,
        "a12":      a12_results,
    }


# ---------- WRITE CSV ----------

def write_csv_results(out_path, analyses):
    fields = [
        "analysis", "model", "metric",
        "kw_stat", "kw_p", "kw_significant",
        "comparison", "dunn_p", "dunn_significant",
        "a12", "a12_magnitude",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for (analysis_name, model, metric), res in analyses.items():
            if res is None:
                continue
            patterns = res["patterns"]
            dunn = res["dunn"]
            a12_dict = res["a12"]

            base_row = {
                "analysis":       analysis_name,
                "model":          model,
                "metric":         metric,
                "kw_stat":        f"{res['kw_stat']:.4f}" if not np.isnan(res['kw_stat']) else "n/a",
                "kw_p":           f"{res['kw_p']:.6f}" if not np.isnan(res['kw_p']) else "n/a",
                "kw_significant": "YES" if (not np.isnan(res['kw_p']) and res['kw_p'] < 0.05) else "NO",
            }
            if dunn is None:
                # Caso degenere
                w.writerow(base_row)
                continue
            for i, p1 in enumerate(patterns):
                for p2 in patterns[i+1:]:
                    dp = dunn.loc[p1, p2]
                    a12 = a12_dict.get((p1, p2), a12_dict.get((p2, p1)))
                    row = dict(base_row)
                    row.update({
                        "comparison":      f"{p1} vs {p2}",
                        "dunn_p":          f"{dp:.6f}",
                        "dunn_significant":"YES" if dp < 0.05 else "NO",
                        "a12":             f"{a12:.4f}",
                        "a12_magnitude":   a12_magnitude(a12),
                    })
                    w.writerow(row)


# ---------- SUMMARY ----------

def append_summary_lines(lines, title, analyses):
    lines.append("=" * 100)
    lines.append(title)
    lines.append("=" * 100)
    for (analysis_name, model, metric), res in analyses.items():
        if res is None:
            continue
        lines.append("")
        lines.append(f"[{model}] {metric}")
        if np.isnan(res["kw_p"]):
            lines.append(f"  Kruskal-Wallis: non applicabile ({res.get('kw_error', 'dati degeneri')})")
            continue
        marker = "**SIGNIFICATIVO**" if res["kw_p"] < 0.05 else "non significativo"
        lines.append(f"  Kruskal-Wallis: H={res['kw_stat']:.3f}, p={res['kw_p']:.4f} -> {marker}")
        if res["kw_p"] < 0.05:
            lines.append("  Coppie significative (Dunn-Bonferroni, p<0.05):")
            patterns = res["patterns"]
            dunn = res["dunn"]
            any_sig = False
            for i, p1 in enumerate(patterns):
                for p2 in patterns[i+1:]:
                    dp = dunn.loc[p1, p2]
                    if dp < 0.05:
                        a12 = res["a12"].get((p1, p2), res["a12"].get((p2, p1)))
                        mag = a12_magnitude(a12)
                        lines.append(f"     {p1:<22} vs {p2:<22} "
                                     f"p={dp:.4f}  A12={a12:.3f} ({mag})")
                        any_sig = True
            if not any_sig:
                lines.append("     (KW significativo ma nessuna coppia singola passa Bonferroni)")


# ---------- MAIN ----------

def main():
    print("Carico bias per task (tuoi 4 pattern)...")
    thesis_rows = load_your_bias_by_task()
    print(f"  {len(thesis_rows)} righe")

    print("\nCalcolo bias per task dalle baseline d'Aloisio...")
    baseline_rows = load_daloisio_bias_by_task()
    print(f"\n  Totale righe baseline: {len(baseline_rows)}")

    combined = merge_thesis_and_baseline(thesis_rows, baseline_rows)

    # ---- ANALISI A ----
    print("\n=== ANALISI A: solo i 4 pattern della tesi ===")
    analyses_A = {}
    for model in MODELS:
        for metric in METRICS:
            combo_data = {}
            for pattern in THESIS_PATTERNS:
                vals = []
                for task, mm in combined.get((model, pattern), {}).items():
                    if metric in mm:
                        vals.append(mm[metric])
                combo_data[pattern] = vals
            res = run_tests_for_combo(combo_data, model, metric, THESIS_PATTERNS)
            analyses_A[("A", model, metric)] = res
    write_csv_results(OUT_A, analyses_A)

    # ---- ANALISI B ----
    print("\n=== ANALISI B: 4 pattern + 3 baseline ===")
    all_patterns = BASELINE_PATTERNS + THESIS_PATTERNS
    analyses_B = {}
    for model in MODELS:
        for metric in METRICS:
            combo_data = {}
            for pattern in all_patterns:
                vals = []
                for task, mm in combined.get((model, pattern), {}).items():
                    if metric in mm:
                        vals.append(mm[metric])
                combo_data[pattern] = vals
            res = run_tests_for_combo(combo_data, model, metric, all_patterns)
            analyses_B[("B", model, metric)] = res
    write_csv_results(OUT_B, analyses_B)

    # ---- SUMMARY ----
    lines = []
    append_summary_lines(lines, "ANALISI A: confronto fra i 4 pattern della tesi", analyses_A)
    lines.append("")
    append_summary_lines(lines, "ANALISI B: confronto fra 4 pattern tesi + 3 baseline d'Aloisio", analyses_B)
    lines.append("")
    lines.append("=" * 100)
    lines.append("LEGENDA")
    lines.append("=" * 100)
    lines.append("  H        = statistica Kruskal-Wallis (chi-quadro)")
    lines.append("  p        = p-value (significativo se < 0.05)")
    lines.append("  A12      = Vargha-Delaney effect size (0.5 = no diff, >0.71/<0.29 = grande diff)")
    lines.append("  Magnit.  = negligible (<0.06), small (<0.14), medium (<0.21), large (>=0.21)")

    text = "\n".join(lines)
    print()
    print(text)
    SUMMARY.write_text(text + "\n")

    print()
    print("Output:")
    print(f"  {OUT_A}")
    print(f"  {OUT_B}")
    print(f"  {SUMMARY}")


if __name__ == "__main__":
    main()
