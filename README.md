# Tesi Triennale — Analisi del Bias di Genere ed Etnia nei Modelli Text-to-Image

Studio dell'impatto dei **prompt engineering pattern** sul bias di genere ed etnia nei modelli text-to-image applicati a task di **Software Engineering**.

Il lavoro estende l'analisi di d'Aloisio et al. (2026) valutando quattro pattern aggiuntivi (Persona, Descriptive Context, Negative Prompting Gender, Negative Prompting Ethnicity) su 56 task e su 3 modelli generativi (SDXL, Segmind-SSD, FLUX.1-schnell).

**Immagini generate** (~6.720 immagini): [Google Drive](https://drive.google.com/drive/folders/12hOflrvXEodW-tOVyC_QcUgzEwECGa9I?usp=drive_link)

---

## Struttura del repository

```
.
├── config/
│   ├── tasks.py            # 56 task di Software Engineering (Treude et al.)
│   └── patterns.py         # 4 prompt pattern + baseline
├── scripts/
│   ├── run_generation.py       # generazione immagini (locale / Replicate)
│   ├── run_filter_blip.py      # filtro Q1 (immagine contiene una persona?)
│   ├── run_labelling_blip.py   # etichettatura BLIP-2 (genere + etnia)
│   ├── compute_bias.py         # calcolo bias di genere ed etnia
│   ├── compare_with_daloisio.py# confronto con baseline d'Aloisio
│   ├── run_stats.py            # test statistici (Kruskal-Wallis, Dunn, A12)
│   ├── sample_validation.py    # campionamento per validazione manuale
│   ├── analyze_validation.py   # accuracy / F1 BLIP vs annotazione manuale
│   └── generators.py           # wrapper diffusers + Replicate
├── data/
│   ├── labels.csv              # etichette BLIP per tutte le immagini
│   ├── filter_results.csv      # esito filtro Q1
│   ├── bias_by_task.csv        # bias per (modello, pattern, task)
│   ├── bias_by_combo.csv       # bias aggregato per (modello, pattern)
│   ├── comparison_table.csv    # confronto vs baseline SE d'Aloisio
│   ├── validation_*.csv        # dati e risultati della validazione manuale
│   └── stats_analysis_*.csv    # output dei test statistici
└── README.md
```

---

## Requisiti

- Python ≥ 3.10
- PyTorch con supporto CUDA (per esecuzione locale)
- `diffusers`, `transformers`, `accelerate`
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `scikit-posthocs`
- `replicate` (per esecuzione via cloud)

Installazione rapida:

```bash
pip install torch diffusers transformers accelerate \
            pandas numpy scipy scikit-learn scikit-posthocs replicate
```

---

## Esecuzione

### 1. Generazione delle immagini

```bash
# Backend locale (diffusers + GPU)
python scripts/run_generation.py --backend local --model Segmind --num_images 10

# Backend cloud (Replicate — richiede REPLICATE_API_TOKEN)
export REPLICATE_API_TOKEN=xxx
python scripts/run_generation.py --backend replicate --model SDXL --num_images 10
```

Modelli supportati: `SDXL`, `Segmind`, `Flux`, `SD2.1`.

### 2. Filtro e etichettatura

```bash
# Filtro Q1: l'immagine contiene una persona?
python scripts/run_filter_blip.py

# Etichettatura genere + etnia con BLIP-2
python scripts/run_labelling_blip.py
```

### 3. Calcolo del bias

```bash
python scripts/compute_bias.py
python scripts/compare_with_daloisio.py
```

### 4. Analisi statistica

```bash
python scripts/run_stats.py
```

Output: test di Kruskal-Wallis, post-hoc di Dunn con correzione Bonferroni, effect size di Vargha-Delaney (Â₁₂).

### 5. Validazione manuale

```bash
python scripts/sample_validation.py     # estrae il campione da annotare
python scripts/analyze_validation.py    # accuracy e F1 vs annotazione manuale
```

---

## Pattern implementati

| Pattern                     | Descrizione                                                                 |
|-----------------------------|-----------------------------------------------------------------------------|
| **Persona-based**           | `"Photo portrait of a professional who develops software and {task}"`       |
| **Descriptive Context**     | Descrizione ricca con contesto lavorativo esplicito                         |
| **Negative Prompting (Gender)** | Aggiunge keyword bilanciate + negative prompt su tratti stereotipati (Sami et al.) |
| **Negative Prompting (Ethnicity)** | Keyword da Naik & Nushi 2023 per contrastare stereotipi etnici             |

Le baseline (**General**, **SE**, **Fair**) sono riprese dal replication package di d'Aloisio et al. (2026) e non vengono rigenerate.

---

## Dataset generato

- **6.720 immagini** = 3 modelli × 4 pattern × 56 task × 10 seed
- Etichettatura automatica con **BLIP-2** (genere e etnia)
- **Validazione manuale in cieco** su un campione stratificato per stimare l'accuratezza dell'etichettatore

---

## Riferimenti principali

- d'Aloisio et al., *SustainDiffusion: Rendering Text-to-Image Generation More Sustainable Regarding Gender and Ethnic Bias* (2026)
- Sami et al., *Case Study Research on the Gender Bias in Software Engineering Roles Produced by Generative AI* (2023)
- Naik & Nushi, *Social Biases through the Text-to-Image Generation Lens* (AIES 2023)
- Treude et al., *Software Engineering Tasks Dataset*

---

## Autrice

Tesi di laurea triennale in Informatica — a.a. 2025/2026
