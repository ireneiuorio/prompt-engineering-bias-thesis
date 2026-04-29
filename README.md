# Tesi Triennale: Analisi del Bias di Genere ed Etnia nei Modelli Text-to-Image

Studio dell'impatto dei prompt engineering pattern sul bias di genere ed etnia nei modelli text-to-image per task di Software Engineering.

## Struttura

- `config/`: definizione delle 56 task e dei 4 prompt pattern
- `scripts/`: script di generazione (locale via diffusers + cloud via Replicate)

## Esecuzione

```bash
# Generazione locale
python scripts/run_generation.py --backend local --model Segmind --num_images 10

# Generazione via Replicate (richiede REPLICATE_API_TOKEN)
python scripts/run_generation.py --backend replicate --model SDXL --num_images 10
```

## Pattern implementati

- Persona-based
- Descriptive Context
- Negative Prompting (Gender)
- Negative Prompting (Ethnicity)

Le baseline (General, SE, Fair) sono prese dal paper di d'Aloisio et al. (2026).
