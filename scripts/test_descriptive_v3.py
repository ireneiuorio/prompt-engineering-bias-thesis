"""
Test prompt Descriptive v3 con keyword rinforzate per inquadrature frontali.
Lo testiamo su SDXL (modello problematico) e poi su Segmind e Flux
per verificare che non rompa quello che gia funziona.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generators import generate_image


PROMPT_V3 = (
    "Front-facing close-up photo portrait of one single person "
    "looking at the camera, working in a modern inclusive software company, "
    "who {task}"
)

TEST_TASKS = [
    "codes",
    "fixes bugs",
    "helps others",
    "has meetings",
    "writes documentation/wiki pages",
]

MODELS_TO_TEST = ["SDXL", "Flux", "Segmind"]

MODEL_PARAMS = {
    "SDXL":    {"guidance_scale": 5.0, "num_inference_steps": 50},
    "Flux":    {"guidance_scale": 0.0, "num_inference_steps": 4},
    "Segmind": {"guidance_scale": 5.0, "num_inference_steps": 50},
}

SEED = 0
TEST_DIR = Path("test_descriptive_v3")
TEST_DIR.mkdir(exist_ok=True)


def slugify(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def main():
    total = len(MODELS_TO_TEST) * len(TEST_TASKS)
    print("=" * 60)
    print("TEST PROMPT DESCRIPTIVE V3")
    print("=" * 60)
    print(f"Prompt v3:")
    print(f"  {PROMPT_V3}")
    print()
    print(f"Modelli: {MODELS_TO_TEST}")
    print(f"Task:    {TEST_TASKS}")
    print(f"Seed:    {SEED}")
    print(f"Totale:  {total} immagini")
    print(f"Costo:   ~${total * 0.005:.2f}")
    print("=" * 60)
    print()

    count = 0
    t_start = time.time()

    for model in MODELS_TO_TEST:
        model_dir = TEST_DIR / model
        model_dir.mkdir(exist_ok=True)
        params = MODEL_PARAMS[model]

        for task in TEST_TASKS:
            count += 1
            task_slug = slugify(task)
            out_path = model_dir / f"{task_slug}.png"

            if out_path.exists():
                print(f"[{count}/{total}] {model}/{task_slug}.png -> SKIP")
                continue

            prompt = PROMPT_V3.format(task=task)

            try:
                t0 = time.time()
                image = generate_image(
                    model_name=model,
                    positive_prompt=prompt,
                    negative_prompt=None,
                    seed=SEED,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    backend="replicate",
                )
                elapsed = time.time() - t0
                image.save(out_path)
                print(f"[{count}/{total}] {model}/{task_slug}.png ({elapsed:.1f}s) -> ok")

            except Exception as e:
                print(f"[{count}/{total}] {model}/{task_slug}.png -> ERROR: {e}")

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Test completato in {elapsed_total/60:.1f} minuti")
    print(f"Apri con: open {TEST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
