"""
Smoke test SDXL: verifica che i 4 pattern producano output sensati
prima di lanciare il batch completo da 2240 immagini.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.patterns import build_prompts
from scripts.generators import generate_image


# 5 task eterogenee
TEST_TASKS = [
    "codes",
    "fixes bugs",
    "helps others",
    "has meetings",
    "writes documentation/wiki pages",
]

# Tutti e 4 i pattern
PATTERNS = ["persona", "descriptive", "negative_gender", "negative_ethnicity"]

# 1 seed solo per ridurre tempi/costi
SEED = 0

# Output dir
TEST_DIR = Path("test_sdxl_smoke")
TEST_DIR.mkdir(exist_ok=True)


def slugify(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def main():
    total = len(TEST_TASKS) * len(PATTERNS)
    print("=" * 60)
    print("SMOKE TEST SDXL")
    print("=" * 60)
    print(f"Task:    {TEST_TASKS}")
    print(f"Pattern: {PATTERNS}")
    print(f"Seed:    {SEED}")
    print(f"Totale:  {total} immagini")
    print(f"Costo:   ~${total * 0.005:.2f}")
    print("=" * 60)
    print()

    count = 0
    t_start = time.time()

    for pattern in PATTERNS:
        pattern_dir = TEST_DIR / pattern
        pattern_dir.mkdir(exist_ok=True)

        for task in TEST_TASKS:
            count += 1
            task_slug = slugify(task)
            out_path = pattern_dir / f"{task_slug}.png"

            if out_path.exists():
                print(f"[{count}/{total}] {pattern}/{task_slug}.png -> SKIP")
                continue

            positive, negative = build_prompts(task, pattern)

            try:
                t0 = time.time()
                image = generate_image(
                    model_name="SDXL",
                    positive_prompt=positive,
                    negative_prompt=negative,
                    seed=SEED,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    backend="replicate",
                )
                elapsed = time.time() - t0
                image.save(out_path)
                print(f"[{count}/{total}] {pattern}/{task_slug}.png ({elapsed:.1f}s) -> ok")

            except Exception as e:
                print(f"[{count}/{total}] {pattern}/{task_slug}.png -> ERROR: {e}")

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Test completato in {elapsed_total/60:.1f} minuti")
    print(f"Apri con: open {TEST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
