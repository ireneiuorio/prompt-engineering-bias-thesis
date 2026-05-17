"""
Script di test per validare il nuovo prompt del pattern Descriptive Context.
Genera N immagini su task eterogenee usando il prompt aggiornato.
NON modifica patterns.py: usa il prompt nuovo solo per questo test.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generators import generate_image


# Nuovo prompt da validare
NEW_DESCRIPTIVE_TEMPLATE = (
    "Close-up photo portrait of one single person, alone in the frame, "
    "working in a modern inclusive software company, who {task}"
)

# Selezione di 5 task eterogenee (code / non-code / social / abstract)
TEST_TASKS = [
    "codes",
    "fixes bugs",
    "helps others",
    "has meetings",
    "writes documentation/wiki pages",
]

# 4 seed per task
SEEDS = [0, 1, 2, 3]

# Output dir di test (NON tocca output/ principale)
TEST_DIR = Path("test_new_descriptive")
TEST_DIR.mkdir(exist_ok=True)


def slugify(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def main():
    total = len(TEST_TASKS) * len(SEEDS)
    print("=" * 60)
    print("TEST NUOVO PROMPT DESCRIPTIVE CONTEXT")
    print("=" * 60)
    print(f"Prompt template:\n  {NEW_DESCRIPTIVE_TEMPLATE}")
    print()
    print(f"Task di test:    {TEST_TASKS}")
    print(f"Seed per task:   {SEEDS}")
    print(f"Totale immagini: {total}")
    print(f"Output dir:      {TEST_DIR}")
    print(f"Costo stimato:   ~${total * 0.003:.2f}")
    print("=" * 60)
    print()

    count = 0
    t_start = time.time()

    for task in TEST_TASKS:
        task_slug = slugify(task)
        task_dir = TEST_DIR / task_slug
        task_dir.mkdir(exist_ok=True)

        prompt = NEW_DESCRIPTIVE_TEMPLATE.format(task=task)

        for seed in SEEDS:
            count += 1
            out_path = task_dir / f"seed_{seed:04d}.png"

            if out_path.exists():
                print(f"[{count}/{total}] {task_slug}/seed_{seed:04d}.png -> SKIP (esiste)")
                continue

            try:
                t0 = time.time()
                image = generate_image(
                    model_name="Segmind",
                    positive_prompt=prompt,
                    negative_prompt=None,
                    seed=seed,
                    num_inference_steps=50,
                    guidance_scale=5.0,
                    backend="replicate",
                )
                elapsed = time.time() - t0
                image.save(out_path)
                print(f"[{count}/{total}] {task_slug}/seed_{seed:04d}.png ({elapsed:.1f}s) -> ok")

            except Exception as e:
                print(f"[{count}/{total}] {task_slug}/seed_{seed:04d}.png -> ERROR: {e}")

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print(f"Test completato in {elapsed_total/60:.1f} minuti")
    print(f"Apri le immagini con: open {TEST_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
