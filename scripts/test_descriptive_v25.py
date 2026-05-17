"""
Test prompt Descriptive v2.5: compromesso tra v2 e v3.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.generators import generate_image


PROMPT_V25 = (
    "Close-up photo portrait of one single person looking at the camera, "
    "in a modern inclusive software company, who {task}"
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
TEST_DIR = Path("test_descriptive_v25")
TEST_DIR.mkdir(exist_ok=True)


def slugify(text):
    return text.lower().replace(" ", "_").replace("/", "_")


def main():
    total = len(MODELS_TO_TEST) * len(TEST_TASKS)
    print("=" * 60)
    print("TEST PROMPT DESCRIPTIVE V2.5")
    print("=" * 60)
    print(f"Prompt v2.5: {PROMPT_V25}")
    print(f"Modelli: {MODELS_TO_TEST}")
    print(f"Totale:  {total} immagini")
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

            prompt = PROMPT_V25.format(task=task)

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
    print(f"Test completato in {elapsed_total/60:.1f} minuti")


if __name__ == "__main__":
    main()
