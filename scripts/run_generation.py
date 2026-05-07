"""
Script principale per la generazione delle immagini.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.tasks import TASKS
from config.patterns import PATTERNS, build_prompts
from scripts.generators import generate_image


MODEL_HYPERPARAMS = {
    "SDXL":      {"guidance_scale": 5.0, "num_inference_steps": 50},
    "Segmind":   {"guidance_scale": 5.0, "num_inference_steps": 50},
    "Flux":      {"guidance_scale": 0.0, "num_inference_steps": 4},
    "SD2.1":     {"guidance_scale": 7.5, "num_inference_steps": 50},
}


def slugify(text):
    return (
        text.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "")
        .replace("?", "")
    )


def get_image_path(output_dir, model, pattern, task, seed):
    task_slug = slugify(task)
    return output_dir / model / pattern / task_slug / f"seed_{seed:04d}.png"


def run(backend, models, num_images, output_dir, seed_start=0):
    total_combinations = len(models) * len(PATTERNS) * len(TASKS) * num_images
    print("=" * 60)
    print("AVVIO GENERAZIONE")
    print("=" * 60)
    print(f"Backend:    {backend}")
    print(f"Modelli:    {models}")
    print(f"Task:       {len(TASKS)}")
    print(f"Pattern:    {len(PATTERNS)}")
    print(f"N img/comb: {num_images}")
    print(f"Totale:     {total_combinations} immagini")
    print(f"Output dir: {output_dir}")
    print("=" * 60)
    print()

    counter_done = 0
    counter_skipped = 0
    counter_errors = 0
    counter_generated = 0
    t_start = time.time()

    for model in models:
        if model not in MODEL_HYPERPARAMS:
            print(f"Modello {model} non in MODEL_HYPERPARAMS, salto.")
            continue

        params = MODEL_HYPERPARAMS[model]
        guidance = params["guidance_scale"]
        steps = params["num_inference_steps"]

        for pattern in PATTERNS:
            for task in TASKS:
                positive, negative = build_prompts(task, pattern)

                for i in range(num_images):
                    seed = seed_start + i
                    img_path = get_image_path(output_dir, model, pattern, task, seed)
                    counter_done += 1

                    if img_path.exists():
                        counter_skipped += 1
                        continue

                    img_path.parent.mkdir(parents=True, exist_ok=True)

                    try:
                        t0 = time.time()
                        image = generate_image(
                            model_name=model,
                            positive_prompt=positive,
                            negative_prompt=negative,
                            seed=seed,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            backend=backend,
                        )
                        elapsed = time.time() - t0

                        image.save(img_path)
                        counter_generated += 1

                        if counter_generated % 10 == 0 or counter_generated == 1:
                            done_pct = 100 * counter_done / total_combinations
                            print(
                                f"[{counter_done}/{total_combinations} = {done_pct:.1f}%] "
                                f"{model}/{pattern}/{slugify(task)} seed={seed} "
                                f"({elapsed:.1f}s) -> ok"
                            )

                    except Exception as e:
                        counter_errors += 1
                        print(
                            f"[ERROR] {model}/{pattern}/{slugify(task)} seed={seed}: "
                            f"{type(e).__name__}: {e}"
                        )
                        continue

    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print("COMPLETATO")
    print("=" * 60)
    print(f"Tempo totale:    {elapsed_total/60:.1f} minuti")
    print(f"Generate ora:    {counter_generated}")
    print(f"Gia esistenti:   {counter_skipped} (skip)")
    print(f"Errori:          {counter_errors}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Genera immagini per la tesi sul bias.")
    parser.add_argument("--backend", choices=["local", "replicate"], required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed_start", type=int, default=0)

    args = parser.parse_args()

    run(
        backend=args.backend,
        models=args.model,
        num_images=args.num_images,
        output_dir=Path(args.output_dir),
        seed_start=args.seed_start,
    )


if __name__ == "__main__":
    main()
