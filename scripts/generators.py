"""
Funzioni di generazione immagini.
Supporta due backend: locale (diffusers su MPS/CUDA) e cloud (Replicate API).
"""
import os
import time
import requests
from PIL import Image


_LOCAL_PIPELINES = {}


def _get_local_pipeline(model_name):
    import torch
    from diffusers import (
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        AutoPipelineForText2Image,
    )

    if model_name in _LOCAL_PIPELINES:
        return _LOCAL_PIPELINES[model_name]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16

    print(f"[generators] Caricamento {model_name} su {device}...")
    t0 = time.time()

    if model_name == "SD2.1":
        pipe = StableDiffusionPipeline.from_pretrained(
            "sd2-community/stable-diffusion-2-1",
            torch_dtype=dtype,
        )
    elif model_name == "SDXL":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
            use_safetensors=True,
        )
    elif model_name == "Segmind":
        pipe = AutoPipelineForText2Image.from_pretrained(
            "segmind/Segmind-Vega",
            torch_dtype=dtype,
            use_safetensors=True,
        )
    else:
        raise ValueError(f"Modello locale sconosciuto: {model_name}")

    pipe = pipe.to(device)
    print(f"[generators]   Caricato in {time.time()-t0:.1f}s")

    _LOCAL_PIPELINES[model_name] = pipe
    return pipe


def generate_local(model_name, positive_prompt, negative_prompt, seed, num_inference_steps, guidance_scale):
    import torch
    pipe = _get_local_pipeline(model_name)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    kwargs = {
        "prompt": positive_prompt,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    result = pipe(**kwargs)
    return result.images[0]


REPLICATE_MODELS = {
    "SDXL": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "Segmind": "lucataco/segmind-vega:1587d0a4f8897740324dea9d66cd9cfa8caaa2bcec3b5f8fb1ce032c67e53fea",
    "Flux": "black-forest-labs/flux-schnell:c846a69991daf4c0e5d016514849d14ee5b2e6846ce6b9d6f21369e564cfe51e",
}

REPLICATE_THROTTLE_SECONDS = 11
REPLICATE_MAX_RETRIES = 5

_last_request_time = 0.0


def _wait_for_rate_limit():
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < REPLICATE_THROTTLE_SECONDS:
        time.sleep(REPLICATE_THROTTLE_SECONDS - elapsed)
    _last_request_time = time.time()


def generate_replicate(model_name, positive_prompt, negative_prompt, seed, num_inference_steps, guidance_scale):
    import replicate

    if model_name not in REPLICATE_MODELS:
        raise ValueError(f"Modello Replicate sconosciuto: {model_name}")

    model_id = REPLICATE_MODELS[model_name]

    if model_name == "Flux":
        input_params = {
            "prompt": positive_prompt,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "num_outputs": 1,
            "aspect_ratio": "1:1",
            "output_format": "png",
        }
    else:
        input_params = {
            "prompt": positive_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "width": 1024,
            "height": 1024,
            "num_outputs": 1,
        }
        if negative_prompt:
            input_params["negative_prompt"] = negative_prompt

    last_error = None
    output = None
    for attempt in range(REPLICATE_MAX_RETRIES):
        try:
            _wait_for_rate_limit()
            output = replicate.run(model_id, input=input_params)
            break
        except Exception as e:
            error_str = str(e)
            last_error = e
            if "429" in error_str or "throttled" in error_str.lower():
                wait_time = 15 * (attempt + 1)
                print(f"  [rate limit] aspetto {wait_time}s prima di riprovare...")
                time.sleep(wait_time)
                continue
            elif "timeout" in error_str.lower() or "timed out" in error_str.lower():
                wait_time = 30 * (attempt + 1)
                print(f"  [timeout] cold start, aspetto {wait_time}s e riprovo...")
                time.sleep(wait_time)
                continue
            else:
                raise

    if output is None:
        raise last_error if last_error else RuntimeError("Generazione fallita senza errori espliciti")

    if isinstance(output, list):
        url = output[0]
    else:
        url = output

    if hasattr(url, 'url'):
        url = url.url

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    tmp_path = f"/tmp/replicate_{seed}_{int(time.time()*1000)}.png"
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    image = Image.open(tmp_path).copy()
    os.remove(tmp_path)

    return image


def generate_image(model_name, positive_prompt, negative_prompt, seed, num_inference_steps, guidance_scale, backend="local"):
    if backend == "local":
        return generate_local(model_name, positive_prompt, negative_prompt, seed, num_inference_steps, guidance_scale)
    elif backend == "replicate":
        return generate_replicate(model_name, positive_prompt, negative_prompt, seed, num_inference_steps, guidance_scale)
    else:
        raise ValueError(f"Backend sconosciuto: {backend}. Usa 'local' o 'replicate'.")
