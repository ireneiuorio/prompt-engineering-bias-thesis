"""
Funzioni di generazione immagini.
Supporta due backend: locale (diffusers su MPS/CUDA) e cloud (Replicate API).
La funzione generate_image() seleziona il backend in base al parametro 'backend'.
"""
import os
import time
import requests
from typing import Optional
from PIL import Image


# ============================================================================
# BACKEND 1: LOCALE (diffusers)
# ============================================================================

# Cache delle pipeline già caricate, per non ricaricare il modello ogni volta
_LOCAL_PIPELINES = {}


def _get_local_pipeline(model_name: str):
    """
    Carica e mantiene in cache la pipeline diffusers per un dato modello.
    """
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


def generate_local(
    model_name: str,
    positive_prompt: str,
    negative_prompt: Optional[str],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    """
    Genera un'immagine in locale usando diffusers.
    """
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


# ============================================================================
# BACKEND 2: REPLICATE (API cloud)
# ============================================================================

# Mappa modello → identificativo su Replicate
REPLICATE_MODELS = {
    "SDXL": "stability-ai/sdxl",
    "Segmind": "lucataco/segmind-vega",
    "SD2.1": "stability-ai/stable-diffusion",
    # Per il futuro:
    # "Flux-Schnell": "black-forest-labs/flux-schnell",
}


def generate_replicate(
    model_name: str,
    positive_prompt: str,
    negative_prompt: Optional[str],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
) -> Image.Image:
    """
    Genera un'immagine via API Replicate.
    Richiede REPLICATE_API_TOKEN nelle variabili d'ambiente.
    """
    import replicate

    if model_name not in REPLICATE_MODELS:
        raise ValueError(f"Modello Replicate sconosciuto: {model_name}")

    model_id = REPLICATE_MODELS[model_name]

    # Parametri compatibili con la maggior parte dei modelli SDXL-class su Replicate
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

    output = replicate.run(model_id, input=input_params)

    # Replicate restituisce una lista di URL (uno per ogni num_outputs)
    if isinstance(output, list):
        url = output[0]
    else:
        url = output

    # Replicate restituisce a volte un FileOutput object, a volte una stringa URL
    if hasattr(url, 'url'):
        url = url.url

    # Scarica subito l'immagine (gli URL Replicate scadono dopo ~1 ora)
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # Salva temporaneamente e apri come PIL
    tmp_path = f"/tmp/replicate_{seed}_{int(time.time()*1000)}.png"
    with open(tmp_path, "wb") as f:
        f.write(response.content)

    image = Image.open(tmp_path).copy()
    os.remove(tmp_path)

    return image


# ============================================================================
# FUNZIONE SMART: sceglie il backend
# ============================================================================

def generate_image(
    model_name: str,
    positive_prompt: str,
    negative_prompt: Optional[str],
    seed: int,
    num_inference_steps: int,
    guidance_scale: float,
    backend: str = "local",
) -> Image.Image:
    """
    Genera un'immagine usando il backend specificato.

    Args:
        model_name: 'SDXL', 'Segmind', 'SD2.1', ecc.
        positive_prompt: testo del prompt principale
        negative_prompt: testo del negative prompt (None se non usato)
        seed: seed per riproducibilità
        num_inference_steps: numero di passi di denoising
        guidance_scale: forza della guidance
        backend: 'local' (diffusers su Mac) o 'replicate' (API cloud)

    Returns:
        Immagine PIL.
    """
    if backend == "local":
        return generate_local(
            model_name, positive_prompt, negative_prompt,
            seed, num_inference_steps, guidance_scale,
        )
    elif backend == "replicate":
        return generate_replicate(
            model_name, positive_prompt, negative_prompt,
            seed, num_inference_steps, guidance_scale,
        )
    else:
        raise ValueError(f"Backend sconosciuto: {backend}. Usa 'local' o 'replicate'.")
