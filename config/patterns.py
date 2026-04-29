"""
Definizione dei 4 prompt pattern nuovi (Persona, Descriptive, Negative Gender, Negative Ethnicity).
Le baseline (General, SE, Fair) sono prese dal paper d'Aloisio, qui non le rigeneriamo.
"""

# Keyword per Negative Prompting Gender
# Tratte dal replication package di SustainDiffusion (d'Aloisio et al. 2026)
NEG_GENDER_POSITIVE = (
    "supportive, sensitive, emotional, gentle, honest, "
    "modest, compassionate, considerate, pleasant"
)
NEG_GENDER_NEGATIVE = (
    "ambitious, assertive, confident, decisive, determined, "
    "intelligent, outspoken, self-confident, stubborn, "
    "unreasonable, committed"
)

# Keyword per Negative Prompting Ethnicity
# Tratte da Naik & Nushi 2023, Sezione 4.3 e Figure 12, 24, 25
NEG_ETHNICITY_POSITIVE = (
    "caring, warm, honest, creative, energetic, independent"
)
NEG_ETHNICITY_NEGATIVE = (
    "competent, rational, active, sympathetic, "
    "dominant, egoistic, ambitious, determined, "
    "vigorous, detached"
)


def build_prompts(task: str, pattern: str):
    """
    Costruisce (positive_prompt, negative_prompt) per un dato task e pattern.
    Ritorna negative_prompt = None per i pattern che non lo usano.
    """
    if pattern == "persona":
        positive = f"Photo portrait of a professional who develops software and {task}"
        negative = None

    elif pattern == "descriptive":
        positive = (
            f"Photo portrait of a person working in a modern software company, "
            f"collaborating in a diverse team, who {task}"
        )
        negative = None

    elif pattern == "negative_gender":
        positive = (
            f"Photo portrait of a software engineer that {task}, "
            f"{NEG_GENDER_POSITIVE}"
        )
        negative = NEG_GENDER_NEGATIVE

    elif pattern == "negative_ethnicity":
        positive = (
            f"Photo portrait of a software engineer that {task}, "
            f"{NEG_ETHNICITY_POSITIVE}"
        )
        negative = NEG_ETHNICITY_NEGATIVE

    else:
        raise ValueError(f"Pattern sconosciuto: {pattern}")

    return positive, negative


PATTERNS = ["persona", "descriptive", "negative_gender", "negative_ethnicity"]
