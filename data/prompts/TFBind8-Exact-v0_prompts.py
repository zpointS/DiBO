
# -----------------------------
# Training templates (0–7)
# -----------------------------
TRAIN_PROMPT_TEMPLATES_TFBind8 = [
    # 0
    (
        "You are a helpful optimization assistant that will help us generate a new "
        "length-8 optimal DNA sequence with maximum binding affinity with a particular "
        "transcription factor SIX6 REF R1.\n"
        "You are given the following existing DNA sequences and their corresponding binding affinities:\n"
        "{pairs}\n"
        "Please propose a new DNA sequence that is different from the existing DNA sequences "
        "and has higher binding affinity than the existing ones. The DNA sequences should be composed "
        "of A, C, G, and T, and differ in at least one position.\n"
    ),
    # 1
    (
        "The goal is to design an 8-length DNA sequence with maximum binding affinity "
        "to the transcription factor SIX6 REF R1.\n"
        "Below are several known sequences with their measured affinities:\n"
        "{pairs}\n"
        "Please infer the pattern and suggest a new sequence with a higher binding affinity. "
        "Ensure that the new design differs from all examples by at least one nucleotide.\n"
    ),
    # 2
    (
        "Optimization Task: Find a DNA sequence (length = 8) that binds more strongly to SIX6 REF R1.\n"
        "Training data (sequence, affinity):\n"
        "{pairs}\n"
        "Output a single new DNA sequence that achieves a higher affinity than any of the above sequences. "
        "Use only A, C, G, and T.\n"
    ),
    # 3
    (
        "You are assisting a molecular biologist in optimizing DNA binding sites.\n"
        "Given the following DNA sequences and their binding affinities with SIX6 REF R1:\n"
        "{pairs}\n"
        "Propose a new candidate sequence that should exhibit higher affinity. "
        "The output should only include the DNA sequence itself, composed of the letters A, C, G, and T.\n"
    ),
    # 4
    (
        "Below is a reference dataset of DNA sequences ranked by binding affinity "
        "to the transcription factor SIX6 REF R1.\n"
        "{pairs}\n"
        "Use the observed patterns to generalize beyond the dataset. Suggest a new sequence expected "
        "to outperform all listed examples.\n"
    ),
    # 5
    (
        "Experiment: We aim to improve the binding affinity of DNA sequences interacting with SIX6 REF R1.\n"
        "Observed results:\n"
        "{pairs}\n"
        "Please generate a new sequence likely to achieve higher affinity. "
        "The design must be unique and contain only A, C, G, or T.\n"
    ),
    # 6
    (
        "You are optimizing a biological sequence for higher binding affinity.\n"
        "Here are the current data samples:\n"
        "{pairs}\n"
        "Explain your reasoning internally and propose the final improved sequence. "
        "The DNA sequence must differ from all shown examples.\n"
    ),
    # 7
    (
        "Task: Generate one new DNA sequence (A, C, G, T only) with higher binding affinity "
        "than all given examples for SIX6 REF R1.\n"
        "{pairs}\n"
        "Return only the sequence inside |design-start| and |design-end|.\n"
    ),
]

# -----------------------------
# Validation templates (8–9)
# -----------------------------
VALID_PROMPT_TEMPLATES_TFBind8 = [
    # 8
    (
        "We are exploring DNA–protein interactions involving SIX6 REF R1.\n"
        "Below are several example sequences and their binding affinities:\n"
        "{pairs}\n"
        "Predict a novel sequence expected to bind more strongly than the best example. "
        "Your response should only include the design sequence.\n"
    ),
    # 9
    (
        "Consider the DNA binding experiment for transcription factor SIX6 REF R1.\n"
        "Given the measured affinities of the following sequences:\n"
        "{pairs}\n"
        "Suggest one new 8-base sequence with higher expected affinity than all given examples.\n"
    ),
]