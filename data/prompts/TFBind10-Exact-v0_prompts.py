
# -----------------------------
# Training templates (0–7)
# -----------------------------
TRAIN_PROMPT_TEMPLATES_TFBind10 = [
    # 0 (paper-style, as provided)
    (
        "You are a helpful optimization assistant that will help us generate a new "
        "length-10 optimal DNA sequence with maximum binding affinity with a particular "
        "transcription factor SIX6 REF R1.\n"
        "You are given the following existing DNA sequences and their corresponding binding affinities:\n"
        "{pairs}\n"
        "Please propose a new DNA sequence that is different from the existing DNA sequences "
        "and has higher binding affinity than the existing DNA sequences. The DNA sequences should be "
        "composed of A, C, G, and T. The new DNA sequence should be different from the existing DNA sequences "
        "in at least 1 position.\n"
    ),

    # 1 (goal-first)
    (
        "The goal is to design a DNA sequence of length 10 with maximum binding affinity "
        "to the transcription factor SIX6 REF R1.\n"
        "Below are several known sequences with their measured affinities:\n"
        "{pairs}\n"
        "Please infer the pattern and suggest one new 10-base sequence with a higher binding affinity. "
        "Ensure it differs from every example by at least one nucleotide.\n"
    ),

    # 2 (task-title / concise)
    (
        "Optimization Task: Find a DNA sequence (length = 10) that binds more strongly to SIX6 REF R1.\n"
        "Training data (sequence, affinity):\n"
        "{pairs}\n"
        "Output a single new DNA sequence that achieves higher affinity than the sequences above. "
        "Use only A, C, G, and T, and change at least one position compared with every listed sequence.\n"
    ),

    # 3 (assistant-to-scientist framing)
    (
        "You are assisting a molecular biologist in optimizing DNA binding sites for SIX6 REF R1.\n"
        "Given the following DNA sequences and their binding affinities:\n"
        "{pairs}\n"
        "Propose a new candidate 10-base sequence expected to exhibit higher affinity. "
        "The sequence must be composed only of A, C, G, and T and differ from all examples by at least one nucleotide.\n"
    ),

    # 4 (dataset / ranked reference)
    (
        "Below is a reference dataset of 10-base DNA sequences evaluated for binding affinity "
        "to the transcription factor SIX6 REF R1.\n"
        "{pairs}\n"
        "Use the observed patterns to generalize beyond the dataset. Suggest a new sequence expected "
        "to outperform all listed examples. The design must use only A, C, G, T and be distinct from all examples.\n"
    ),

    # 5 (experiment-log style)
    (
        "Experiment: We aim to improve the binding affinity of length-10 DNA sequences interacting with SIX6 REF R1.\n"
        "Observed results (sequence, affinity):\n"
        "{pairs}\n"
        "Please generate one new sequence likely to achieve higher affinity. "
        "The design must be unique, contain only A/C/G/T, and differ from every shown example in at least one position.\n"
    ),

    # 6 (internal reasoning request, like TFBind8 #6)
    (
        "You are optimizing a biological sequence for higher binding affinity to SIX6 REF R1.\n"
        "Here are the current data samples:\n"
        "{pairs}\n"
        "Explain your reasoning internally and propose the final improved 10-base sequence. "
        "The sequence must be different from all listed sequences and contain only A, C, G, and T.\n"
    ),

    # 7 (output-format constrained, like TFBind8 #7)
    (
        "Task: Generate one new DNA sequence (A, C, G, T only) of length 10 with higher binding affinity "
        "than all given examples for SIX6 REF R1.\n"
        "{pairs}\n"
        "Return only the sequence inside |design-start| and |design-end|.\n"
    ),
]

# -----------------------------
# Validation templates (8–9)
# -----------------------------
VALID_PROMPT_TEMPLATES_TFBind10 = [
    # 8
    (
        "We are exploring DNA–protein interactions involving SIX6 REF R1.\n"
        "Below are several example 10-base sequences and their binding affinities:\n"
        "{pairs}\n"
        "Suggest one novel 10-base sequence expected to bind more strongly than the best example. "
        "Your response should only include the design sequence.\n"
    ),

    # 9
    (
        "Consider the DNA binding experiment for transcription factor SIX6 REF R1.\n"
        "Given the measured affinities of the following 10-base sequences:\n"
        "{pairs}\n"
        "Suggest one new length-10 sequence with higher expected affinity than all given examples.\n"
    ),
]