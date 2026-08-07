
# -----------------------------
# Training templates (0–7)
# -----------------------------
TRAIN_PROMPT_TEMPLATES_DKittyMorphology = [
    # 0 (paper-style, verbose spec)
    # (
    #     "You are a helpful optimization assistant that will help us generate a new robot morphology design.\n"
    #     "The goal is to optimize the morphological structure of a simulated robot: D’Kitty.\n"
    #     "For D’Kitty Morphology, we aim to optimize the body and leg structure of the robot to maximize its locomotion "
    #     "ability to navigate the robot to a fixed location (higher performance score is better).\n"
    #     "Each design consists of 56 continuous morphology parameters, grouped into 4 legs, with 14 parameters per leg.\n"
    #     "For each leg, the 14 parameters are ordered as follows: "
    #     "x (x-coordinate of the hip joint), y (y-coordinate of the hip joint), z (z-coordinate of the hip joint), "
    #     "a (angle of the hip joint), b (angle of the knee joint), "
    #     "hip center (center of the hip joint), hip range (range of the hip joint), "
    #     "knee center (center of the knee joint), knee range (range of the knee joint), "
    #     "hip size (size of the hip joint), knee size (size of the knee joint), "
    #     "foot center (center of the foot joint), foot range (range of the foot joint), foot size (size of the foot joint).\n"
    #     "You are given the following existing designs and their corresponding performance scores:\n"
    #     "{pairs}\n"
    #     "Please propose a new robot morphology design to maximize the performance score.\n"
    #     "Each feature should be a float number with three decimal places, including an explicit sign.\n"
    # ),
    (
        "You are a \"helpful optimization assistant\" for D\u2019Kitty robot morphology design in simulation.\n"
        "Goal: \"maximize the performance score\" (better locomotion/navigation to a fixed target; higher is better).\n"
        "Each design is \"56 continuous parameters\" (\"4 legs\" \u00d7 \"14 parameters per leg\").\n"
        "Per-leg parameters (in order): "
        "\"x\" (hip x), \"y\" (hip y), \"z\" (hip z), "
        "\"a\" (hip angle), \"b\" (knee angle), "
        "\"hip center\" (hip center), \"hip range\" (hip range), "
        "\"knee center\" (knee center), \"knee range\" (knee range), "
        "\"hip size\" (hip size), \"knee size\" (knee size), "
        "\"foot center\" (foot center), \"foot range\" (foot range), \"foot size\" (foot size).\n"
        "Existing designs and their \"performance scores\":\n"
        "{pairs}\n"
        "Please propose \"one new morphology design\" to maximize the score.\n"
        "Format: each value must be a signed float with three decimal places (e.g., \"+0.123\", \"-1.500\").\n"
    ),

    # 1 (spec sheet / structured)
    (
        "Role: optimization assistant for simulated robot design.\n"
        "System: D’Kitty morphology optimization. Objective: maximize the performance score (navigation locomotion).\n"
        "Design variables: 56 continuous parameters = 4 legs × 14 parameters per leg.\n"
        "Per-leg parameter order and meaning: "
        "x (hip x), y (hip y), z (hip z), a (hip angle), b (knee angle), "
        "hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "Observed data (design, performance score):\n"
        "{pairs}\n"
        "Task: propose one new 56-parameter morphology design that differs from the above designs and is expected to achieve "
        "a higher performance score.\n"
        "Render each number as a signed float with three decimal places.\n"
    ),

    # 2 (experiment log tone)
    (
        "Experiment log: D’Kitty Morphology optimization in simulation.\n"
        "We evaluate D’Kitty morphologies and record a performance score reflecting locomotion/navigation ability.\n"
        "Each morphology is encoded as 56 continuous values grouped into 4 legs (14 parameters per leg).\n"
        "Within each leg, the 14 parameters are: "
        "x/y/z (hip joint coordinates), a (hip joint angle), b (knee joint angle), "
        "hip center/range, knee center/range, "
        "hip size, knee size, foot center/range/size.\n"
        "Collected trials:\n"
        "{pairs}\n"
        "Based on these trials, propose a new morphology (56 floats) that is distinct from the listed ones and aims to improve "
        "the performance score.\n"
        "Use signed floats with three decimal places.\n"
    ),

    # 3 (teaching / infer pattern)
    (
        "You are helping us optimize a D’Kitty robot morphology by learning from previous examples.\n"
        "A morphology design is a 56-number vector describing 4 legs. Each leg has 14 parameters in a fixed order:\n"
        "x (hip x), y (hip y), z (hip z), a (hip angle), b (knee angle), "
        "hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "Here are existing morphologies and their performance scores:\n"
        "{pairs}\n"
        "Please infer useful patterns and propose a new 56-parameter morphology that is not identical to any example and is "
        "expected to achieve a higher performance score.\n"
        "Output each value as a signed float with three decimal places.\n"
    ),

    # 4 (checklist style)
    (
        "D’Kitty morphology optimization checklist:\n"
        "1) Objective: maximize the performance score (better locomotion/navigation).\n"
        "2) Design vector: 56 continuous parameters = 4 legs × 14 parameters.\n"
        "3) Per-leg order and semantics:\n"
        "   - x, y, z: hip joint coordinates\n"
        "   - a: hip joint angle\n"
        "   - b: knee joint angle\n"
        "   - hip center, hip range\n"
        "   - knee center, knee range\n"
        "   - hip size, knee size\n"
        "   - foot center, foot range, foot size\n"
        "4) Reference evaluations (design, score):\n"
        "{pairs}\n"
        "Now propose one new morphology design that differs from the above designs and is likely to obtain a higher score.\n"
        "Represent every parameter as a signed float with three decimal places.\n"
    ),

    # 5 (data-science framing)
    (
        "We are modeling a black-box relationship between D’Kitty morphology parameters and a performance score.\n"
        "Input: a 56-dimensional continuous morphology vector (4 legs × 14 parameters).\n"
        "Within each leg, parameters are ordered as: "
        "x (hip x), y (hip y), z (hip z), a (hip angle), b (knee angle), "
        "hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "Training samples (morphology, score):\n"
        "{pairs}\n"
        "Please propose a new morphology vector that is different from all given samples and is predicted to achieve a higher score.\n"
        "Use signed floats with three decimal places for all values.\n"
    ),

    # 6 (beat-the-best directive)
    (
        "Optimization Task: D’Kitty Morphology (simulator).\n"
        "Goal: propose a new morphology design that achieves a higher performance score than the best design shown.\n"
        "Representation: 56 continuous values = 4 legs × 14 parameters per leg.\n"
        "Per-leg parameters (in order): "
        "x, y, z (hip coordinates), a (hip angle), b (knee angle), "
        "hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "Given designs and scores:\n"
        "{pairs}\n"
        "Propose one new 56-value design that differs from the above and aims to outperform them.\n"
        "Use signed floats with three decimal places.\n"
    ),

    # 7 (output-format constrained like TFBind #7)
    (
        "Task: Generate one improved D’Kitty robot morphology design.\n"
        "Objective: maximize the performance score in simulation (navigation locomotion).\n"
        "A morphology design is a list of 56 continuous values grouped into 4 legs (14 parameters per leg).\n"
        "Per-leg parameter order and meaning: "
        "x/y/z (hip joint coordinates), a (hip joint angle), b (knee joint angle), "
        "hip center/range, knee center/range, "
        "hip size, knee size, foot center/range/size.\n"
        "Examples (design, score):\n"
        "{pairs}\n"
        "Return only the design inside |design-start| and |design-end|. "
        "Use exactly 56 signed float values with three decimal places.\n"
    ),
]

# -----------------------------
# Validation templates (8–9)
# -----------------------------
VALID_PROMPT_TEMPLATES_DKittyMorphology = [
    # 8 (validation, narrative but full semantics kept)
    (
        "We are optimizing the morphology of the D’Kitty robot in simulation.\n"
        "Each morphology design is represented by 56 continuous values grouped into 4 legs, "
        "with 14 parameters per leg in this order: "
        "x, y, z (hip coordinates), a (hip angle), b (knee angle), "
        "hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "Given the following evaluated designs and their performance scores:\n"
        "{pairs}\n"
        "Propose one new morphology design expected to achieve a higher score than the best example.\n"
        "Use signed floats with three decimal places.\n"
    ),

    # 9 (validation, compact but includes full per-leg mapping)
    (
        "D’Kitty Morphology optimization (maximize performance score).\n"
        "Design = 56 floats = 4 legs × 14 parameters per leg.\n"
        "Per-leg order: x, y, z, a, b, hip center, hip range, knee center, knee range, "
        "hip size, knee size, foot center, foot range, foot size.\n"
        "{pairs}\n"
        "Suggest one new 56-value design that is different from the above and likely to achieve a higher score.\n"
        "Use signed floats with three decimal places.\n"
    ),
]