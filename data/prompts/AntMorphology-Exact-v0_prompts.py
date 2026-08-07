
# -----------------------------
# Training templates (0–7)
# -----------------------------
TRAIN_PROMPT_TEMPLATES_AntMorphology = [
    # 0
    # (
    #     "You are a helpful optimization assistant that will help us generate a new robot "
    #     "morphology design.\n"
    #     "The goal is to optimize the morphological structure of a simulated quadruped robot "
    #     "called Ant from OpenAI Gym, such that the robot can run as fast as possible.\n"
    #     "For each robot design, we represent the morphology using 60 continuous parameters, "
    #     "which are grouped into 4 legs, with 15 parameters per leg.\n"
    #     "For each leg, the parameters are ordered as follows: "
    #     "x (x-coordinate of the hip joint), "
    #     "y (y-coordinate of the hip joint), "
    #     "z (z-coordinate of the hip joint), "
    #     "a (angle of the hip joint), "
    #     "b (angle of the thigh joint), "
    #     "c (angle of the ankle joint), "
    #     "hip center (center of the hip joint), "
    #     "hip range (range of the hip joint), "
    #     "thigh center (center of the thigh joint), "
    #     "thigh range (range of the thigh joint), "
    #     "ankle center (center of the ankle joint), "
    #     "ankle range (range of the ankle joint), "
    #     "hip size (size of the hip joint), "
    #     "thigh size (size of the thigh joint), "
    #     "ankle size (size of the ankle joint).\n"
    #     "You are given the following existing robot morphology designs and their corresponding "
    #     "performance scores:\n"
    #     "{pairs}\n"
    #     "Please propose a new robot morphology design that is different from the existing ones "
    #     "and aims to achieve a higher performance score than the existing designs.\n"
    #     "Each morphology parameter should be a floating-point number rounded to three decimal places.\n"
    # ),
    (
        "You are a \"helpful optimization assistant\" for Ant robot morphology design in OpenAI Gym.\n"
        "The goal is to \"maximize the running performance\" of a simulated quadruped robot (\"Ant\").\n"
        "Each morphology is represented by \"60 continuous parameters\" (\"4 legs\" × \"15 parameters per leg\").\n"
        "For each leg, parameters are ordered as follows:\n"
        "\"x\" (hip x), \"y\" (hip y), \"z\" (hip z), "
        "\"a\" (hip angle), \"b\" (thigh angle), \"c\" (ankle angle), "
        "\"hip center\" (hip center), \"hip range\" (hip range), "
        "\"thigh center\" (thigh center), \"thigh range\" (thigh range), "
        "\"ankle center\" (ankle center), \"ankle range\" (ankle range), "
        "\"hip size\" (hip size), \"thigh size\" (thigh size), \"ankle size\" (ankle size).\n"
        "Existing designs and their \"performance scores\":\n"
        "{pairs}\n"
        "Please propose \"one new morphology design\" that is different from the above designs "
        "and is expected to achieve a \"higher performance score\".\n"
        "All parameters must be \"floating-point numbers\" rounded to \"three decimal places\".\n"
    ),
    # 1 (spec sheet / structured description)
    (
        "Role: optimization assistant for simulated robot design.\n"
        "System: Ant (quadruped) in OpenAI Gym. Objective: maximize the performance score by changing morphology.\n"
        "Design variables: 60 continuous parameters = 4 legs × 15 parameters per leg.\n"
        "Parameter semantics per leg (in order): "
        "x (hip joint x), y (hip joint y), z (hip joint z), "
        "a (hip joint angle), b (thigh joint angle), c (ankle joint angle), "
        "hip center (hip center), hip range (hip range), "
        "thigh center (thigh center), thigh range (thigh range), "
        "ankle center (ankle center), ankle range (ankle range), "
        "hip size (hip size), thigh size (thigh size), ankle size (ankle size).\n"
        "Observed data (design, performance score):\n"
        "{pairs}\n"
        "Task: propose one new morphology design that differs from the above designs and is expected to achieve a higher score. "
        "Round every value to three decimals.\n"
    ),

    # 2 (experiment log tone)
    (
        "Experiment log: Ant Morphology optimization in simulation.\n"
        "We evaluate different Ant morphologies and record a performance score for each.\n"
        "Each morphology is encoded as 60 continuous values (4 legs × 15 parameters).\n"
        "Within each leg, the 15 parameters are: "
        "x/y/z (hip joint coordinates), a/b/c (hip/thigh/ankle joint angles), "
        "hip center/range, thigh center/range, ankle center/range, "
        "and hip/thigh/ankle sizes.\n"
        "Collected trials:\n"
        "{pairs}\n"
        "Based on these trials, propose a new morphology design (60 floats, three decimals) that is different from the listed ones "
        "and aims to improve the performance score.\n"
    ),

    # 3 (teaching / “interpret then propose”)
    (
        "You are helping us optimize an Ant robot morphology by learning from previous examples.\n"
        "A morphology design is a 60-number vector describing 4 legs. Each leg has 15 parameters in this fixed order:\n"
        "x (hip x), y (hip y), z (hip z), "
        "a (hip angle), b (thigh angle), c (ankle angle), "
        "hip center, hip range, "
        "thigh center, thigh range, "
        "ankle center, ankle range, "
        "hip size, thigh size, ankle size.\n"
        "Here are existing morphologies and their performance scores:\n"
        "{pairs}\n"
        "Please infer useful patterns from the data and propose a new 60-parameter morphology that is not identical to any example "
        "and is expected to achieve a higher performance score. Use floats rounded to three decimals.\n"
    ),

    # 4 (checklist style)
    (
        "Ant robot morphology optimization checklist:\n"
        "1) Objective: maximize the performance score in simulation.\n"
        "2) Decision variables: 60 continuous parameters = 4 legs × 15 parameters.\n"
        "3) Per-leg parameter order and meaning:\n"
        "   - x, y, z: hip joint coordinates\n"
        "   - a, b, c: hip/thigh/ankle joint angles\n"
        "   - hip center, hip range\n"
        "   - thigh center, thigh range\n"
        "   - ankle center, ankle range\n"
        "   - hip size, thigh size, ankle size\n"
        "4) Reference evaluations (design, score):\n"
        "{pairs}\n"
        "Now propose one new morphology design that differs from the above designs and is likely to obtain a higher score. "
        "All 60 values must be floats rounded to three decimals.\n"
    ),

    # 5 (data-science / regression framing)
    (
        "We are modeling a black-box relationship between Ant morphology parameters and a performance score.\n"
        "Input: a 60-dimensional continuous morphology vector (4 legs × 15 parameters).\n"
        "Within each leg, parameters are ordered as: "
        "x (hip x), y (hip y), z (hip z), "
        "a (hip angle), b (thigh angle), c (ankle angle), "
        "hip center, hip range, "
        "thigh center, thigh range, "
        "ankle center, ankle range, "
        "hip size, thigh size, ankle size.\n"
        "Training samples (morphology, score):\n"
        "{pairs}\n"
        "Please propose a new morphology vector that is different from all given samples and is predicted to achieve a higher score. "
        "Round each number to three decimals.\n"
    ),

    # 6 (more directive, “beat the best”)
    (
        "Optimization Task: Ant Morphology (simulator).\n"
        "Goal: propose a new morphology design that achieves a higher performance score than the best design shown.\n"
        "Representation: 60 continuous values = 4 legs × 15 parameters per leg.\n"
        "Per-leg parameters (in order): "
        "x, y, z (hip joint coordinates), "
        "a (hip angle), b (thigh angle), c (ankle angle), "
        "hip center, hip range, "
        "thigh center, thigh range, "
        "ankle center, ankle range, "
        "hip size, thigh size, ankle size.\n"
        "Given designs and scores:\n"
        "{pairs}\n"
        "Propose one new 60-value design (floats, three decimals) that differs from the above and aims to outperform them.\n"
    ),

    # 7 (output-format constrained, still keeps semantics)
    (
        "Task: Generate one improved Ant robot morphology.\n"
        "Objective: maximize the performance score in simulation.\n"
        "A morphology design is a list of 60 continuous values grouped into 4 legs (15 parameters per leg).\n"
        "Per-leg parameter order and meaning: "
        "x/y/z (hip joint coordinates), a/b/c (hip/thigh/ankle angles), "
        "hip center/range, thigh center/range, ankle center/range, "
        "hip/thigh/ankle sizes.\n"
        "Examples (design, score):\n"
        "{pairs}\n"
        "Return only the design inside |design-start| and |design-end|. "
        "Use exactly 60 float values rounded to three decimal places.\n"
    ),
]

# -----------------------------
# Validation templates (8–9)
# -----------------------------
VALID_PROMPT_TEMPLATES_AntMorphology = [
    # 8 (validation, narrative but full semantics kept)
    (
        "We are optimizing the morphology of the Ant quadruped robot in simulation.\n"
        "Each morphology design is represented by 60 continuous values grouped into 4 legs, "
        "with 15 parameters per leg in this order: "
        "x, y, z (hip coordinates), a, b, c (hip/thigh/ankle angles), "
        "hip center, hip range, thigh center, thigh range, ankle center, ankle range, "
        "hip size, thigh size, ankle size.\n"
        "Given the following evaluated designs and their performance scores:\n"
        "{pairs}\n"
        "Propose one new morphology design expected to achieve a higher score than the best example. "
        "Round values to three decimals.\n"
    ),

    # 9 (validation, compact but still includes full per-leg mapping)
    (
        "Ant Morphology optimization (maximize performance score).\n"
        "Design = 60 floats = 4 legs × 15 parameters per leg.\n"
        "Per-leg order: x, y, z, a, b, c, hip center, hip range, thigh center, thigh range, "
        "ankle center, ankle range, hip size, thigh size, ankle size.\n"
        "{pairs}\n"
        "Suggest one new 60-value design that is different from the above and likely to achieve a higher score. "
        "Use floats rounded to three decimals.\n"
    ),
]