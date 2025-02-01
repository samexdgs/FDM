

SEED = 42


EXERCISE_2_LABEL = {
    "Strengthening": 0,
    "Neuromuscular": 1,
    "Flexibility": 2,
}

LABEL_2_EXERCISE = {
    0: "Strengthening",
    1: "Neuromuscular",
    2: "Flexibility"
}

LABEL_2_TREATMENT = {
    "Flexibility": [
        "Stretching exercises (e.g., calf stretches)",
        "Passive range of motion exercises",
        "Gait training focusing on foot clearance and flexibility"
    ],

    "Strengthening": [
        "Resistance band exercises",
        "Strengthening exercises for dorsiflexors (anterior tibialis)",
        "AFO (support for dorsiflexion and stability)",
        "Gait training focusing on stability and foot clearance"
    ],

    "Neuromuscular": [
        "Functional Electrical Stimulation (FES)",
        "Neuromuscular re-education",
        "Gait therapy to retrain proper walking patterns",
        "Pain management with medications if necessary"
    ]

}

CAT_COLS = ['Severity', 'Assistive Devices Used', 'Cause of Foot Drop', 'Pain Intensity',
            'Pain Frequency', 'Pain Type']

TEXT_COLS = ["All Info"]
