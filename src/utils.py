import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import random
import re
import string
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

import torch

from scipy.stats import chi2_contingency

from src.config import SEED, EXERCISE_2_LABEL, EXERCISE_2_TREATMENT
from src.paths import DATA_DIR


# -------------------------------------- DATA SIMULATION -----------------------------------------------

def set_seed(seed: int = SEED, seed_torch: bool = True):
    """
    This function sets the seed for PyTorch (both CPU and CUDA), NumPy, and the Python `random` module,
    enabling CuDNN benchmarking and deterministic algorithms. It is crucial for experiments requiring
    reproducibility, like model performance comparisons. Note that enabling CuDNN benchmarking and
    deterministic operations may impact performance and limit certain optimizations.

    Args:
        seed (int, optional):
            A non-negative integer that defines the random state. Defaults to 'SEED' value in config file.

        seed_torch (bool, optional): 
            If `True` sets the random seed for pytorch tensors, so pytorch module
            must be imported. Defaults to True.
    Returns:
        None
            This function does not return a value but sets the random seed for various libraries.

    Notes:
        - When using multiple GPUs, `th.cuda.manual_seed_all(seed)` ensures all GPUs are seeded, 
        crucial for reproducibility in multi-GPU setups.

    Example:
        >>> SEED = 42
        >>> set_seed(SEED)
    """
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def generate_noisy_data(sample_size: int):
    """Generate a noisy version of the dataset to increase complexity."""
    noisy_data = []

    severity_levels = ["Mild", "Moderate", "Severe"]
    causes_of_foot_drop = [
        "Postural imbalance", "Mild nerve weakness", "Spinal cord injury",
        "Trauma to lower limb", "Stroke", "Early-stage neuropathy",
        "Peripheral nerve damage", "Severe neuropathy"
    ]
    
    symptoms = [
        "Slight foot drop", "Occasional tripping", "Calf soreness, tightness sensation",
        "Difficulty lifting foot", "Noticeable foot drag", "Mild pain in the lower leg",
        "Complete inability to dorsiflex", "Severe pain", "Numbness in foot"
    ]
    
    triggers = [
        "Prolonged standing or sitting", "Fatigue", "Uneven surfaces",
        "Prolonged walking or standing", "Any weight-bearing activity"
    ]
    
    assistive_devices = ["None", "AFO", "Cane", "Wheelchair", "Walker"]
    exercise_classes = ["Flexibility", "Strengthening", "Neuromuscular"]
    pain_intensity_levels = ["None", "Mild", "Moderate", "Severe"]
    pain_frequency_levels = ["None", "Occasional", "Frequent", "Constant"]
    pain_type_levels = ["None", "Aching", "Burning", "Sharp", "Numbness"]

    for _ in range(sample_size):
        severity = random.choice(severity_levels)
        cause = random.choice(causes_of_foot_drop)

        # Adding noise by randomly shuffling symptoms and triggers
        symptom_list = random.sample(symptoms, k=random.randint(1, 4))
        trigger_list = random.sample(triggers, k=random.randint(1, 3))

        device = random.choice(assistive_devices)
        exercise = random.choice(exercise_classes)
        pain_intensity = random.choice(pain_intensity_levels)
        pain_frequency = random.choice(pain_frequency_levels)
        pain_type = random.choice(pain_type_levels)

        noisy_data.append([
            severity, cause, ", ".join(symptom_list), ", ".join(trigger_list),
            device, exercise, pain_intensity, pain_frequency, pain_type
        ]) 
        
    columns = ["Severity", "Cause of Foot Drop", "Symptoms", "Triggers", "Assistive Devices Used",
               "Exercise", "Pain Intensity", "Pain Frequency", "Pain Type"]

    return pd.DataFrame(noisy_data, columns=columns)


def simulate_dataset(sample_size: int = 2000, add_noise: bool = True, save: bool = False) -> pd.DataFrame:
    """
    Generate a synthetic dataset for foot drop cases with multiple symptoms and triggers per row.

    Args:
        sample_size (int, optional): Number of rows to generate. Defaults to 2000.

    Returns:
        pd.DataFrame: A Pandas dataframe
    """
    import random

    set_seed(seed=SEED, seed_torch=False)

    # Map causes to severity
    causes_to_severity = {
        "Mild": ["Postural imbalance", "Mild nerve weakness"],
        "Moderate": ["Trauma to lower limb", "Peripheral nerve damage", "Early-stage neuropathy"],
        "Severe": ["Stroke", "Severe neuropathy", "Spinal cord injury"]
    }

    # Define other attributes based on clinical notes
    symptoms = {
        "Mild": ["Slight foot drop", "Occasional tripping", "Calf soreness, tightness sensation"],
        "Moderate": ["Difficulty lifting foot", "Noticeable foot drag", "Mild pain in the lower leg"],
        "Severe": ["Complete inability to dorsiflex", "Severe pain", "Numbness in foot"]
    }

    triggers = {
        "Mild": ["Prolonged standing or sitting", "Fatigue"],
        "Moderate": ["Uneven surfaces", "Prolonged walking or standing"],
        "Severe": ["Any weight-bearing activity", "Prolonged standing or walking"]
    }

    assistive_devices = {
        "Mild": ["None"],
        "Moderate": ["AFO", "Cane"],
        "Severe": ["Wheelchair", "Walker"]
    }

    # Map severity to exercise class
    exercise_classes = {
        "Mild": "Flexibility",
        "Moderate": "Strengthening",
        "Severe": "Neuromuscular"
    }

    # Generate rows
    data = []
    for _ in range(sample_size):
        # Randomly select severity
        severity = random.choice(list(causes_to_severity.keys()))

        # Select a cause consistent with the severity
        cause = random.choice(causes_to_severity[severity])

        # Select multiple symptoms and triggers based on severity
        symptom_list = random.sample(
            symptoms[severity], k=random.randint(1, 3))
        trigger_list = random.sample(
            triggers[severity], k=random.randint(1, 2))

        # Select other attributes based on severity
        device = random.choice(assistive_devices[severity])
        exercise = exercise_classes[severity]

        # Simulate other columns (e.g., pain)
        pain_intensity = random.choice(["None", "Mild", "Moderate", "Severe"])
        pain_frequency = random.choice(
            ["None", "Occasional", "Frequent", "Constant"])
        pain_type = random.choice(
            ["None", "Aching", "Burning", "Sharp", "Numbness"])

        # Append row to the dataset
        data.append([
            severity, cause, ", ".join(symptom_list), ", ".join(trigger_list),
            device, exercise, pain_intensity, pain_frequency, pain_type
        ])

    # Create DataFrame
    columns = ["Severity", "Cause of Foot Drop", "Symptoms", "Triggers", "Assistive Devices Used",
               "Exercise", "Pain Intensity", "Pain Frequency", "Pain Type"
               ]

    df_clean = pd.DataFrame(data, columns=columns)
    
    
    # Add noise if required and concatenate
    if add_noise:
        df_noisy = generate_noisy_data(int(0.25 * sample_size))
        df_final = pd.concat([df_clean, df_noisy], ignore_index=True)
    else:
        df_final = df_clean


    if save:
        # Save to CSV (optional)
        df_final.to_csv(f"{DATA_DIR}/simulated_dataset.csv", index=False)

    return df_final


def create_treatment_data() -> pd.DataFrame:
    """
    """
    # Define the treatment plans based on the notes for each severity and specialist
    treatment_data = {
        "Severity": ["Mild", "Moderate", "Severe"],

        "Orthopedic Rehabilitation Specialist": [
            "AFO, Strengthening exercises for dorsiflexors, Gait training",
            "AFO, FES, Gait training, Strengthening exercises for dorsiflexors",
            "AFO, FES, Pain management with medications, Gait therapy"
        ],
        "Neurological Rehabilitation Specialist": [
            None,  # No explicit mention of mild cases in the provided clinical notes for neurology
            "AFO, FES, Gait training, Strengthening exercises",
            "FES, Neuroplasticity exercises, Gait training, Occupational therapy"
        ],
        "Physiotherapist": [
            "Daily dorsiflexor strengthening exercises, Gait retraining, Stretching, Posture correction",
            "AFO, Strengthening exercises, Gait training for stability",
            "AFO, FES, Pain management, Gait training with assistive devices"
        ]

    }

    return pd.DataFrame(treatment_data)


def get_all_treatment_data() -> pd.DataFrame:
    """
    Combine treatment plans from all 3 specialists into a single column for a unique treatment plan overview.

    Returns:
        pd.DataFrame: A DataFrame with a column that combines the treatment plans of all specialists.
    """
    # Create the treatment DataFrame
    treatment_df = create_treatment_data()

    # Function to combine treatments row-wise
    def combine_row_treatments(row):
        all_treatments = [
            row["Orthopedic Rehabilitation Specialist"],
            row["Neurological Rehabilitation Specialist"],
            row["Physiotherapist"],
        ]

        # Filter out None values and split into unique treatments
        unique_treatments = set(
            treatment.strip()
            for treatments in all_treatments if treatments
            for treatment in treatments.split(",")
        )
        return ", ".join(sorted(unique_treatments))

    # Apply the function to each row of the DataFrame
    treatment_df["Treatment Plan"] = treatment_df.apply(
        combine_row_treatments, axis=1)

    return treatment_df


def map_exercises_to_treatments(predicted_exercise: str) -> list:
    """
    Map predicted exercise classes to recommended treatments based on clinical notes.

    Args:
        predicted_exercise (str): Predicted exercise class (Flexibility, Strengthening, Neuromuscular).

    Returns:
        list: Recommended treatments corresponding to the exercise class.
    """
    # Refined treatment mapping based on clinical notes
    treatment_mapping = EXERCISE_2_TREATMENT

    # Return treatments for the predicted exercise class
    return treatment_mapping.get(predicted_exercise, [])


# ---------------------------------- PLOTS --------------------------------------

class EDAPlotter:
    """
    Exploratory Data Analysis Plot and Visualizations
    """

    def treatment_severity_plot(self, treatment_df):

        # Create a directed graph to represent the relationships
        G = nx.DiGraph()

        # Add nodes for severity levels and treatments
        severity_levels = treatment_df["Severity"].tolist()
        treatments = treatment_df["Treatment Plan"].tolist()

        for severity, treatment_plan in zip(severity_levels, treatments):
            for treatment in treatment_plan.split(", "):
                G.add_edge(severity, treatment)

        node_colors = [
            "blue" if node in severity_levels else "lightgreen" for node in G.nodes()]

        # Plot the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Layout for the graph
        nx.draw(
            G, pos, with_labels=True, node_color=node_colors, node_size=3000,
            font_size=10, font_weight="bold", edge_color="gray", alpha=0.8
        )
        plt.title(
            "Relationship Diagram: Foot Drop Treatment by Severity Level", fontsize=14)
        plt.show()

    def severity_plot(self, df, col="Severity"):
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x=col, color="blue")
        plt.title("Distribution of Severity Levels")
        plt.show()

    def class_label_dist(self, df, col="Exercise"):
        df[col] = df[col].apply(lambda x: x.split(":")[-1].strip())
        plt.figure(figsize=(8, 3))
        sns.countplot(data=df, x=col, color="blue")
        plt.xticks(rotation=30)
        plt.title("Distribution of Class Label")
        plt.xlabel("\nRecommended Exercise")
        plt.show()

    def univariate_subplot(self, nrows: int, ncols: int, df: pd.DataFrame):

        assert nrows * ncols == 4, "Subplot dimension specified didn't match"

        fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))

        axes[0, 0].set_title("Common Triggers")
        sns.countplot(data=df, x="Triggers", ax=axes[0, 0], color="blue")
        axes[0, 0].tick_params(axis='x', rotation=50)

        axes[0, 1].set_title("Distribution of Foot Drop Cause")
        sns.countplot(data=df, x="Cause of Foot Drop",
                      ax=axes[0, 1], color="blue")
        axes[0, 1].tick_params(axis='x', rotation=50)

        axes[1, 0].set_title("Assistive Devices Used per Exercise Class")
        sns.countplot(data=df, x="Assistive Devices Used",
                      hue="Exercise", ax=axes[1, 0], palette="viridis")

        axes[1, 1].set_title(
            "Distribution of  Assitive Device Used Across Severity")
        sns.countplot(data=df, x='Assistive Devices Used',
                      hue='Severity', palette="viridis")
        plt.xlabel('Assistive Devices Used')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        plt.show()


def chi_square(df: pd.DataFrame, v1: str, v2: str, alpha: float):
    """
    Perform statistical hypothesis testing using Chi-square

    Args:
        df (pd.DataFrame): Pandas dataframe
        v1 (str): categorical variable in df
        v2 (str): categorical variable in df
        alpha (float): level of significance
    """
    # Create a contingency table with margins
    contingency_table = pd.crosstab(df[v1], df[v2], margins=True)

    # Exclude the margin row and margin column
    tab_values = contingency_table.iloc[:-1, :-1].values

    # Perform the chi-square test
    stat, p, dof, expected_value = chi2_contingency(tab_values)
    print(f"P-value: {p:.5g}")

    # Hypothesis test conclusion
    if p < alpha:
        print("Decision: Reject Null Hypothesis")
        answer = "Yes"
    else:
        print("Decision: Do Not Reject Null Hypothesis")
        answer = "No"

    # Plotting grouped plot

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    sns.countplot(data=df, x=v1, hue=v2, ax=axes[0], palette="coolwarm")
    axes[0].set_xticks(range(len(axes[0].get_xticklabels())))
    axes[0].set_xticklabels(labels=axes[0].get_xticklabels(), rotation=60)
    axes[0].set_title(f"Countplot of {v1} with respect to {v2}")

    # Plotting percent stacked barplot

    ax1 = df.groupby(v1)[v2].value_counts(normalize=True).unstack()
    ax1.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title(f'Staked Barplot of {v1} and {v2}')

    plt.suptitle(
        f"P-value = {p:.5g}\n Is There A Significant Difference? = {answer}\n")
    plt.subplots_adjust(top=0.8)
    plt.tight_layout()
    plt.show()


# ---------------------------------- DATA PREPROCESSING --------------------------------------

def remove_duplicate(df, holdout_set=None):
    """
    Detect and remove duplicate rows in the dataset.
    """
    b4 = df.shape[0]
    print(f'{holdout_set} -- Before Removing Duplicate: {b4:,}')
    df.drop_duplicates(keep='first', inplace=True)
    after = df.shape[0]
    print(f'{holdout_set} -- After Removing Duplicate: {after:,}', '\n')

    if b4 == after:
        print(f"There are no duplicate rows in the {holdout_set}", '\n')
    else:
        print(str(b4 - after) + ' ' + "duplicate row(s) have been removed")


def encode_class_label(df, label):
    df[label] = df[label].apply(lambda row: EXERCISE_2_LABEL[row])
    print("Target label encoded succesfully")


# Prepare cleaning functions
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def preprocess_text(text: str) -> str:
    """
    Preprocesses the text column by performing the following:
    - Converts text to lowercase.
    - Removes HTML tags.
    - Removes punctuation.
    - Removes extra whitespace.

    Args:
        text (str): The text corpus to clean.

    Returns:
        str: The cleaned text.
    """
    # Lowercase and strip leading/trailing white space
    text = text.lower().strip()

    # Remove HTML tags
    text = re.compile("<.*?>").sub("", text)

    # Remove punctuation
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(" ", text)

    # Remove extra white space
    text = re.sub('\s+', " ", text)

    return text


def lexicon_process(text: str, stop_words: set, stemmer) -> str:
    """
    Processes text by:
    - Removing stopwords.
    - Applying stemming using the given stemmer.

    Args:
        text (str): The text to process.
        stop_words (set): A set of stopwords to remove.
        stemmer: A stemming object (e.g., SnowballStemmer).

    Returns:
        str: Processed text with stopwords removed and words stemmed.
    """
    filtered_text = []
    words = text.split(" ")
    for w in words:
        if w not in stop_words:
            filtered_text.append(stemmer.stem(w))
    text = " ".join(filtered_text)

    return text


def clean_text_data(df: pd.DataFrame, text_col: list):
    """
    Cleans specified text columns in a dataframe by:
    - Applying text preprocessing (lowercasing, removing HTML, punctuation, and whitespace).
    - Removing stopwords.
    - Applying stemming.

    Args:
        df (pd.DataFrame): The dataframe containing text columns.
        text_cols (list): A list of column names to clean.

    Returns:
        None: The dataframe is modified in place.
    """
    def clean_sentence(text, stop_words, stemmer):
        return lexicon_process(preprocess_text(text), stop_words, stemmer)
    for col in text_col:
        print(f"Cleaning '{col}' column ...")
    df[col] = [clean_sentence(item, stop_words=stop_words, stemmer=stemmer) for item in df[col].values]

    print("Dataset succesfully cleaned.")
