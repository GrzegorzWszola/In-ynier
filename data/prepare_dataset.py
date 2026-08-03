# This file contains script to transform different datasets into usable ones in single format
#   transforms each different method of emotion representation to the VAD
import csv
import re
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

LEXICON_FILE=""
LEXICON_OUTPUT=""
EMO_BANK_FILE=""
EMO_BANK_OUTPUT=""
GO_EMOTIONS_FILE=""
GO_EMOTIONS_OUTPUT=""

lexicon_file_path = os.getenv("LEXICON_FILE")
lexicon_output_file_path = os.getenv("LEXICON_OUTPUT")
emo_bank_file_path = os.getenv("EMO_BANK_FILE")
emo_output_file_path = os.getenv("EMO_BANK_OUTPUT")
go_emotions_file_path = os.getenv("GO_EMOTIONS_FILE")
go_output_file_path = os.getenv("GO_EMOTIONS_OUTPUT")

calculate_vad_from_text = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

# Emotions categorized by go emotions, the VAD values are taken for transformation from lexicon
#admiration,amusement,anger,annoyance,approval,caring,confusion,curiosity,desire,disappointment,disapproval,disgust,embarrassment,excitement,fear,gratitude,grief,joy,love,nervousness,optimism,pride,realization,relief,remorse,sadness,surprise,neutral

# Emotion map mapping VAD 1:1 with given emotion
emotion_map = {
    "admiration":           [0.938, 0.166, 0.452],
    "amusement":            [0.858, 0.674, 0.606],
    "anger":                [-0.666, 0.73, 0.314],
    "annoyance":            [-0.666, 0.436, -0.316],
    "approval":             [0.708, -0.08, 0.778],
    "caring":               [0.27, -0.062, 0.0],
    "confusion":            [-0.49, 0.334, -0.446],
    "curiosity":            [0.5, 0.51, -0.074],
    "desire":               [0.792, 0.384, 0.294],
    "disappointment":       [-0.77, -0.02, -0.328],
    "disapproval":          [-0.83, 0.102, -0.266],
    "disgust":              [-0.896, 0.55, -0.366],
    "embarrassment":        [-0.714, 0.37, -0.548],
    "excitement":           [0.792, 0.368, 0.462],
    "fear":                 [-0.854, 0.68, -0.414],
    "gratitude":            [0.77, -0.118, 0.22],
    "grief":                [-0.86, 0.28, -0.052],
    "joy":                  [0.96, 0.648, 0.588],
    "love":                 [0.996, 0.334, 0.234],
    "nervousness":          [-0.674, 0.83, -0.518],
    "neutral":              [-0.062, -0.632, -0.286],
    "optimism":             [0.898, 0.13, 0.628],
    "pride":                [0.458, 0.268, 0.696],
    "realization":          [0.108, 0.02, 0.672],
    "relief":               [0.688, -0.444, -0.038],
    "remorse":              [-0.794, 0.346, -0.246],
    "sadness":              [-0.896, -0.424, -0.672],
    "surprise":             [0.75, 0.75, 0.124]
}

# Also we can build VAD value based on amount of words that are present in lexicon and average their VAD value
# Later we compare the averaged value to the 1:1 mapping for more stability
def load_lexicon() -> dict:
    df = pd.read_csv(lexicon_file_path, sep="\t")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.dropna(subset=["term", "valence", "arousal", "dominance"])
    
    return {
        str(row.term).lower(): (row.valence, row.arousal, row.dominance)
        for row in df.itertuples()
    }

# To not skip multi worded phrases like {a bit} we introduce indexing for words starting with "a" etc.
def build_lexicon_index(lexicon: dict):
    unigrams = {k: v for k, v in lexicon.items() if " " not in k}
    mwe_by_first_word = {}
    for term, vad in lexicon.items():
        if " " in term:
            first_word = term.split(" ")[0]
            mwe_by_first_word.setdefault(first_word, []).append((term.split(" "), vad))
    return unigrams, mwe_by_first_word

# Function calculates VAD values based on the average VAD values of words in the sentence
def build_vad_for_sentence(sentence: str, unigrams: dict, mwe_by_first_word: dict):
    words = re.findall(r"[a-z']+", sentence.lower())
    matched = []
    i = 0
    while i < len(words):
        candidates = mwe_by_first_word.get(words[i], [])
        found = False

        for phrase_words, vad in sorted(candidates, key=lambda x: -len(x[0])):
            n = len(phrase_words)
            if words[i:i + n] == phrase_words:
                matched.append(vad)
                i += n
                found = True
                break
        if not found:
            if words[i] in unigrams:
                matched.append(unigrams[words[i]])
            i += 1

    if not matched:
        return (None, None, None)
    v, a, d = zip(*matched)
    return (sum(v) / len(v), sum(a) / len(a), sum(d) / len(d))

def estimate_vad_from_emotion_label(row, emotion_cols):
    active = [emotion_map[e] for e in emotion_cols if row.get(e, 0) == 1 and e in emotion_map]
    if not active:
        return (None, None, None)
    v, a, d = zip(*active)
    return (round(sum(v) / len(v), 4), round(sum(a) / len(a), 4), round(sum(d) / len(d), 4))

def transform_go_emotions():
    lexicon = load_lexicon()
    unigrams, mwe_by_first_word = build_lexicon_index(lexicon)

    df = pd.read_csv(go_emotions_file_path, sep=",")
    print(f"[Go_Emotions] File read and dataframe created: {go_emotions_file_path}")
    df = df[df["example_very_unclear"] == False]
    print("[Go_Emotions] Deleted unclear senteces")

    emotion_cols = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral",
    ]

    df[["estimated_V", "estimated_A", "estimated_D"]] = df.apply(
        lambda row: pd.Series(estimate_vad_from_emotion_label(row, emotion_cols)), axis=1
    )
    print("[Go_Emotions] Estimated VAD from given emotions labels")
    
    # Keep id, text, example_very_unclear + estimated VAD before trimming
    estimated_cols = ["id", "text", "example_very_unclear", "estimated_V", "estimated_A", "estimated_D"]  
    df = df[estimated_cols]
    # Calculate VAD values based on the given text
    if calculate_vad_from_text:
        df[["calculated_V", "calculated_A", "calculated_D"]] = df["text"].apply(
            lambda t: pd.Series(build_vad_for_sentence(t, unigrams, mwe_by_first_word))
        )
        df[["calculated_V", "calculated_A", "calculated_D"]] = df[["calculated_V", "calculated_A", "calculated_D"]].round(4)
        print("[Go_Emotions] Calculated VAD from read text")

    df.to_csv(go_output_file_path, index=False)

def transform_nrc_vad_lexicon():
    df = pd.read_csv(lexicon_file_path, sep='\t')
    print(f"[Lexicon] File read and dataframe created: {lexicon_file_path}")
    df = df.rename(columns={
        "valence": "V",
        "arousal": "A",
        "dominance": "D"
    })
    print(f"[Lexicon] File created and data transformed, saved to: {lexicon_output_file_path}")
    df.to_csv(lexicon_output_file_path, index=False)

def transform_emo_bank():
    lexicon = load_lexicon()
    unigrams, mwe_by_first_word = build_lexicon_index(lexicon)

    df = pd.read_csv(emo_bank_file_path)
    df = df.dropna(subset=["text"])
    print(f"[Emo_Bank] File read and dataframe created: {emo_bank_file_path}")
    # Transform the emo bank scale (1 to 5), to the lexicon scale (-1, 1), by the formula
    #   tranformed_X = (old_X - 3) * 0.5
    df[["estimated_V", "estimated_A", "estimated_D"]] = ((df[["V", "A", "D"]] - 3) * 0.5).round(4)
    print(f"[Emo_Bank] \"Estimated\" values from the VAD values")
    if calculate_vad_from_text:
        df[["calculated_V", "calculated_A", "calculated_D"]] = df["text"].apply(
            lambda t: pd.Series(build_vad_for_sentence(t, unigrams, mwe_by_first_word))
        )
        df[["calculated_V", "calculated_A", "calculated_D"]] = df[["calculated_V", "calculated_A", "calculated_D"]].round(4)
        print("[Emo_bank] Calculated VAD from read text")

    df = df.drop(columns=["V", "A", "D"])
    df.to_csv(emo_output_file_path, index=False)
    print(f"[Emo_Bank] File created and data transformed, saved to: {emo_output_file_path}")

if __name__ == "__main__":
    transform_emo_bank()
    transform_go_emotions()
    transform_nrc_vad_lexicon()
