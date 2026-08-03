# This file contains script to transform different datasets into usable ones in single format
#   transforms each different method of emotion representation to the VAD
import csv
import re
import pandas as pd

lexicon_file_path = "/home/grzeg/inz/data/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1.txt"
emo_bank_file_path = "/home/grzeg/inz/data/emo_bank/corpus/emobank.csv"
emo_output_file_path = "/home/grzeg/inz/data/emo_bank_dataset.csv"
go_emotions_file_path = "/home/grzeg/inz/data/go_emotions/archive/go_emotions_dataset.csv"
go_output_file_path = "/home/grzeg/inz/data/go_emotions_dataset.csv"

count_static_emotion_labels = True

def transform_emo_bank():
    df = pd.read_csv(emo_bank_file_path)
    # Transform the emo bank scale (1 to 5), to the lexicon scale (-1, 1), by the formula
    #   tranformed_X = (old_X - 3) * 0.5
    df[["V", "A", "D"]] = ((df[["V", "A", "D"]] - 3) * 0.5).round(4)
    df.to_csv(emo_output_file_path, index=False)

    print(f"[Emo_Bank] File created and data transformed, saved to: {emo_output_file_path}")

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

def estimate_vad_from_emotion_label():
    return

def transform_go_emotions():
    lexicon = load_lexicon()
    unigrams, mwe_by_first_word = build_lexicon_index(lexicon)

    df = pd.read_csv(go_emotions_file_path, sep=",")
    print("[Go_Emotions] Opened files")
    df = df[df["example_very_unclear"] == False]
    print("[Go_Emotions] Deleted unclear senteces")

    if count_static_emotion_labels:
        emotion_cols = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring",
            "confusion", "curiosity", "desire", "disappointment", "disapproval",
            "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
            "joy", "love", "nervousness", "optimism", "pride", "realization",
            "relief", "remorse", "sadness", "surprise", "neutral",
        ]

        df[["s_V", "s_A", "s_D"]] = df.apply(
            lambda t: pd.Series()
        )

    df = df.iloc[:, :3]
    # Calculate VAD values based on the given text
    df[["c_V", "c_A", "c_D"]] = df["text"].apply(
        lambda t: pd.Series(build_vad_for_sentence(t, unigrams, mwe_by_first_word))
    )
    df[["c_V", "c_A", "c_D"]] = df[["c_V", "c_A", "c_D"]].round(4)

    df.to_csv(go_output_file_path, index=False)



        






if __name__ == "__main__":
    print("Emo_bank transformation\n" + "-"*60)
    transform_emo_bank()
    print("-"*60 + "\nGo_emotions transformation\n" + "-"*60)
    transform_go_emotions()

