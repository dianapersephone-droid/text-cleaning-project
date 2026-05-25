import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

texts = [
    "This movie was amazing! I really loved the story.",
    "The food was terrible, and the service was slow.",
    "I enjoyed reading this news article about technology.",
    "The product is good, but the delivery was late.",
    "This book was boring and too long."
]

# Create table
df = pd.DataFrame(texts, columns=["original_text"])

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)
    cleaned_words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(cleaned_words)

df["cleaned_text"] = df["original_text"].apply(clean_text)

df.to_csv("cleaned_text_dataset.csv", index=False)

print("Done! CSV file created.")
