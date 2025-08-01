import tempfile, os, pdfplumber
from sentence_transformers import SentenceTransformer, util
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_from_pdf(file_bytes):
    temp_path = tempfile.mktemp()
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        with pdfplumber.open(temp_path) as pdf:
            text = "\n".join([page.extract_text() or "" for page in pdf.pages])
    finally:
        os.remove(temp_path)

    return text.strip()

def calculate_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return round(util.cos_sim(emb1, emb2).item() * 100, 2)

def extract_keywords(text):
    doc = nlp(text.lower())
    return set([
        token.lemma_ for token in doc
        if token.pos_ in {"NOUN", "PROPN", "VERB"} and not token.is_stop
    ])
