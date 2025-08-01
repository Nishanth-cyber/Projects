import tempfile, os, pdfplumber, docx
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
import spacy

nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedder)

def extract_text_from_file(file_bytes, filename):
    temp_path = tempfile.mktemp()
    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    try:
        if filename.endswith(".pdf"):
            with pdfplumber.open(temp_path) as pdf:
                text = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif filename.endswith(".docx"):
            doc = docx.Document(temp_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        else:
            text = file_bytes.decode("utf-8")
    finally:
        os.remove(temp_path)

    return text.strip()

def calculate_similarity(text1, text2):
    emb1 = embedder.encode(text1, convert_to_tensor=True)
    emb2 = embedder.encode(text2, convert_to_tensor=True)
    return round(util.cos_sim(emb1, emb2).item() * 100, 2)

def extract_keywords(text, top_n=15):
    return [kw for kw, _ in kw_model.extract_keywords(text, top_n=top_n, stop_words='english')]
