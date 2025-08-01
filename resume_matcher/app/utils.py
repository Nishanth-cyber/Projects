import io
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_bytes):
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return round(float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]) * 100, 2)

def extract_keywords(text):
    doc = nlp(text.lower())
    return set([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
