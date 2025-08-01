import io
import spacy
import pdfplumber
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT()

def extract_text_from_pdf(file_bytes):
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def extract_keywords(text):
    doc = nlp(text.lower())
    words = {token.lemma_ for token in doc if token.is_alpha and not token.is_stop}
    return words

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
    tfidf = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])
    return float(round(sim[0][0] * 100, 2))
