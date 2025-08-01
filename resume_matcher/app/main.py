from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from .utils import extract_text_from_file, calculate_similarity, extract_keywords
import spacy
import subprocess

# Lazy-load spaCy model (in case it’s not pre-downloaded)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

app = FastAPI(title="Resume Matcher API")

# Enable CORS for frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple health check (Render uses this sometimes)
@app.get("/")
def read_root():
    return {"status": "ok"}

# Main endpoint
@app.post("/match-resume")
async def match_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        resume_bytes = await resume.read()
        resume_text = extract_text_from_file(resume_bytes, resume.filename)

        if not resume_text.strip():
            return JSONResponse(status_code=400, content={"error": "Resume has no readable text."})

        score = calculate_similarity(resume_text, job_description)
        resume_keywords = set(extract_keywords(resume_text))
        jd_keywords = set(extract_keywords(job_description))

        matched = sorted(jd_keywords & resume_keywords)
        missing = sorted(jd_keywords - resume_keywords)

        return {
            "score": score,
            "matched_keywords": matched,
            "missing_keywords": missing
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
