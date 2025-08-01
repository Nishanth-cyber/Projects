from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from utils import extract_text_from_pdf, calculate_similarity, extract_keywords

app = FastAPI(title="Resume Matcher API (PDF Only)")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Resume Matcher API is running"}

@app.post("/match-resume")
async def match_resume(resume: UploadFile = File(...), job_description: str = Form(...)):
    try:
        if not resume.filename.endswith(".pdf"):
            return JSONResponse(status_code=400, content={"error": "Only PDF files are supported."})

        resume_bytes = await resume.read()
        resume_text = extract_text_from_pdf(resume_bytes)

        if not resume_text.strip():
            return JSONResponse(status_code=400, content={"error": "No readable text in resume PDF."})

        score = calculate_similarity(resume_text, job_description)

        resume_keywords = extract_keywords(resume_text)
        jd_keywords = extract_keywords(job_description)

        missing_keywords = sorted(jd_keywords - resume_keywords)

        return {
            "score": score,
            "missing_keywords": missing_keywords
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
