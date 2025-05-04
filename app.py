import os
import json
import base64
import re
import logging
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import docx
import mammoth
import altair as alt
import google.generativeai as genai

# ————— Logging Configuration —————
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ————— Load Environment Variables —————
load_dotenv()
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    st.error("Missing GENAI_API_KEY. Please set it in your .env file.")
    st.stop()
genai.configure(api_key=GENAI_API_KEY)

# ————— Streamlit Page Config —————
st.set_page_config(
    page_title="📄 Advanced ATS Resume Scanner",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ————— CV Parsing —————
@st.cache_data(show_spinner=False)
def parse_cv(file) -> str:
    """Extract text from PDF, DOCX, or TXT file."""
    ext = Path(file.name).suffix.lower()
    text = ""
    if ext == ".pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == ".docx":
        document = docx.Document(file)
        text = "\n".join(p.text for p in document.paragraphs)
    else:
        text = file.read().decode("utf-8", errors="ignore")
    return text

# ————— Regex-based Contact Extraction —————
def extract_contact_info(text: str) -> Dict[str, str]:
    """Extract name, email, and phone using regex heuristics."""
    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}", text)
    phone_match = re.search(r"\+?\d[\d\s\-]{7,}\d", text)
    name_line = text.strip().split("\n")[0]
    return {
        "name": name_line.strip(),
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
    }

# ————— Enhanced LLM-based Section Extraction —————
@st.cache_data(show_spinner=False)
def extract_sections_with_llm(cv_text: str) -> Dict[str, Any]:
    """Use LLM to extract structured CV sections as JSON, with robust parsing."""
    prompt = (
        "Please extract the following CV sections strictly in valid JSON and wrap the JSON in markdown ```json blocks. "
        "Keys: Personal Information, Summary, Work Experience, Education, Skills, Projects, Certifications, Languages. "
        "Return only markdown-wrapped JSON.\n\nCV Text:\n" + cv_text
    )
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    raw = response.text
    match = re.search(r"```json([\s\S]*?)```", raw)
    json_str = match.group(1).strip() if match else raw
    if not match:
        brace_match = re.search(r"(\{[\s\S]*\})", raw)
        json_str = brace_match.group(1) if brace_match else raw
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {}

# ————— Semantic ATS Scoring via LLM —————
@st.cache_data(show_spinner=False)
def compute_semantic_score(cv_text: str, job_desc: str) -> float:
    """Use LLM to rate CV against job description from 0 to 100."""
    prompt = (
        "On a scale from 0 to 100, rate how well the following CV matches the job description. "
        "Answer only with the numeric score (no text).\n\n"
        f"Job Description:\n{job_desc}\n\nCV:\n{cv_text}"
    )
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    try:
        return float(re.search(r"\d+\.?\d*", response.text).group())
    except Exception:
        logger.warning("Failed to parse semantic score; defaulting to 0")
        return 0.0

# ————— Application Interface —————
def main():
    st.title("🚀 Advanced ATS Resume Scanner")
    st.write("Upload a CV and paste the job description to get instant insights, semantic match scores, and structured extraction.")

    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload CV (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    with col2:
        job_description = st.text_area("Paste Job Description", height=200)

    if st.button("🔍 Analyze Resume"):
        if not uploaded_file or not job_description.strip():
            st.warning("Please upload a CV and provide the job description.")
            return

        cv_text = parse_cv(uploaded_file)
        contact_info = extract_contact_info(cv_text)
        sections = extract_sections_with_llm(cv_text)
        semantic_score = compute_semantic_score(cv_text, job_description)

        # Display Scores
        st.subheader("📈 ATS Match Scores")
        col_score, col_sem = st.columns(2)
        keyword_score = round(len(set(job_description.lower().split()) & set(cv_text.lower().split()))/len(job_description.split())*100,2)
        col_score.metric("Keyword Rule-Based Score", f"{keyword_score}%")
        col_sem.metric("Semantic LLM Score", f"{semantic_score:.2f}%")
        st.progress(min(max(semantic_score/100, 0.0), 1.0))

        # Contact Info
        st.subheader("👤 Contact Information")
        st.markdown(
            f"- **Name:** {contact_info.get('name','N/A')}  \n"
            f"- **Email:** {contact_info.get('email','N/A')}  \n"
            f"- **Phone:** {contact_info.get('phone','N/A')}"
        )

        # Extracted Sections
        st.subheader("🗂️ Extracted Sections")
        if sections:
            for section, content in sections.items():
                with st.expander(section, expanded=True):
                    if isinstance(content, list) and all(isinstance(item, dict) for item in content):
                        for item in content:
                            with st.container():
                                c1, c2 = st.columns([2, 5])
                                # Left: job/education header
                                with c1:
                                    st.markdown(f"#### {item.get('Title','')}  ")
                                    st.markdown(f"**{item.get('Company','')}**  ")
                                    st.markdown(f"*{item.get('Start Date','')} – {item.get('End Date','')}*  ")
                                # Right: description bullets
                                with c2:
                                    for line in item.get('Description','').split('\n'):
                                        if line.strip():
                                            st.markdown(f"- {line.strip()}")
                            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
                    elif isinstance(content, list):
                        for item in content:
                            st.markdown(f"- {item}")
                    elif isinstance(content, dict):
                        for key, val in content.items():
                            st.markdown(f"**{key}:** {val}")
                    else:
                        st.write(content)
        else:
            st.info("No structured sections extracted. Try a different CV or format.")

        # Skills Chart
        if isinstance(sections.get("Skills"), list) and sections.get("Skills"):
            counts = {skill: sections["Skills"].count(skill) for skill in sections["Skills"]}
            df = list(counts.items())
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("0:Q", title="Count"),
                y=alt.Y("1:N", sort="-x", title="Skill")
            )
            st.altair_chart(chart, use_container_width=True)

       

if __name__ == "__main__":
    main()
