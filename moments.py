import io
import zipfile
import logging
import time
from typing import List, Dict, Any

import requests
import pandas as pd
import json
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# -----------------------------
# Streamlit Sidebar Configuration
# -----------------------------
st.sidebar.header("🔧 Configuration")

# API Keys
OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")
ALSOASKED_API_KEY = st.sidebar.text_input("AlsoAsked API Key", type="password")

# Model selection
SBERT_MODEL = st.sidebar.text_input("SBERT Model", value="all-MiniLM-L6-v2")
OPENAI_MODEL = st.sidebar.text_input("OpenAI Model", value="gpt-3.5-turbo")

# Pipeline parameters
TOP_X = st.sidebar.number_input("Top X results", min_value=1, value=50)
THRESHOLD = st.sidebar.slider("Similarity Threshold", min_value=0.0, max_value=1.0, value=0.4, step=0.01)

# Seed terms input
seeds_input = st.sidebar.text_area("Enter seed terms, one per line:")

# Logging level
LOG_LEVEL = st.sidebar.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)

# Validate credentials and seeds
if not OPENAI_API_KEY or not ALSOASKED_API_KEY:
    st.error("Both OpenAI and AlsoAsked API keys are required.")
    st.stop()
if not seeds_input.strip():
    st.error("Please enter at least one seed term.")
    st.stop()

# Parse seeds
seeds = [line.strip() for line in seeds_input.splitlines() if line.strip()]

# -----------------------------
# Helper Functions and Classes
# -----------------------------
def setup_logging(level: str = LOG_LEVEL) -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=getattr(logging, level.upper(), logging.INFO)
    )

class AlsoAskedClient:
    def __init__(self, api_key: str, base_url: str = "https://alsoaskedapi.com/v1/search"):
        self.url = base_url
        self.headers = {"Content-Type": "application/json", "X-Api-Key": api_key}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_questions(self, seed_query: str, limit: int = TOP_X, depth: int = 2, region: str = "gb", language: str = "en") -> List[str]:
        payload = {"terms": [seed_query], "language": language, "region": region, "depth": depth, "fresh": True, "async": False, "notify_webhooks": False}
        def flatten(qs: Any) -> List[str]:
            flat: List[str] = []
            if not isinstance(qs, list):
                return flat
            for q in qs:
                if not isinstance(q, dict): continue
                text = q.get("question") or q.get("query")
                if text: flat.append(text)
                nested = q.get("results") or []
                flat.extend(flatten(nested))
            return flat
        for attempt in range(3):
            try:
                logging.info(f"Fetching PAA for '{seed_query}', attempt {attempt+1}")
                resp = self.session.post(self.url, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json() or {}
                queries = data.get("queries") or []
                if not queries or not isinstance(queries, list):
                    logging.warning(f"No 'queries' list returned for '{seed_query}'")
                    return []
                results = (queries[0] or {}).get("results") or []
                return flatten(results)[:limit]
            except Exception as e:
                logging.warning(f"AlsoAsked attempt {attempt+1} failed: {e}")
                time.sleep(3)
        logging.error(f"All AlsoAsked attempts failed for '{seed_query}'")
        return []

class SBERTRelevance:
    def __init__(self, model_name: str = SBERT_MODEL):
        logging.info(f"Loading SBERT model '{model_name}'")
        self.model = SentenceTransformer(model_name)

    def score(self, seed: str, questions: List[str]) -> List[float]:
        embeddings = self.model.encode([seed] + questions, convert_to_tensor=True)
        seed_emb, question_embs = embeddings[0], embeddings[1:]
        return util.cos_sim(seed_emb, question_embs)[0].tolist()

class OpenAIClassifier:
    def __init__(self, client: OpenAI, model: str = OPENAI_MODEL):
        self.client = client
        self.model = model

        def group_by_moment(self, seed: str, questions: List[str]) -> Dict[str, List[str]]:
        # Build a unified prompt with all questions
        questions_block = "
".join(f"- {q}" for q in questions)
        prompt = f"""
You are a customer journey specialist organizing questions about '{seed}'. Group them into user-centric moments—stages reflecting what a person is thinking or trying to achieve. Use descriptive stage names and list the associated questions. Return ONLY a JSON object with moment names as keys and arrays of questions as values.

Questions:
{questions_block}
"""
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an assistant grouping questions into user journey stages."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                logging.warning(f"OpenAI attempt {attempt+1} failed: {e}")
                time.sleep(3)
        logging.error(f"All OpenAI attempts failed for '{seed}'")
        return {}

# Main Streamlit App
st.title("🔍 PAA & Clustering Pipeline")
if st.sidebar.button("Run Pipeline"):
    setup_logging(LOG_LEVEL)
    st.info("Starting pipeline...")

    also_client = AlsoAskedClient(api_key=ALSOASKED_API_KEY)
    sbert = SBERTRelevance()
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    classifier = OpenAIClassifier(client=openai_client)

    # In-memory zip buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for seed in seeds:
            st.write(f"Processing seed: {seed}")
            questions = also_client.get_questions(seed)
            if not questions:
                st.warning(f"No questions retrieved for '{seed}'. Skipping.")
                continue
            scores = sbert.score(seed, questions)
            filtered = [q for q, s in zip(questions, scores) if s >= THRESHOLD]
            st.write(f" - {len(filtered)}/{len(questions)} passed threshold")
            groups = classifier.group_by_moment(seed, filtered) if filtered else {}

            # Create CSV strings
            q_df = pd.DataFrame({"seed": seed, "question": questions, "similarity": scores})
            m_rows = []
            for moment, qs in groups.items():
                m_rows.append({"seed": seed, "moment": moment, "questions": "|".join(qs)})
            m_df = pd.DataFrame(m_rows)

            # Write CSVs into zip
            zf.writestr(f"{seed.replace(' ', '_')}_questions.csv", q_df.to_csv(index=False))
            zf.writestr(f"{seed.replace(' ', '_')}_moments.csv", m_df.to_csv(index=False))

    zip_buffer.seek(0)
    st.download_button(
        label="📥 Download All Outputs",
        data=zip_buffer.getvalue(),
        file_name="outputs.zip",
        mime="application/zip"
    )
    st.success("Pipeline complete!")
