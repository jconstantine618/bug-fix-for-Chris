import streamlit as st
import os
import re
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI
from rank_bm25 import BM25Okapi
import pytesseract
from PIL import Image
from git import Repo

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🐞 Bug Navigator", layout="wide")
st.title("🐞 Bug Navigator with GPT-5 Pro")


# ---------------------------
# Utilities
# ---------------------------
def slugify_repo_url(repo_url: str) -> str:
    m = re.search(r'[:/]+([^/]+)/([^/]+?)(?:\.git)?$', repo_url)
    if m:
        return f"{m.group(1)}-{m.group(2)}".lower()
    return re.sub(r'[^a-zA-Z0-9]+', '-', repo_url).strip('-').lower()[:120] or "repo"


def clone_repo(repo_url: str) -> str:
    slug = slugify_repo_url(repo_url)
    dest = tempfile.mkdtemp(prefix=f"{slug}-")
    Repo.clone_from(repo_url, dest)
    return dest


def list_source_files(root_dir: str):
    exts = {
        ".py",".js",".jsx",".ts",".tsx",".mjs",".cjs",".vue",
        ".go",".java",".rb",".php",".cs",".cpp",".c",".h",".hpp",
        ".rs",".kt",".swift",".m",".mm"
    }
    ignores = {".git", "node_modules", "dist", "build", ".next", ".venv", "venv", "__pycache__", ".cache"}
    files = []
    for p in Path(root_dir).rglob("*"):
        if p.is_dir() and p.name in ignores:
            continue
        if p.is_file() and p.suffix.lower() in exts and p.stat().st_size < 2_000_000:
            files.append(str(p))
    return files


def read_file_lines(path: str):
    try:
        return Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return []


# ---------------------------
# Indexing
# ---------------------------
@dataclass
class Chunk:
    file_path: str
    start_line: int
    end_line: int
    text: str
    tokens: List[str]


def tokenize(text: str):
    return [t.lower() for t in re.split(r"[^\w]+", text) if t and not t.isnumeric()]


class RepoIndex:
    def __init__(self, repo_root: str, window_size: int = 50, stride: int = 25):
        self.repo_root = repo_root
        self.window_size = window_size
        self.stride = stride
        self.chunks: List[Chunk] = []
        self._bm25 = None

    def build(self):
        files = list_source_files(self.repo_root)
        doc_tokens = []
        for fp in files:
            rel = os.path.relpath(fp, self.repo_root)
            lines = read_file_lines(fp)
            n = len(lines)
            i = 0
            while i < n:
                start, end = i+1, min(n, i+self.window_size)
                text = "\n".join(lines[i:end])
                tokens = tokenize(text)
                self.chunks.append(Chunk(rel, start, end, text, tokens))
                doc_tokens.append(tokens)
                if end >= n: break
                i += self.stride
        self._bm25 = BM25Okapi(doc_tokens if doc_tokens else [["empty"]])

    def query(self, text: str, k: int = 5):
        q = tokenize(text)
        scores = self._bm25.get_scores(q)
        results = []
        for i, chunk in enumerate(self.chunks):
            results.append({
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "snippet": chunk.text,
                "score": float(scores[i]),
            })
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:k]


# ---------------------------
# OCR
# ---------------------------
def ocr_image(img_file) -> str:
    img = Image.open(img_file)
    return pytesseract.image_to_string(img)


# ---------------------------
# GPT-5 Pro Explanation
# ---------------------------
def explain_with_gpt(results, bug_text=""):
    context = "You are an AI bug triage assistant. Given code search results, explain which files/lines are most likely responsible for the bug."
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Bug report: {bug_text}\n\nResults:\n{results}"}
    ]
    completion = client.chat.completions.create(
        model="gpt-5.0",  # adjust if needed (gpt-4.1 etc.)
        messages=messages,
        temperature=0.3
    )
    return completion.choices[0].message.content


# ---------------------------
# Streamlit UI
# ---------------------------
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.repo_root = None

repo_url = st.text_input("🔗 GitHub Repository URL")

if st.button("Ingest Repository"):
    with st.spinner("Cloning and indexing repo..."):
        repo_root = clone_repo(repo_url)
        idx = RepoIndex(repo_root)
        idx.build()
        st.session_state.index = idx
        st.session_state.repo_root = repo_root
        st.success(f"Indexed {len(idx.chunks)} code chunks from {repo_url}")

st.divider()

mode = st.radio("Provide your bug report", ["📝 Type description", "🖼️ Upload screenshot"])

if mode == "📝 Type description":
    text_input = st.text_area("Paste bug description or stack trace")
    if st.button("Find Bug") and st.session_state.index:
        results = st.session_state.index.query(text_input, k=5)
        st.subheader("Raw Matches")
        for r in results:
            st.markdown(f"**{r['file_path']}** (Lines {r['start_line']}-{r['end_line']})")
            st.code(r["snippet"])
        with st.spinner("Summarizing with ChatGPT-5 Pro..."):
            st.markdown("### 🤖 GPT-5 Pro Explanation")
            st.write(explain_with_gpt(results, bug_text=text_input))

if mode == "🖼️ Upload screenshot":
    uploaded = st.file_uploader("Upload a screenshot", type=["png","jpg","jpeg"])
    if uploaded and st.button("Analyze Screenshot") and st.session_state.index:
        text = ocr_image(uploaded)
        st.text_area("Extracted Text", value=text, height=100)
        results = st.session_state.index.query(text, k=5)
        st.subheader("Raw Matches")
        for r in results:
            st.markdown(f"**{r['file_path']}** (Lines {r['start_line']}-{r['end_line']})")
            st.code(r["snippet"])
        with st.spinner("Summarizing with ChatGPT-5 Pro..."):
            st.markdown("### 🤖 GPT-5 Pro Explanation")
            st.write(explain_with_gpt(results, bug_text=text))
