import streamlit as st
import requests
import os
from openai import OpenAI
from PIL import Image

API_URL = "http://127.0.0.1:8000"  # Backend URL

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="Bug Navigator", layout="wide")
st.title("🐞 Bug Navigator with ChatGPT-5 Pro")

# --- Repo input ---
repo_url = st.text_input("🔗 GitHub Repository URL", "https://github.com/pallets/flask")

if st.button("Ingest Repository"):
    with st.spinner("Cloning and indexing repo..."):
        resp = requests.post(f"{API_URL}/ingest", json={"repo_url": repo_url})
        if resp.status_code == 200:
            data = resp.json()
            st.success(f"Repo ingested: {data['files_indexed']} files indexed ({data['chunks']} chunks)")
        else:
            st.error(f"Failed: {resp.text}")

st.divider()

# --- Bug input ---
mode = st.radio("Provide your bug report", ["📝 Type description", "🖼️ Upload screenshot"])

def explain_with_gpt(results, bug_text=""):
    """Use GPT-5 Pro to explain ranked results in natural language."""
    context = "You are an AI bug triage assistant. Given code search results, explain which files/lines are most likely responsible for the bug."
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Bug report: {bug_text}\n\nResults:\n{results}"}
    ]
    completion = client.chat.completions.create(
        model="gpt-5.0",  # adjust if Streamlit supports "gpt-4.1" or latest
        messages=messages,
        temperature=0.3
    )
    return completion.choices[0].message.content

if mode == "📝 Type description":
    text_input = st.text_area("Paste bug description or stack trace")
    if st.button("Find Bug"):
        if text_input.strip():
            with st.spinner("Analyzing repo..."):
                resp = requests.post(f"{API_URL}/query", json={"text": text_input, "k": 5})
                if resp.ok:
                    results = resp.json()["results"]
                    st.subheader("Raw Matches")
                    for r in results:
                        st.markdown(f"**{r['file_path']}** (Lines {r['start_line']}-{r['end_line']})")
                        st.code(r["snippet"], language="python")
                        st.caption(f"Reasoning: {r['rationale']}")

                    with st.spinner("Summarizing with ChatGPT-5 Pro..."):
                        explanation = explain_with_gpt(results, bug_text=text_input)
                        st.markdown("### 🤖 GPT-5 Pro Explanation")
                        st.write(explanation)
                else:
                    st.error(resp.text)

if mode == "🖼️ Upload screenshot":
    uploaded = st.file_uploader("Upload a screenshot", type=["png", "jpg", "jpeg"])
    if uploaded and st.button("Analyze Screenshot"):
        with st.spinner("Extracting text and analyzing..."):
            files = {"image": uploaded.getvalue()}
            resp = requests.post(f"{API_URL}/upload_screenshot", files={"image": uploaded})
            if resp.ok:
                data = resp.json()
                st.text_area("Extracted Text", value=data["extracted_text"], height=100)
                results = data["results"]

                st.subheader("Raw Matches")
                for r in results:
                    st.markdown(f"**{r['file_path']}** (Lines {r['start_line']}-{r['end_line']})")
                    st.code(r["snippet"], language="python")
                    st.caption(f"Reasoning: {r['rationale']}")

                with st.spinner("Summarizing with ChatGPT-5 Pro..."):
                    explanation = explain_with_gpt(results, bug_text=data["extracted_text"])
                    st.markdown("### 🤖 GPT-5 Pro Explanation")
                    st.write(explanation)
            else:
                st.error(resp.text)
