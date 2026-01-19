# GMFM-66 Video Analysis Agent

## Getting started

1. Create and activate a virtual environment:

```
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

3. Start your local LLM server in LM Studio with an OpenAI-compatible API:

- Base URL: `http://localhost:1234/v1`
- Model name: update in the app sidebar if needed

4. Run the Streamlit app:

```
streamlit run app.py
```

5. Open the provided URL (usually `http://localhost:8501`), select a GMFM item, and upload a video.
