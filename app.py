"""Streamlit UI for GMFM-66 video analysis and scoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import List

import streamlit as st
from openai import OpenAI

from llm_engine import GMFMItem, GMFMScorer
from video_analyzer import MovementAnalyzer


CONFIG_PATH = Path(__file__).with_name("gmfm_config.json")


@st.cache_data(show_spinner=False)
def load_items() -> List[GMFMItem]:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [GMFMItem(**item) for item in data]


def _save_uploaded_video(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> str:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        return tmp.name


@st.cache_data(show_spinner=False)
def load_models(base_url: str) -> List[str]:
    try:
        client = OpenAI(base_url=base_url, api_key="lm-studio")
        models = client.models.list()
        return [model.id for model in models.data]
    except Exception:  # noqa: BLE001
        return []


def main() -> None:
    st.set_page_config(page_title="GMFM-66 Analys", layout="wide")
    st.title("GMFM-66 Videoanalys")

    items = load_items()
    item_map = {item.name: item for item in items}

    with st.sidebar:
        st.header("Inställningar")
        base_url = st.text_input("LLM-bas-URL", value="http://localhost:1234/v1")
        selected_name = st.selectbox(
            "Välj GMFM-objekt",
            options=list(item_map.keys()),
            index=0,
        )
        if st.button("Uppdatera modellista"):
            load_models.clear()
        model_options = load_models(base_url)
        if model_options:
            llm_model = st.selectbox("LLM-modellnamn", options=model_options, index=0)
        else:
            llm_model = st.text_input("LLM-modellnamn", value="local-model")
        frame_stride = st.slider("Bildruta-steg", min_value=1, max_value=5, value=2)

    st.subheader("Ladda upp video")
    uploaded = st.file_uploader("Välj videofil", type=["mp4", "mov", "avi", "mkv"])

    if not uploaded:
        st.info("Ladda upp en video för att starta analysen.")
        return

    video_path = _save_uploaded_video(uploaded)
    st.video(video_path)

    analyzer = MovementAnalyzer(frame_stride=frame_stride)
    scorer = GMFMScorer(model=llm_model, base_url=base_url)

    item = item_map[selected_name]

    with st.spinner("Analyserar biomekanik..."):
        try:
            analysis = analyzer.analyze_video(video_path)
        except ValueError as exc:
            st.error(f"Kunde inte analysera video: {exc}")
            return

    with st.spinner("Konsulterar AI-expert..."):
        try:
            score_result = scorer.evaluate_item(item, analysis)
        except Exception as exc:  # noqa: BLE001
            st.error(f"LLM-fel: {exc}")
            return

    st.subheader("Slutpoäng")
    st.metric(label="Poäng (0-3)", value=score_result.score)

    st.subheader("Motivering")
    st.write(score_result.reasoning)

    st.subheader("Tekniska mätvärden")
    st.json(analysis)


if __name__ == "__main__":
    main()
