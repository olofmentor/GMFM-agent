"""LLM reasoning engine for GMFM-66 scoring using a local OpenAI-compatible API."""

from __future__ import annotations

import json
import re
from typing import Dict, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field


class GMFMItem(BaseModel):
    """Configuration for a GMFM-66 item."""

    id: str
    name: str
    instruction: str
    criteria_0_to_3: Dict[str, str] = Field(default_factory=dict)
    required_checks: list[str] = Field(default_factory=list)


class GMFMScoreResult(BaseModel):
    """Structured result from the LLM scoring."""

    score: int = Field(ge=0, le=3)
    reasoning: str


class GMFMScorer:
    """Scores GMFM items based on CV data and clinical reasoning."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        model: str = "local-model",
        temperature: float = 0.2,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_prompt(self, item: GMFMItem, video_data: Dict[str, object]) -> str:
        criteria_lines = "\n".join(
            f"{k}: {v}" for k, v in sorted(item.criteria_0_to_3.items())
        )
        video_json = json.dumps(video_data, indent=2, ensure_ascii=False)
        return (
            "You are a pediatric physiotherapy expert scoring GMFM-66.\n"
            f"Item: {item.name}\n"
            f"Instruction: {item.instruction}\n"
            "Scoring criteria (0-3):\n"
            f"{criteria_lines}\n\n"
            "Computer vision measurements (hard facts):\n"
            f"{video_json}\n\n"
            "Rules:\n"
            "- If criteria for 3 is met physically, assume 3 unless quality is poor.\n"
            "- Use soft factors: stability, smoothness, symmetry, compensations.\n"
            "- Output a numeric score 0-3 and a concise reasoning.\n"
            "Return format:\n"
            "Score: <0-3>\n"
            "Reasoning: <one short paragraph>\n"
        )

    @staticmethod
    def _parse_score(text: str) -> int:
        match = re.search(r"Score:\s*([0-3])", text)
        if match:
            return int(match.group(1))
        fallback = re.search(r"\b([0-3])\b", text)
        if fallback:
            return int(fallback.group(1))
        return 0

    def evaluate_item(self, item: GMFMItem, video_analysis_data: Dict[str, object]) -> GMFMScoreResult:
        """Call the local LLM and return a score with reasoning."""
        prompt = self._build_prompt(item, video_analysis_data)
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": "You are a careful clinical assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        score = self._parse_score(content)
        reasoning = content.strip() or "No reasoning returned by model."
        return GMFMScoreResult(score=score, reasoning=reasoning)
