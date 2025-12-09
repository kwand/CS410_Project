from __future__ import annotations

import json
import requests
from dataclasses import dataclass
from typing import Dict, Protocol

import click

class SentimentAnalyzer(Protocol):
    backend_name: str

    def analyze(self, text: str) -> Dict:
        ...


@dataclass
class VaderAnalyzer:
    backend_name: str = "vader"

    def __post_init__(self):
        try:
            from nltk.sentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError("nltk is required for VADER analysis") from exc
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> Dict:
        scores = self.analyzer.polarity_scores(text or "")
        label = "neutral"
        if scores["compound"] >= 0.05:
            label = "positive"
        elif scores["compound"] <= -0.05:
            label = "negative"
        normalized = (scores["compound"] + 1) / 2  # map [-1,1] to [0,1]
        return {
            "backend": self.backend_name,
            "label": label,
            "score": scores["compound"],
            "normalized_score": normalized,
            "raw": scores,
        }


@dataclass
class TransformersAnalyzer:
    backend_name: str = "transformers"
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer_kwargs: dict[str, object] | None = None

    def __post_init__(self):
        try:
            from transformers import AutoTokenizer, pipeline
        except ImportError as exc:
            raise ImportError("transformers is required for HF sentiment analysis") from exc

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            use_fast=False, 
            max_length=512,
            truncation=True,
        )
        self.pipeline = pipeline(
            "sentiment-analysis",
            model=self.model_name,
            tokenizer=tokenizer,
            max_length=512,
            truncation=True,
        )

    def analyze(self, text: str) -> Dict:
        result = self.pipeline(text or "")[0]
        hf_label = result.get("label", "").lower()
        prob = float(result.get("score", 0.5))

        if "neg" in hf_label:
            label = "negative"
            normalized = 0.5 - (0.5 * prob)
        elif "pos" in hf_label:
            label = "positive"
            normalized = 0.5 + (0.5 * prob)
        else:
            label = "neutral"
            normalized = 0.5

        return {
            "backend": self.backend_name,
            "label": label,
            "score": prob,
            "normalized_score": normalized,
            "raw": result,
        }


@dataclass
class InContextLearningOllamaAnalyzer:
    backend_name: str = "icl_ollama"
    # model: str = "qwen3:4b"
    model: str = "qwen3:0.6b"
    endpoint: str = "http://localhost:11434/api/generate"
    prompt: str = (
        "You are a sentiment classifier of YouTube comments from a music competition. "
        "Only classify comments that seem to be relevant to the performer or performance. "
        "For example, if the comment is mostly directed to the competition itself or jury, output \"irrelevant\"."
        "Do not think.\n\n"
        "Output valid JSON with keys: label (positive/negative/neutral/irrelevant), confidence (0–1), rationale (short).\n\n"
        "TEXT:"
    )
    timeout: int = 60

    def analyze(self, text: str) -> Dict:
        prompt = self._build_prompt(text or "")
        payload = {"model": self.model, "prompt": prompt, "stream": False}

        response_json = self._call_ollama(payload)

        raw_content = response_json.get("response", "")
        try:
            parsed = json.loads(raw_content) if raw_content else {}
        except json.JSONDecodeError:
            click.echo(f"Error parsing Ollama response: {raw_content}. Using fallback result...", err=True)
            return self._fallback_result("parse_failed", text, raw_response=response_json)

        label = str(parsed.get("label", "")).strip().lower() or "neutral"
        confidence = self._clamp_confidence(parsed.get("confidence", 0.5))
        normalized = self._normalize_score(label, confidence)

        return {
            "backend": self.backend_name,
            "label": label,
            "score": confidence,
            "normalized_score": normalized,
            "rationale": parsed.get("rationale", ""),
        }

    def _build_prompt(self, text: str) -> str:
        stripped = text.strip()
        return f"{self.prompt}\n\n{stripped}"

    def _call_ollama(self, payload: Dict) -> Dict:
        try:
            response = requests.post(self.endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            click.echo(f"Error calling Ollama: {exc}. Please ensure Ollama server is running.", err=True)
            raise 

    def _clamp_confidence(self, value) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, numeric))

    def _normalize_score(self, label: str, confidence: float) -> float | None:
        if label == "positive":
            return 0.5 + (0.5 * confidence)
        if label == "negative":
            return 0.5 - (0.5 * confidence)
        if label == "neutral":
            return 0.5
        if label == "irrelevant":
            return None
        return 0.5

    def _fallback_result(self, reason: str, text: str, error: str | None = None, raw_response: Dict | None = None) -> Dict:
        return {
            "backend": self.backend_name,
            "label": "neutral",
            "score": None,
            "normalized_score": None,
            "rationale": "",
            "raw": {
                "endpoint": self.endpoint,
                "error": error or reason,
                "response": raw_response,
                "text": text,
            },
        }


def create_analyzer(name: str) -> SentimentAnalyzer:
    normalized = name.lower()
    if normalized == "vader":
        return VaderAnalyzer()
    if normalized in {"hf", "transformers"}:
        return TransformersAnalyzer()
    if normalized in {"icl_ollama", "ollama", "icl"}:
        return InContextLearningOllamaAnalyzer()
    raise ValueError(f"Unknown analyzer {name!r}. Options: vader, transformers, icl_ollama.")
