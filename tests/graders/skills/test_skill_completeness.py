#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for :class:`openjudge.graders.skills.completeness.SkillCompletenessGrader`.

Includes:

1. **Unit** (offline, mocked LLM): initialization, threshold validation, happy path, errors.
2. **Quality** (optional API): benchmark JSON aligned with
   ``agentscope-ai/OpenJudge`` evaluation format; accuracy and consistency checks.

Benchmark file layout (for HuggingFace upload)::

    skills/skill_completeness/skill_completeness_eval_v1.json

Local copy::

    tests/graders/skills/skill_completeness_eval_v1.json

Run unit tests::

    pytest tests/graders/skills/test_skill_completeness.py -m unit -v

Run quality tests (requires ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` in the
environment or in the repo root ``.env`` — loaded automatically)::

    pytest tests/graders/skills/test_skill_completeness.py -m quality -v
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from dotenv import load_dotenv

try:
    from openai import RateLimitError as _OpenAIRateLimitError
except ImportError:
    _OpenAIRateLimitError = None  # type: ignore[assignment,misc]

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer
from openjudge.graders.skills.completeness import SkillCompletenessGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
# ``.env`` lives at the repository root (same level as ``pyproject.toml``).

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[2]  # skills/graders/tests -> OpenJudge root
DOTENV_PATH = _REPO_ROOT / ".env"
DATA_FILE = _TESTS_DIR / "skill_completeness_eval_v1.json"

load_dotenv(DOTENV_PATH)

# Quality tests: same gate as other grader suites
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
RUN_QUALITY_TESTS = bool(OPENAI_API_KEY and OPENAI_BASE_URL)


def _load_hf_json(path: Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def hf_records_to_eval_samples(records: List[dict]) -> List[Dict[str, Any]]:
    """Flatten HuggingFace-style rows into grader inputs plus ``expected_score`` label."""
    samples: List[Dict[str, Any]] = []
    for item in records:
        meta_in = item["input"]["metadata"]
        exp = item["metadata"]["expected_score"]
        samples.append(
            {
                "task_description": item["input"].get("query") or "",
                "skill_name": meta_in["skill_name"],
                "skill_manifest": meta_in["skill_manifest"],
                "instruction_body": meta_in.get("instruction_body", ""),
                "script_contents": meta_in.get("script_contents") or [],
                "reference_contents": meta_in.get("reference_contents") or [],
                "expected_score": int(exp),
            }
        )
    return samples


def _completeness_mapper(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Strip label fields before calling :meth:`SkillCompletenessGrader.aevaluate`."""
    return {
        "task_description": sample.get("task_description") or None,
        "skill_name": sample["skill_name"],
        "skill_manifest": sample["skill_manifest"],
        "instruction_body": sample["instruction_body"],
        "script_contents": sample.get("script_contents") or [],
        "reference_contents": sample.get("reference_contents") or [],
    }


# ==================== UNIT TESTS ====================


@pytest.mark.unit
class TestSkillCompletenessGraderUnit:
    """Offline tests with a mocked chat model."""

    def test_initialization(self) -> None:
        mock_model = AsyncMock()
        grader = SkillCompletenessGrader(model=mock_model, threshold=2)
        assert grader.name == "skill_completeness"
        assert grader.threshold == 2

    def test_invalid_threshold_raises(self) -> None:
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillCompletenessGrader(model=mock_model, threshold=4)

    @pytest.mark.asyncio
    async def test_successful_evaluation(self) -> None:
        mock_response = AsyncMock()
        mock_response.parsed = {"score": 3, "reason": "Clear steps, inputs, outputs, and edge cases."}

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model, threshold=2)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                task_description="Summarize a document.",
                skill_name="doc-sum",
                skill_manifest="name: doc-sum\ndescription: Summarizes documents.",
                instruction_body="# Doc\n## Steps\n1. Load\n2. Summarize\n",
                script_contents=[],
                reference_contents=[],
            )

        assert result.score == 3
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == 2

    @pytest.mark.asyncio
    async def test_evaluation_error_returns_grader_error(self) -> None:
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = RuntimeError("API unavailable")

            mock_model = AsyncMock()
            grader = SkillCompletenessGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="x",
                skill_manifest="name: x\ndescription: y",
                instruction_body="body",
                script_contents=[],
                reference_contents=[],
            )

        assert "Evaluation error" in result.error


# ==================== QUALITY TESTS ====================


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillCompletenessGraderQuality:
    """Live LLM tests against the curated JSON benchmark."""

    @pytest.fixture
    def dataset(self) -> List[Dict[str, Any]]:
        if not DATA_FILE.exists():
            pytest.skip(f"Benchmark file not found: {DATA_FILE}")
        raw = _load_hf_json(DATA_FILE)
        return hf_records_to_eval_samples(raw)

    @pytest.fixture
    def model(self) -> OpenAIChatModel:
        config: Dict[str, Any] = {"model": os.getenv("OPENAI_MODEL", "qwen-max"), "api_key": OPENAI_API_KEY}
        if OPENAI_BASE_URL:
            config["base_url"] = OPENAI_BASE_URL
        return OpenAIChatModel(**config)

    @pytest.mark.asyncio
    async def test_runner_batch_scores_in_range(self, dataset: List[Dict[str, Any]], model: OpenAIChatModel) -> None:
        grader = SkillCompletenessGrader(model=model, threshold=2)
        grader_configs = {
            "skill_completeness": GraderConfig(
                grader=grader,
                mapper=_completeness_mapper,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        results = await runner.arun(dataset)

        assert len(results["skill_completeness"]) == len(dataset)
        for r in results["skill_completeness"]:
            assert 1 <= r.score <= 3
            assert len(r.reason) > 0

    @pytest.mark.asyncio
    async def test_accuracy_vs_expected(self, dataset: List[Dict[str, Any]], model: OpenAIChatModel) -> None:
        grader = SkillCompletenessGrader(model=model, threshold=2)
        grader_configs = {
            "skill_completeness": GraderConfig(
                grader=grader,
                mapper=_completeness_mapper,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        results = await runner.arun(dataset)

        analyzer = AccuracyAnalyzer()
        acc = analyzer.analyze(
            dataset=dataset,
            grader_results=results["skill_completeness"],
            label_path="expected_score",
        )

        # Subjective rubric: allow moderate disagreement vs fixed labels
        assert acc.accuracy >= 0.5, f"Accuracy below threshold: {acc.accuracy}"
        assert acc.name == "Accuracy Analysis"
        assert "explanation" in acc.metadata

    @pytest.mark.asyncio
    async def test_consistency_two_runs(self, dataset: List[Dict[str, Any]], model: OpenAIChatModel) -> None:
        grader = SkillCompletenessGrader(model=model, threshold=2)
        grader_configs = {
            "run_a": GraderConfig(grader=grader, mapper=_completeness_mapper),
            "run_b": GraderConfig(grader=grader, mapper=_completeness_mapper),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        try:
            results = await runner.arun(dataset)
        except Exception as exc:
            if _OpenAIRateLimitError and isinstance(exc, _OpenAIRateLimitError):
                pytest.skip(f"Skipped: API quota exceeded ({exc})")
            raise

        consistency = ConsistencyAnalyzer().analyze(
            dataset=dataset,
            grader_results=results["run_a"],
            another_grader_results=results["run_b"],
        )
        assert math.isnan(consistency.consistency) or consistency.consistency >= 0.85


@pytest.mark.unit
def test_hf_fixture_loads() -> None:
    """Sanity check: JSON is valid and matches the loader (no API)."""
    if not DATA_FILE.exists():
        pytest.skip(f"Missing {DATA_FILE}")
    raw = _load_hf_json(DATA_FILE)
    samples = hf_records_to_eval_samples(raw)
    assert len(samples) >= 1
    assert all(1 <= s["expected_score"] <= 3 for s in samples)
