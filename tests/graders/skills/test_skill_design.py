#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for :class:`openjudge.graders.skills.design.SkillDesignGrader`.

Includes:

1. **Unit** (offline, mocked LLM): initialization, threshold validation, happy path, errors.
2. **Quality** (optional API): benchmark JSON aligned with
   ``agentscope-ai/OpenJudge`` evaluation format; accuracy and consistency checks.

Benchmark file layout (for HuggingFace upload)::

    skills/skill_design/skill_design_eval_v1.json

Local copy::

    tests/graders/skills/skill_design_eval_v1.json

Run unit tests::

    pytest tests/graders/skills/test_skill_design.py -m unit -v

Run quality tests (requires ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` in the
environment or in the repo root ``.env`` — loaded automatically)::

    pytest tests/graders/skills/test_skill_design.py -m quality -v
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

from openjudge.analyzer.statistical import ConsistencyAnalyzer
from openjudge.analyzer.validation import AccuracyAnalyzer
from openjudge.graders.skills.design import SkillDesignGrader
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.runner.grading_runner import GraderConfig, GradingRunner

# ── Paths ─────────────────────────────────────────────────────────────────────
_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parents[2]
DOTENV_PATH = _REPO_ROOT / ".env"
DATA_FILE = _TESTS_DIR / "skill_design_eval_v1.json"

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
                "skill_name": meta_in["skill_name"],
                "skill_manifest": meta_in["skill_manifest"],
                "instruction_body": meta_in.get("instruction_body", ""),
                "script_contents": meta_in.get("script_contents") or [],
                "reference_contents": meta_in.get("reference_contents") or [],
                "expected_score": int(exp),
            }
        )
    return samples


def _design_mapper(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Strip label fields before calling :meth:`SkillDesignGrader.aevaluate`."""
    return {
        "skill_name": sample["skill_name"],
        "skill_manifest": sample["skill_manifest"],
        "instruction_body": sample["instruction_body"],
        "script_contents": sample.get("script_contents") or [],
        "reference_contents": sample.get("reference_contents") or [],
    }


# ==================== UNIT TESTS ====================


@pytest.mark.unit
class TestSkillDesignGraderUnit:
    """Offline tests with a mocked chat model."""

    def test_initialization(self) -> None:
        mock_model = AsyncMock()
        grader = SkillDesignGrader(model=mock_model, threshold=3)
        assert grader.name == "skill_design"
        assert grader.threshold == 3

    def test_invalid_threshold_raises(self) -> None:
        mock_model = AsyncMock()
        with pytest.raises(ValueError, match="threshold must be in range"):
            SkillDesignGrader(model=mock_model, threshold=6)

    @pytest.mark.asyncio
    async def test_successful_evaluation_excellent(self) -> None:
        """Test successful evaluation when skill is excellent (score 5)."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 5,
            "reason": "Excellent skill design with pure knowledge delta, expert thinking frameworks, comprehensive description, proper progressive disclosure, and practical usability.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillDesignGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="excellent-skill",
                skill_manifest="name: excellent-skill\ndescription: A well-designed skill with clear triggers and expert knowledge.",
                instruction_body="# Excellent Skill\n## NEVER\n- NEVER do X because...\n\nClear expert knowledge and decision trees.",
                script_contents=[],
                reference_contents=[],
            )

        assert result.score == 5
        assert "threshold" in result.metadata
        assert result.metadata["threshold"] == 3

    @pytest.mark.asyncio
    async def test_successful_evaluation_poor(self) -> None:
        """Test successful evaluation when skill is poorly designed (score 1)."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 1,
            "reason": "Poor skill design with redundant content explaining basics Claude already knows, vague description without WHEN triggers, and no actionable guidance.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillDesignGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="poor-skill",
                skill_manifest="name: poor-skill\ndescription: A helpful skill for various tasks.",
                instruction_body="# Poor Skill\n\nThis skill helps you do things. Be careful with errors.",
                script_contents=[],
                reference_contents=[],
            )

        assert result.score == 1

    @pytest.mark.asyncio
    async def test_successful_evaluation_adequate(self) -> None:
        """Test successful evaluation when skill is adequate (score 3)."""
        mock_response = AsyncMock()
        mock_response.parsed = {
            "score": 3,
            "reason": "Adequate skill design with some expert knowledge but mixed with redundant content. Description covers WHAT but WHEN triggers could be stronger.",
        }

        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.return_value = mock_response
            mock_model = AsyncMock()
            grader = SkillDesignGrader(model=mock_model, threshold=3)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="adequate-skill",
                skill_manifest="name: adequate-skill\ndescription: Does something useful with files and data.",
                instruction_body="# Adequate Skill\n\nSteps to follow:\n1. Load data\n2. Process\n3. Save results",
                script_contents=[],
                reference_contents=[],
            )

        assert result.score == 3

    @pytest.mark.asyncio
    async def test_evaluation_error_returns_grader_error(self) -> None:
        with patch("openjudge.graders.llm_grader.BaseChatModel.achat", new_callable=AsyncMock) as mock_achat:
            mock_achat.side_effect = RuntimeError("API unavailable")

            mock_model = AsyncMock()
            grader = SkillDesignGrader(model=mock_model)
            grader.model.achat = mock_achat

            result = await grader.aevaluate(
                skill_name="test-skill",
                skill_manifest="name: test-skill\ndescription: A test skill.",
                instruction_body="# Test",
                script_contents=[],
                reference_contents=[],
            )

        assert "Evaluation error" in result.error


# ==================== QUALITY TESTS ====================


@pytest.mark.skipif(not RUN_QUALITY_TESTS, reason="Requires OPENAI_API_KEY and OPENAI_BASE_URL")
@pytest.mark.quality
class TestSkillDesignGraderQuality:
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
        grader = SkillDesignGrader(model=model, threshold=3)
        grader_configs = {
            "skill_design": GraderConfig(
                grader=grader,
                mapper=_design_mapper,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        results = await runner.arun(dataset)

        assert len(results["skill_design"]) == len(dataset)
        for r in results["skill_design"]:
            assert 1 <= r.score <= 5
            assert len(r.reason) > 0

    @pytest.mark.asyncio
    async def test_accuracy_vs_expected(self, dataset: List[Dict[str, Any]], model: OpenAIChatModel) -> None:
        grader = SkillDesignGrader(model=model, threshold=3)
        grader_configs = {
            "skill_design": GraderConfig(
                grader=grader,
                mapper=_design_mapper,
            ),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        results = await runner.arun(dataset)

        analyzer = AccuracyAnalyzer()
        acc = analyzer.analyze(
            dataset=dataset,
            grader_results=results["skill_design"],
            label_path="expected_score",
        )

        # Design evaluation is subjective: allow moderate disagreement vs fixed labels
        assert acc.accuracy >= 0.6, f"Accuracy below threshold: {acc.accuracy}"
        assert acc.name == "Accuracy Analysis"
        assert "explanation" in acc.metadata

    @pytest.mark.asyncio
    async def test_consistency_two_runs(self, dataset: List[Dict[str, Any]], model: OpenAIChatModel) -> None:
        grader = SkillDesignGrader(model=model, threshold=3)
        grader_configs = {
            "run_a": GraderConfig(grader=grader, mapper=_design_mapper),
            "run_b": GraderConfig(grader=grader, mapper=_design_mapper),
        }
        runner = GradingRunner(grader_configs=grader_configs, max_concurrency=4)
        results = await runner.arun(dataset)

        consistency = ConsistencyAnalyzer().analyze(
            dataset=dataset,
            grader_results=results["run_a"],
            another_grader_results=results["run_b"],
        )
        assert math.isnan(consistency.consistency) or consistency.consistency >= 0.70


@pytest.mark.unit
def test_hf_fixture_loads() -> None:
    """Sanity check: JSON is valid and matches the loader (no API)."""
    if not DATA_FILE.exists():
        pytest.skip(f"Missing {DATA_FILE}")
    raw = _load_hf_json(DATA_FILE)
    samples = hf_records_to_eval_samples(raw)
    assert len(samples) >= 1
    assert all(1 <= s["expected_score"] <= 5 for s in samples)
