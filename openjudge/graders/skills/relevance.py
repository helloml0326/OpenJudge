# -*- coding: utf-8 -*-
"""
Skill Relevance Grader

Evaluates whether an AI Agent Skill's capabilities directly address a given task description.
"""

import textwrap
from typing import List, Optional

from loguru import logger

from openjudge.evaluation_strategy import BaseEvaluationStrategy
from openjudge.graders.base_grader import GraderError, GraderMode, GraderScore
from openjudge.graders.llm_grader import LLMGrader
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# pylint: disable=line-too-long

# English Prompt
SKILL_RELEVANCE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI skill evaluator. Your task is to assess how well an AI Agent Skill matches a given task description.

<Rubrics>
A well-matched skill should:
- Directly address the core capability required by the task without substantial rework.
- Be scoped to the same domain and type of task being requested.
- Have a name and description that unambiguously identify it as the right tool for the task.
- Provide concrete, actionable techniques or patterns that accomplish the primary goal — not merely infrastructure around it.

Points should be deducted in the following cases:
- The skill only partially overlaps with the task or requires significant domain adaptation.
- The skill targets a fundamentally different domain or problem type.
- The skill name and description suggest a different use case than the one requested.
- The skill's PRIMARY purpose is to measure, evaluate, or verify task outcomes rather than to directly accomplish the task goal itself. A measurement/evaluation tool for domain X is adjacent to — but not a direct match for — a task asking to DO X.
- The skill's content focuses on process scaffolding or infrastructure (e.g. how to define pass/fail criteria, how to track regressions) rather than on the concrete improvement techniques, patterns, or implementations the user needs.
</Rubrics>

<Steps>
- Read the task description carefully to understand what primary capability or outcome is needed.
- Identify whether the skill's core purpose is to directly accomplish the task or to measure/evaluate/scaffold around it.
- Evaluate the skill's name, description, and SKILL.md content against the task.
- Assign a score [1, 3] based on how directly the skill's primary purpose addresses the task.
- Provide a concise reason citing concrete evidence from the skill content.
</Steps>

<Constraints>
Base your evaluation strictly on the provided skill content; do not infer capabilities that are not described.
A score of 3 means the skill's primary purpose directly and unambiguously accomplishes the task.
A score of 1 means the skill targets a different domain or task type entirely.
Critical: a skill that is relevant to the task domain but whose primary purpose is evaluation, measurement, or process scaffolding rather than direct accomplishment should receive a score of 2, not 3.
</Constraints>

<Scale>
- 3: Direct match — the skill's primary purpose directly accomplishes the task goal; its name, description, and SKILL.md provide concrete actionable techniques/patterns that solve the task with little to no adaptation
- 2: Partial or adjacent match — the skill is relevant to the task's domain but either: (a) covers only a subset of what is needed while leaving notable gaps; or (b) its primary focus is on measuring, evaluating, or scaffolding around the task rather than directly doing it; meaningful overlap but the skill cannot fully substitute for a direct solution
- 1: Poor match — skill targets a different domain or fundamentally different task type; applying it to this task would require substantial rework
</Scale>

<Evaluation Examples>
The following examples illustrate how to apply the scoring scale. Use them as reference calibration points when evaluating.

Example 1: Score 3 — Direct match (code review skill vs. PR review task)
- Task: "Review a pull request for code quality issues, bugs, and style violations."
- Skill name: "code-review"
- Skill description: "Perform automated code reviews on pull requests, checking for bugs, style issues, and best practices."
- SKILL.md excerpt: "# Code Review\\n## Steps\\n1. Fetch the PR diff.\\n2. Analyze each changed file for bugs and style violations.\\n3. Post inline comments.\\n## Triggers: pull request, diff, code quality"
- Expected score: 3
- Reason: The skill is explicitly named and designed for code review; its description, trigger keywords, and step-by-step workflow directly match the requested task with no adaptation needed.

Example 2: Score 2 — Partial match (general document summarizer vs. meeting transcript task)
- Task: "Summarize a recorded meeting transcript, extracting action items and decisions."
- Skill name: "document-summarizer"
- Skill description: "Summarizes text documents up to 10 pages, producing concise paragraph summaries."
- SKILL.md excerpt: "# Document Summarizer\\n## Steps\\n1. Load the text.\\n2. Chunk by paragraphs.\\n3. Generate a unified summary."
- Expected score: 2
- Reason: The skill can summarize text and would partially address the task, but it is not designed for meeting transcripts specifically — it lacks support for extracting structured outputs like action items or decisions, requiring moderate adaptation.

Example 3: Score 1 — Poor match (financial report generator vs. code review task)
- Task: "Review a pull request for code quality issues."
- Skill name: "financial-report-generator"
- Skill description: "Generates quarterly financial reports from CSV data, including revenue charts and KPI summaries."
- SKILL.md excerpt: "# Financial Report Generator\\n## Steps\\n1. Load CSV data.\\n2. Compute KPIs.\\n3. Render charts and export PDF."
- Expected score: 1
- Reason: The skill targets financial data processing and report generation — a completely different domain and task type from code review. Applying it to the requested task would require a complete rewrite of all functionality.

Example 4: Score 2 — Adjacent match (evaluation framework vs. direct improvement task)
- Task: "Improve the quality of my AI agent's outputs."
- Skill name: "agent-eval-harness"
- Skill description: "Formal evaluation framework implementing eval-driven development principles: define pass/fail criteria, run capability and regression evals, measure pass@k reliability."
- SKILL.md excerpt: "# Eval Harness\\n## Philosophy\\nEval-Driven Development treats evals as unit tests of AI development.\\n## Workflow\\n1. Define success criteria.\\n2. Run evals.\\n3. Generate pass@k report."
- Expected score: 2
- Reason: The skill is domain-relevant (AI agent outputs) but its primary purpose is to measure and verify whether outputs improved, not to provide direct improvement techniques. It tells the user HOW to evaluate change, not HOW to achieve the improvement. This makes it an adjacent tool rather than the direct solution.

Example 5: Score 3 — Direct match (improvement patterns skill vs. direct improvement task)
- Task: "Improve the quality of my AI agent's outputs."
- Skill name: "agentic-eval-patterns"
- Skill description: "Patterns and techniques for evaluating and improving AI agent outputs: self-critique loops, evaluator-optimizer pipelines, test-driven refinement workflows."
- SKILL.md excerpt: "# Agentic Evaluation Patterns\\n## Pattern 1: Basic Reflection\\nGenerate → Evaluate → Critique → Refine loop.\\n## Pattern 2: Evaluator-Optimizer\\nSeparate generation and evaluation into distinct components.\\n## Pattern 3: Test-Driven Refinement\\nRun tests, capture failures, auto-fix."
- Expected score: 3
- Reason: The skill's primary purpose directly matches the task — it provides concrete improvement patterns (reflection, evaluator-optimizer, test-driven refinement) that an agent can immediately apply to raise output quality. Domain, name, description, and techniques all align with no adaptation needed.
</Evaluation Examples>

<Task Description>
{task_description}
</Task Description>

<Skill Name>
{skill_name}
</Skill Name>

<YAML Manifest>
{skill_manifest}
</YAML Manifest>

<Instruction Body>
{instruction_body}
</Instruction Body>

<Output Schema>
Provide your evaluation in the following structured JSON format:
{{
    "reason": "<brief explanation citing concrete evidence from the skill content for the assigned score>",
    "score": <integer 1, 2, or 3, where 3 = direct match and 1 = poor match>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SKILL_RELEVANCE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 评估员。你的任务是评估 AI Agent Skill 与给定任务描述的匹配程度。

<评分标准>
匹配度高的 Skill 应该：
- 直接满足任务所需的核心能力，无需大量重构。
- 与任务所属领域和任务类型保持一致。
- 名称和描述能够明确标识其为该任务的合适工具。
- 提供具体可操作的技术或模式来完成主要目标——而非仅提供围绕目标的基础设施。

以下情况应扣分：
- Skill 与任务仅部分重叠，或需要大幅领域适配。
- Skill 针对完全不同的领域或问题类型。
- Skill 的名称和描述暗示了与所请求任务不同的使用场景。
- Skill 的核心目的是对任务结果进行衡量、评估或验证，而非直接完成任务本身。针对领域 X 的测量/评估工具是完成 X 任务的辅助工具，而非直接匹配。
- Skill 内容侧重于过程脚手架或基础设施（例如如何定义通过/失败标准、如何追踪回归），而非用户实际需要的具体改进技术、模式或实现。
</评分标准>

<评估步骤>
- 仔细阅读任务描述，了解所需的核心能力或成果。
- 判断 Skill 的核心目的是直接完成任务，还是对其进行衡量/评估/搭建脚手架。
- 对照任务评估 Skill 的名称、描述和 SKILL.md 内容。
- 根据 Skill 的核心目的对任务的直接针对程度，给出评分 [1, 3]。
- 提供简明的理由，引用 Skill 内容中的具体证据。
</评估步骤>

<注意事项>
严格基于提供的 Skill 内容进行评估，不要推断未描述的能力。
3 分表示 Skill 的核心目的直接且明确地完成该任务。
1 分表示 Skill 完全针对不同的领域或任务类型。
重要：若 Skill 与任务领域相关，但其核心目的是评估、衡量或搭建过程脚手架，而非直接完成任务，则应给 2 分而非 3 分。
</注意事项>

<评分量表>
- 3：直接匹配——Skill 的核心目的直接完成任务目标；其名称、描述和 SKILL.md 提供了具体可操作的技术/模式，能以很少甚至不需要改动地解决该任务
- 2：部分匹配或邻近匹配——Skill 与任务领域相关，但：(a) 仅涵盖所需能力的子集，存在明显差距；或 (b) 其核心关注点是对任务进行衡量、评估或搭建脚手架，而非直接完成任务；有意义的重叠，但无法完全替代直接解决方案
- 1：匹配较差——Skill 针对不同领域或完全不同类型的任务；将其用于给定任务需要大量重构
</评分量表>

<评估示例>
以下示例说明如何应用评分量表，请将其作为参考校准点进行评估。

示例 1：3 分 — 直接匹配（代码审查 Skill vs. PR 审查任务）
- 任务：「审查一个 Pull Request，检查代码质量问题、bug 和风格违规。」
- Skill 名称：「code-review」
- Skill 描述：「对 Pull Request 执行自动化代码审查，检查 bug、风格问题和最佳实践。」
- SKILL.md 摘录：「# Code Review\\n## 步骤\\n1. 获取 PR diff。\\n2. 分析每个变更文件的 bug 和风格违规。\\n3. 发布行内注释。\\n## 触发词：pull request、diff、代码质量」
- 预期分数：3
- 理由：该 Skill 的名称和设计明确针对代码审查；其描述、触发词和分步工作流与请求任务直接匹配，无需任何适配。

示例 2：2 分 — 部分匹配（通用文档摘要 Skill vs. 会议记录任务）
- 任务：「对一份会议录音转写文本进行摘要，提取行动项和决策结论。」
- Skill 名称：「document-summarizer」
- Skill 描述：「对最长 10 页的文本文档进行摘要，生成简洁的段落摘要。」
- SKILL.md 摘录：「# Document Summarizer\\n## 步骤\\n1. 加载文本。\\n2. 按段落分块。\\n3. 生成统一摘要。」
- 预期分数：2
- 理由：该 Skill 具备文本摘要能力，可部分满足任务需求，但并非专门针对会议记录设计——缺乏对行动项或决策结论等结构化输出的支持，需要适度调整才能适用。

示例 3：1 分 — 匹配较差（财务报告生成 Skill vs. 代码审查任务）
- 任务：「审查一个 Pull Request，检查代码质量问题。」
- Skill 名称：「financial-report-generator」
- Skill 描述：「从 CSV 数据生成季度财务报告，包括营收图表和 KPI 摘要。」
- SKILL.md 摘录：「# Financial Report Generator\\n## 步骤\\n1. 加载 CSV 数据。\\n2. 计算 KPI。\\n3. 渲染图表并导出 PDF。」
- 预期分数：1
- 理由：该 Skill 面向财务数据处理和报告生成，与代码审查完全属于不同领域和任务类型。将其用于请求任务需要彻底重写所有功能。

示例 4：2 分 — 邻近匹配（评估框架 Skill vs. 直接改进任务）
- 任务：「提升我的 AI Agent 输出质量。」
- Skill 名称：「agent-eval-harness」
- Skill 描述：「实现 eval 驱动开发原则的正式评估框架：定义通过/失败标准、运行能力和回归评估、衡量 pass@k 可靠性。」
- SKILL.md 摘录：「# Eval Harness\\n## 理念\\nEval 驱动开发将 eval 视为 AI 开发的单元测试。\\n## 工作流\\n1. 定义成功标准。\\n2. 运行 eval。\\n3. 生成 pass@k 报告。」
- 预期分数：2
- 理由：该 Skill 领域相关（AI Agent 输出），但其核心目的是衡量和验证输出是否得到改善，而非提供直接的改进技术。它告诉用户如何评估变化，而非如何实现改进。这使它成为邻近工具，而非直接解决方案。

示例 5：3 分 — 直接匹配（改进模式 Skill vs. 直接改进任务）
- 任务：「提升我的 AI Agent 输出质量。」
- Skill 名称：「agentic-eval-patterns」
- Skill 描述：「评估和改进 AI Agent 输出的模式与技术：自我批评循环、评估器-优化器流水线、测试驱动精炼工作流。」
- SKILL.md 摘录：「# Agentic Evaluation Patterns\\n## 模式 1：基础反思\\n生成 → 评估 → 批评 → 精炼循环。\\n## 模式 2：评估器-优化器\\n将生成与评估分离为独立组件。\\n## 模式 3：测试驱动精炼\\n运行测试、捕获失败、自动修复。」
- 预期分数：3
- 理由：该 Skill 的核心目的与任务直接匹配——提供了具体的改进模式（反思、评估器-优化器、测试驱动精炼），Agent 可立即应用以提升输出质量。领域、名称、描述和技术完全一致，无需适配。
</评估示例>

<任务描述>
{task_description}
</任务描述>

<Skill 名称>
{skill_name}
</Skill 名称>

<YAML Manifest>
{skill_manifest}
</YAML Manifest>

<指令正文>
{instruction_body}
</指令正文>

<输出格式>
请按以下结构化 JSON 格式提供你的评估：
{{
    "reason": "<对所给分数的简要解释，引用 Skill 内容中的具体证据>",
    "score": <整数 1、2 或 3，其中 3 = 直接匹配，1 = 匹配较差>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SKILL_RELEVANCE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=SKILL_RELEVANCE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=SKILL_RELEVANCE_PROMPT_ZH,
            ),
        ],
    },
)


class SkillRelevanceGrader(LLMGrader):
    """
    Skill Relevance Grader

    Purpose:
        Evaluates whether an AI Agent Skill's described capabilities directly address
        a given task description. Useful for skill retrieval, routing, and gap analysis.

    What it evaluates:
        - Domain alignment: whether the skill targets the same domain as the task
        - Capability fit: whether the skill's features match what the task requires
        - Adaptation cost: how much rework would be needed to apply the skill to the task

    When to use:
        - Skill retrieval ranking: given a task, rank candidate skills by how well they match
        - Skill routing: select the most appropriate skill for an incoming user request
        - Skill gap analysis: detect tasks for which no existing skill provides a direct match
        - Recommender systems: surface the most applicable skill for a user request

    Scoring (3-level scale):
        - 3 (Direct match): Skill's primary purpose directly accomplishes the task goal with concrete actionable techniques; little to no adaptation needed
        - 2 (Partial/Adjacent match): Skill is domain-relevant but either covers only a subset, or its primary focus is on measuring/evaluating/scaffolding around the task rather than directly doing it
        - 1 (Poor match): Skill targets a different domain or fundamentally different task type

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 3] to pass (default: 2)
        template: Custom evaluation template (default: DEFAULT_SKILL_RELEVANCE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)

    Returns:
        GraderScore object with:
            - score: Matching score [1, 3] where 3 = direct match, 1 = poor match
            - reason: Explanation citing concrete evidence from the skill content
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.relevance import SkillRelevanceGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillRelevanceGrader(model=model, threshold=2)
        >>>
        >>> # Direct match
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Review a pull request for code quality issues.",
        ...     skill_name="code-review",
        ...     skill_manifest="name: code-review\\ndescription: Perform code reviews on pull requests, checking for bugs and style.",
        ...     instruction_body="# Code Review\\n## Steps\\n1. Fetch PR diff\\n2. Analyze for bugs...",
        ...     script_contents=[],
        ...     reference_contents=[],
        ... ))
        >>> print(result.score)   # 3 - Direct match
        >>>
        >>> # Poor match
        >>> result = asyncio.run(grader.aevaluate(
        ...     task_description="Generate a financial report from CSV data.",
        ...     skill_name="code-review",
        ...     skill_manifest="name: code-review\\ndescription: Perform code reviews on pull requests.",
        ...     instruction_body="# Code Review\\nReview code diffs for quality issues.",
        ...     script_contents=[],
        ...     reference_contents=[],
        ... ))
        >>> print(result.score)   # 1 - Poor match
        >>> print(result.reason)  # "Skill is designed for code review, not financial reporting."
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_RELEVANCE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 2,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillRelevanceGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum score [1, 3] to pass (default: 2)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_RELEVANCE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 3]
        """
        if not 1 <= threshold <= 3:
            raise ValueError(f"threshold must be in range [1, 3], got {threshold}")

        super().__init__(
            name="skill_relevance",
            mode=GraderMode.POINTWISE,
            description="Evaluate whether an AI Agent Skill is relevant to the task requirements",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.threshold = threshold

    async def _aevaluate(
        self,
        task_description: str,
        skill_name: str,
        skill_manifest: str,
        instruction_body: str,
        script_contents: List[str],
        reference_contents: List[str],
    ) -> GraderScore:
        """
        Evaluate how well an AI Agent Skill matches a given task description.

        Args:
            task_description: Description of the task the skill should accomplish
            skill_name: Name of the skill (from SkillManifest.name)
            skill_manifest: Raw YAML frontmatter string (from SkillManifest.raw_yaml)
            instruction_body: Markdown body of SKILL.md after the YAML frontmatter
                (from SkillPackage.instruction_body)
            script_contents: Text content of each executable script file
                (from SkillPackage.script_contents — SkillFile.content
                where SkillFile.is_script is True)
            reference_contents: Text content of each non-script referenced file
                (from SkillPackage.reference_contents — SkillFile.content
                for files in references/assets directories)

        Returns:
            GraderScore: Score in [1, 3] where:
                        3 = Direct match (skill's primary purpose directly accomplishes the task),
                        2 = Partial/adjacent match (domain-relevant but measurement/scaffolding focus, or only subset coverage),
                        1 = Poor match (different domain or fundamentally different task)

        Example:
            >>> result = await grader.aevaluate(
            ...     task_description="Summarize a PDF document.",
            ...     skill_name="pdf-summarizer",
            ...     skill_manifest="name: pdf-summarizer\\ndescription: Extracts and summarizes PDF documents up to 20 pages.",
            ...     instruction_body="# PDF Summarizer\\n## Steps\\n1. Load PDF\\n2. Summarize.",
            ...     script_contents=[],
            ...     reference_contents=[],
            ... )
        """
        try:
            result = await super()._aevaluate(
                task_description=task_description,
                skill_name=skill_name,
                skill_manifest=skill_manifest or "(none)",
                instruction_body=instruction_body or "(none)",
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata={**result.metadata, "threshold": self.threshold},
            )

        except Exception as e:
            logger.exception(f"Error evaluating skill relevance: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SkillRelevanceGrader", "DEFAULT_SKILL_RELEVANCE_TEMPLATE"]
