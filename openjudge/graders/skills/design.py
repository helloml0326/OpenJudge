# -*- coding: utf-8 -*-
"""
Skill Design Grader

Evaluates whether an AI Agent Skill's internal structure is well-designed across six
dimensions: Knowledge Delta, Mindset + Procedures, Specification Compliance,
Progressive Disclosure, Freedom Calibration, and Practical Usability.
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
SKILL_STRUCTURE_PROMPT_EN = textwrap.dedent(
    """
You are a professional AI Skill architect. Your task is to assess whether an AI Agent Skill is well-designed across seven dimensions: Knowledge Delta, Mindset + Procedures, Specification Compliance, Progressive Disclosure, Freedom Calibration, Practical Usability, and Anti-Pattern Quality (supplementary).

<Rubrics>
D1–D6 are primary dimensions that determine the score. D7 is a supplementary dimension with lower weight — a strong NEVER list can lift a borderline score but its absence alone does not pull a score below 3.

**D1 — Knowledge Delta**
Does the Skill add genuine expert knowledge beyond what Claude already knows?
- Core formula: Good Skill = Expert-only Knowledge − What Claude Already Knows
- Expert content (keep): decision trees for non-obvious choices, trade-offs only experts know, edge cases from real-world experience, domain-specific thinking frameworks
- Redundant content (penalize): "What is X" explanations for basic concepts, step-by-step tutorials for standard operations, generic best practices ("write clean code", "handle errors"), definitions of industry-standard terms
- The test: "Does Claude already know this?" — redundant content wastes shared context window tokens

**D2 — Mindset + Procedures**
Does the Skill transfer expert thinking patterns along with necessary domain-specific procedures?
- Valuable mindset: "Before doing X, ask yourself..." frameworks that shape how the Agent approaches problems; purpose/constraints/trade-off questions
- Valuable domain procedures: workflows Claude hasn't been trained on, non-obvious correct ordering ("validate BEFORE packing, not after"), critical steps easy to miss, domain-specific sequences
- Redundant procedures (penalize): generic file operations, standard programming patterns, common library usage well-documented elsewhere
- The test: (1) Does it tell Claude WHAT to think about? (2) Does it tell Claude HOW to do things it wouldn't know?

**D3 — Specification Compliance (especially description)**
Does the Skill follow official format requirements? The description is THE MOST CRITICAL field — it is the only thing the Agent reads before deciding to load the skill.
Skill activation flow:
  User Request → Agent sees ALL skill descriptions → Decides which to activate
                 (only descriptions, not bodies!)
  If description doesn't match → Skill NEVER gets loaded
- Valid `name`: lowercase, alphanumeric + hyphens only, ≤ 64 characters
- Description must answer THREE questions: WHAT (specific capabilities, not "handles X tasks"), WHEN (explicit trigger scenarios: "Use when...", "When user asks for..."), KEYWORDS (file extensions, domain terms, action verbs)
- Penalize: description is vague; missing WHEN triggers; "When to use this Skill" guidance placed only in the body — the body is loaded AFTER the triggering decision is already made

**D4 — Progressive Disclosure**
Does the Skill implement proper content layering?
Three loading layers:
  Layer 1 — Metadata (always in memory): only name + description (~100 tokens per skill)
  Layer 2 — SKILL.md body (loaded after triggering): detailed guidelines, decision trees — ideal < 500 lines, < 300 preferred
  Layer 3 — References (loaded on demand): scripts/, references/, assets/ — no size limit
- High value: MANDATORY loading triggers embedded at relevant workflow decision points (not just listed at the end); "Do NOT Load" guidance to prevent over-loading; SKILL.md stays focused as a routing/decision layer
- Penalize: SKILL.md is a dump of all content (>500 lines, no offloading); references directory exists but files are never triggered (orphan references); loading guidance only listed at the end without workflow integration
- For simple skills (<100 lines, no references): evaluate on conciseness and self-containment instead

**D5 — Freedom Calibration**
Is the constraint level appropriate for the task's fragility?
The freedom spectrum:
  Creative/design tasks      → High freedom:   principles, intent, trade-offs — NOT rigid step-by-step scripts
  Code review / analysis     → Medium freedom:  prioritized criteria, judgment-based ordering
  File format / irreversible → Low freedom:    exact scripts, precise parameters, explicit do-not-deviate instructions
- The test: "If the Agent makes a mistake, what is the consequence?" — high consequence → low freedom; low consequence → high freedom
- Penalize: rigid scripts for creative tasks; vague guidance for fragile/destructive operations; uniform constraint level applied regardless of per-section fragility

**D6 — Practical Usability**
Can an Agent actually use this Skill effectively?
- Decision trees: for multi-path scenarios, is there clear guidance on which path to take?
- Code examples: do they actually work, or are they pseudocode that would break?
- Error handling: what if the main approach fails — are fallbacks provided?
- Edge cases: are unusual but realistic scenarios covered?
- Actionability: can Agent immediately act, or does it need to figure things out first?
- Penalize: vague instructions ("use appropriate tools", "handle errors properly"); missing fallbacks for known failure modes; no guidance on edge cases

**D7 — Anti-Pattern Quality [supplementary, lower weight]**
Does the Skill contain an effective NEVER list encoding hard-won expert knowledge?
- High value: specific domain anti-patterns with non-obvious reasons ("NEVER use X because [specific problem only experience teaches]"); failure modes from real-world practice; the test — would an expert say "yes, I learned this the hard way"?
- Low value: absent NEVER list; only generic warnings that apply to any task ("be careful", "avoid errors", "handle edge cases") with no domain-specific reasoning
- Scoring note: a strong NEVER list can lift a borderline score by half a point; a missing or vague NEVER list is a minor gap that does not independently drive the score below 3
</Rubrics>

<CommonFailurePatterns>
Watch for these patterns — each indicates a specific dimension failure:
- The Tutorial [D1]: explains basic concepts, standard library usage — wastes tokens on what Claude already knows
- The Checkbox Procedure [D2]: generic Step 1/2/3 procedures with no domain-specific thinking frameworks
- The Invisible Skill [D3]: great content but description missing WHEN triggers and domain KEYWORDS
- The Wrong Location [D3]: "When to use this Skill" section placed in the body, not in the description field
- The Dump [D4]: SKILL.md is 500+ lines with everything included, no content offloading to references/
- The Orphan References [D4]: references/ directory exists but files are never loaded (no MANDATORY triggers embedded in workflow)
- The Freedom Mismatch [D5]: rigid scripts for creative tasks, or vague guidance for fragile/destructive operations
- The Vague Usability [D6]: "use appropriate tools", "consider edge cases" — no decision trees, no fallbacks, no actionable guidance
- The Vague Warning [D7]: "be careful", "avoid errors" — NEVER list absent or contains only generic statements with no domain-specific reasoning
</CommonFailurePatterns>

<Steps>
1. Read the skill's name, description, and full SKILL.md content completely.
2. Check Knowledge Delta (D1): for each section, ask "Does Claude already know this?" — mark Expert / Activation / Redundant.
3. Check Mindset + Procedures (D2): does it shape thinking AND provide domain-specific procedures Claude wouldn't know?
4. Check Specification Compliance (D3): does the description answer WHAT + WHEN + contain searchable KEYWORDS? Is any trigger guidance buried in the body?
5. Check Progressive Disclosure (D4): is SKILL.md appropriately sized? If references exist, are they loaded with MANDATORY triggers embedded in the workflow, not just listed?
6. Check Freedom Calibration (D5): for each section, does the constraint level match the consequence of error?
7. Check Practical Usability (D6): are there decision trees, working examples, fallbacks, and edge case coverage?
8. Check Anti-Pattern Quality (D7, supplementary): is the NEVER list specific, domain-relevant, and explained with non-obvious reasons? Or absent / generic?
9. Note any common failure patterns detected.
10. Assign a score [1, 5] based primarily on D1–D6; use D7 as a tiebreaker for borderline cases.
11. Provide a concise reason citing specific evidence from the skill content.
</Steps>

<Constraints>
Base your evaluation strictly on the provided skill content; do not infer structure or intent that is not present.
If SKILL.md content is empty or missing, score is 1.
D1–D6 are primary — the score is determined mainly by how well the skill satisfies these six dimensions.
D7 is supplementary — a strong NEVER list can push a borderline score up; its absence or weakness alone does not reduce the score below what D1–D6 warrant.
A score of 5 means the skill excels across all primary dimensions with no significant gaps.
A score of 1 means the skill fails most criteria and needs fundamental redesign.
</Constraints>

<Scale>
- 5: Excellent — pure knowledge delta; expert thinking frameworks + domain procedures Claude wouldn't know; description fully answers WHAT + WHEN + KEYWORDS; SKILL.md properly sized with MANDATORY triggers embedded in workflow; constraint level per-section calibrated to task fragility; comprehensive usability with decision trees, working examples, and fallbacks; bonus: expert-grade NEVER list with specific non-obvious domain reasons
- 4: Strong — mostly expert knowledge with minor redundancy; good mindset and procedures with small gaps; description covers WHAT/WHEN but may lack some keywords; content layering mostly correct with minor trigger gaps; freedom mostly calibrated with one mismatch; usability covers common cases but misses some edge cases; NEVER list may be partially specific or thin
- 3: Adequate — mixed expert and redundant content; procedures present but lean generic or lack thinking frameworks; description has WHAT but weak or missing WHEN triggers; SKILL.md borderline oversized or mediocre trigger quality; some freedom or usability issues; NEVER list generic or missing (acceptable at this level)
- 2: Weak — mostly redundant content; generic procedures without thinking frameworks; vague description missing trigger scenarios; SKILL.md dump or orphan references; significant freedom mismatch; usability relies on vague guidance with no fallbacks
- 1: Poor — explains basics Claude already knows; no domain-specific thinking or procedures; description too generic to trigger correctly; no content layering; severely mismatched constraint level; no actionable guidance or decision trees
</Scale>

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
    "reason": "<concise explanation citing concrete evidence from the skill content, covering: (1) knowledge delta — expert vs redundant ratio, (2) mindset and procedures — thinking frameworks and domain-specific workflows, (3) description completeness — WHAT/WHEN/KEYWORDS present?, (4) content layering — SKILL.md size and trigger quality, (5) freedom calibration — constraint level vs task fragility, (6) practical usability — decision trees, fallbacks, edge cases, (7) anti-pattern quality — NEVER list specific or generic?, and any failure patterns detected>",
    "score": <integer 1–5, where 5 = excellent across all dimensions and 1 = poor>
}}
</Output Schema>

JSON:
"""
).strip()

# Chinese Prompt
SKILL_STRUCTURE_PROMPT_ZH = textwrap.dedent(
    """
你是一名专业的 AI Skill 架构师。你的任务是从以下七个维度评估 AI Agent Skill 的设计质量：知识增量、思维模式与流程、规范合规性、渐进式披露、自由度校准、实用性、反模式质量（补充维度）。

<评分标准>
D1–D6 是主要维度，决定最终评分。D7 是补充维度，权重较低——高质量的 NEVER 列表可以提升临界评分，但 NEVER 列表缺失或模糊本身不会将评分拉低至 3 以下。

**D1 — 知识增量**
Skill 是否提供了超越 Claude 已有知识的真正专家知识？
- 核心公式：好的 Skill = 专家专属知识 − Claude 已知的知识
- 专家内容（保留）：针对非直觉选择的决策树、只有专家才知道的权衡取舍、来自实战经验的边界情况、领域特有思维框架
- 冗余内容（扣分）：对基础概念的"什么是 X"解释、标准操作的逐步教程、通用最佳实践（"写干净的代码"、"处理错误"）、行业标准术语的定义
- 测试方法："Claude 已经知道这个了吗？"——冗余内容浪费共享的上下文窗口 token

**D2 — 思维模式与领域流程**
Skill 是否传递了专家思维模式以及必要的领域专属流程？
- 有价值的思维模式："在做 X 之前，问自己……"框架，引导 Agent 如何思考问题；目的/约束/权衡取舍问题
- 有价值的领域流程：Claude 未经训练的工作流、非直觉的正确顺序（"先验证再打包，而非之后"）、容易遗漏的关键步骤、领域特有序列
- 冗余流程（扣分）：通用文件操作、标准编程模式、有完善文档的常用库用法
- 测试方法：（1）它是否告诉 Claude 该思考什么？（2）它是否告诉 Claude 如何做它本来不知道的事？

**D3 — 规范合规性（尤其是 description）**
Skill 是否遵循官方格式要求？description 是最关键的字段——这是 Agent 决定是否加载 Skill 前唯一读取的内容。
Skill 激活流程：
  用户请求 → Agent 查看所有 Skill 的 description → 决定激活哪个
              （只看 description，不看正文！）
  description 不匹配 → Skill 永远不会被加载
- 有效的 `name`：小写字母数字 + 连字符，≤ 64 字符
- description 必须回答三个问题：WHAT（具体能做什么，而非"处理 X 相关功能"）、WHEN（明确的触发场景："使用时机……"、"当用户要求……"）、KEYWORDS（文件扩展名、领域术语、动作动词，使其可被检索）
- 扣分：description 模糊；缺少 WHEN 触发词；"使用时机"信息只出现在正文而非 description 字段——正文是激活决策做出之后才加载的

**D4 — 渐进式披露**
Skill 是否实现了合理的内容分层？
三层加载架构：
  第1层 — 元数据（始终在内存中）：仅 name + description（每个 Skill 约100 token）
  第2层 — SKILL.md 正文（触发后加载）：详细指引、决策树——理想 < 500 行，建议 < 300 行
  第3层 — 参考资源（按需加载）：scripts/、references/、assets/——无大小限制
- 高价值：MANDATORY 加载触发器嵌入在工作流的相关决策节点（而非仅在末尾列出）；附有"Do NOT Load"指引防止过度加载；SKILL.md 保持精简，作为路由/决策层
- 扣分：SKILL.md 堆砌所有内容（>500行，无内容卸载）；references 目录存在但文件从未被触发（孤立引用）；加载指引仅在末尾列出，未集成到工作流
- 简单 Skill（<100行，无 references）：改为基于简洁性和自包含性进行评估

**D5 — 自由度校准**
约束程度是否与任务脆弱性相匹配？
自由度光谱：
  创意/设计任务       → 高自由度：原则、意图、权衡——而非刚性步骤脚本
  代码审查/分析       → 中等自由度：优先级标准，需要判断
  文件格式/不可逆操作  → 低自由度：精确脚本、明确参数、不得偏离的明确指令
- 测试方法："如果 Agent 在这里出错，后果是什么？"——后果严重 → 低自由度；后果轻微 → 高自由度
- 扣分：对创意任务强加刚性步骤脚本；对可能导致数据丢失、文件损坏的操作只给出模糊指引；全文使用统一约束级别而不考虑各章节脆弱性差异

**D6 — 实用性**
Agent 是否能真正有效地使用此 Skill？
- 决策树：对于多路径场景，是否有清晰的路径选择指引？
- 代码示例：示例是否真实可用，还是会报错的伪代码？
- 错误处理：主方案失败时怎么办——是否提供了备选方案？
- 边界情况：是否覆盖了不常见但现实存在的场景？
- 可操作性：Agent 是否能立即行动，还是需要自己摸索？
- 扣分：模糊指令（"使用合适的工具"、"妥善处理错误"）；已知失败情形缺少备选方案；无边界情况指引

**D7 — 反模式质量【补充维度，权重较低】**
Skill 是否包含传递实战知识的有效 NEVER 列表？
- 高价值：具有非直觉原因的具体领域反模式（"NEVER 使用 X，因为[只有经验才能告诉你的具体问题]"）；来自实战的失败模式；测试标准——专家看到这条是否会说"是的，我就是这样踩坑的"？
- 低价值：NEVER 列表缺失；仅包含适用于任何任务的通用警告（"小心"、"避免错误"、"处理边界情况"），没有领域特有的具体原因
- 评分说明：高质量 NEVER 列表可将临界评分上调半档；NEVER 列表缺失或模糊属于小缺口，本身不会将评分拉低至 D1–D6 应得分数以下
</评分标准>

<常见失败模式>
识别以下模式——每种模式对应特定维度的失败：
- 教程模式 [D1]：解释基础概念、标准库用法——浪费 token 在 Claude 已知的知识上
- 清单流程 [D2]：通用的第1步/第2步/第3步，无领域特有思维框架
- 隐形 Skill [D3]：内容优质但 description 模糊，缺少 WHEN 触发词和领域 KEYWORDS
- 错误位置 [D3]："使用时机"放在正文而非 description 字段
- 堆砌模式 [D4]：SKILL.md 超过 500 行，包含所有内容，无内容卸载到 references/
- 孤立引用 [D4]：references/ 目录存在但文件从未被加载（工作流无嵌入的 MANDATORY 触发器）
- 自由度失配 [D5]：对创意任务的刚性脚本，或对脆弱/破坏性操作的模糊指引
- 模糊实用性 [D6]："使用合适的工具"、"处理边界情况"——无决策树、无备选方案、无可操作指引
- 模糊警告 [D7]："小心"、"避免错误"——NEVER 列表缺失或仅含通用表述，无领域特有原因
</常见失败模式>

<评估步骤>
1. 完整阅读 Skill 的 name、description 和完整 SKILL.md 内容。
2. 检查知识增量（D1）：对每个章节问"Claude 已经知道这个吗？"——标记为专家级/激活提醒/冗余。
3. 检查思维模式与流程（D2）：是否既塑造了思维方式，又提供了 Claude 本来不知道的领域专属流程？
4. 检查规范合规性（D3）：description 是否回答了 WHAT + WHEN + 包含可检索的关键词？是否有触发信息被埋在正文中？
5. 检查渐进式披露（D4）：SKILL.md 是否大小合适？如果存在 references，是否通过嵌入工作流的 MANDATORY 触发器加载，而非仅列出？
6. 检查自由度校准（D5）：每个章节的约束级别是否与该章节的出错后果相匹配？
7. 检查实用性（D6）：是否有决策树、可用的代码示例、备选方案以及边界情况覆盖？
8. 检查反模式质量（D7，补充）：NEVER 列表是否具体、领域相关，且附有非直觉原因？还是缺失/模糊？
9. 记录检测到的常见失败模式。
10. 以 D1–D6 为主要依据给出 [1, 5] 评分；D7 作为临界情况的加分项。
11. 提供简明理由，引用 Skill 内容的具体证据。
</评估步骤>

<注意事项>
严格基于提供的 Skill 内容进行评估，不要推断文本中未呈现的结构或意图。
如果 SKILL.md 内容为空或缺失，则评分为 1。
D1–D6 是主要维度——评分主要由这六个维度决定。
D7 是补充维度——高质量 NEVER 列表可将临界评分上调；NEVER 列表缺失或薄弱本身不会将评分拉低至 D1–D6 应得分数以下。
5 分表示 Skill 在全部主要维度上均表现卓越，无明显缺口。
1 分表示 Skill 未能满足大多数标准，需要从根本上重新设计。
</注意事项>

<评分量表>
- 5：卓越——纯知识增量无冗余；专家思维框架 + 领域专属流程；description 完整回答 WHAT + WHEN + KEYWORDS；SKILL.md 大小合适，MANDATORY 触发器嵌入工作流；约束级别逐章节匹配任务脆弱性；全面实用性含决策树、可用示例和备选方案；加分项：含非直觉原因的专家级 NEVER 列表
- 4：良好——以专家知识为主，有少量冗余；思维模式和流程较好，有小缺口；description 覆盖 WHAT/WHEN，但部分关键词缺失；内容分层基本正确，触发器有小问题；自由度基本校准，有一处失配；实用性覆盖常见情形，遗漏部分边界情况；NEVER 列表可能较薄弱或部分具体
- 3：尚可——专家知识与冗余内容混杂；流程存在但偏通用或缺乏思维框架；description 有 WHAT 但 WHEN 触发词薄弱或缺失；SKILL.md 接近超限或触发质量一般；存在一定自由度校准问题；实用性基本可用但局部模糊；NEVER 列表模糊或缺失（此分段可接受）
- 2：薄弱——以解释 Claude 已知知识的冗余内容为主；以通用流程为主，缺乏思维框架；description 模糊或缺少触发场景；SKILL.md 堆砌或存在孤立引用；关键章节有明显自由度失配；实用性依赖模糊指引，无备选方案
- 1：较差——解释 Claude 已知的基础知识；无领域特有思维或流程；description 过于通用无法正确触发；无内容分层；约束级别与任务严重失配；无可操作指引或决策树
</评分量表>

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
    "reason": "<简明解释，引用 Skill 内容的具体证据，涵盖：（1）知识增量——专家级与冗余内容的比例，（2）思维模式与流程——思维框架和领域专属工作流，（3）description 完整性——WHAT/WHEN/关键词是否齐全？，（4）内容分层——SKILL.md 大小及触发器质量，（5）自由度校准——约束级别与任务脆弱性是否匹配，（6）实用性——决策树、备选方案、边界情况，（7）反模式质量——NEVER 列表具体还是模糊？，以及检测到的失败模式>",
    "score": <整数 1–5，其中 5 = 全维度卓越，1 = 较差>
}}
</输出格式>

JSON:
"""
).strip()

# Build default template from prompts
DEFAULT_SKILL_STRUCTURE_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_EN,
            ),
            ChatMessage(
                role="user",
                content=SKILL_STRUCTURE_PROMPT_EN,
            ),
        ],
        LanguageEnum.ZH: [
            ChatMessage(
                role="system",
                content=LLMGrader.SYSTEM_PROMPT_ZH,
            ),
            ChatMessage(
                role="user",
                content=SKILL_STRUCTURE_PROMPT_ZH,
            ),
        ],
    },
)


class SkillDesignGrader(LLMGrader):
    """
    Skill Design Grader

    Purpose:
        Evaluates whether an AI Agent Skill is well-designed by assessing seven dimensions
        derived from official Skill design specifications. Helps identify structural issues
        and improvement opportunities before deployment.

    What it evaluates:
        - Knowledge Delta (primary): Whether the skill adds genuine expert knowledge beyond
          what Claude already knows — expert decision trees, domain trade-offs, real-world
          edge cases — rather than redundant basic explanations or generic best practices.
        - Mindset + Procedures (primary): Whether the skill transfers expert thinking frameworks
          ("Before doing X, ask yourself...") AND domain-specific procedures Claude wouldn't
          know — not generic Step 1/2/3 operations Claude can figure out on its own.
        - Specification Compliance (primary): Whether frontmatter is valid and the description
          field answers WHAT/WHEN/KEYWORDS so the Agent can discover and trigger the skill.
          The description is the only field read before the loading decision — vague = invisible.
        - Progressive Disclosure (primary): Whether heavy content is offloaded to references/
          with MANDATORY loading triggers embedded at workflow decision points (not just listed),
          keeping SKILL.md focused (< 500 lines, < 300 preferred).
        - Freedom Calibration (primary): Whether the constraint level per section matches the
          task's fragility — high freedom (principles) for creative tasks, exact scripts for
          destructive/fragile operations, calibrated per section not uniformly applied.
        - Practical Usability (primary): Whether an Agent can actually act on the skill —
          decision trees for multi-path scenarios, working code examples, fallbacks for failure
          modes, and edge case coverage.
        - Anti-Pattern Quality (supplementary, lower weight): Whether the skill contains an
          effective NEVER list with specific, domain-relevant anti-patterns and non-obvious
          reasons. A strong NEVER list can lift a borderline score; its absence alone does
          not pull the score below what the primary dimensions warrant.

    When to use:
        - Auditing newly authored Skill packages before merging into a skill library
        - Automated CI checks on skill quality in a skills repository
        - Comparing competing skill designs for the same capability
        - Coaching skill authors on structural improvements

    Scoring (5-level scale):
        - 5 (Excellent): Pure knowledge delta; expert thinking frameworks + domain procedures;
          description fully answers WHAT/WHEN/KEYWORDS; SKILL.md properly sized with MANDATORY
          triggers embedded in workflow; per-section freedom calibration; comprehensive usability
        - 4 (Strong): Mostly expert knowledge with minor redundancy; good mindset and procedures
          with small gaps; description mostly complete; content layering mostly correct; minor
          freedom or usability gaps
        - 3 (Adequate): Mixed expert and redundant content; procedures present but lean generic;
          description has WHAT but weak WHEN; borderline SKILL.md size or mediocre trigger
          quality; some freedom or usability issues
        - 2 (Weak): Mostly redundant content; generic procedures; vague description; SKILL.md
          dump or orphan references; significant freedom mismatch; no fallbacks
        - 1 (Poor): Explains basics Claude knows; no domain procedures or thinking frameworks;
          description too generic to trigger; no content layering; severely mismatched freedom;
          no actionable guidance

    Args:
        model: BaseChatModel instance or dict config for OpenAIChatModel
        threshold: Minimum score [1, 5] to pass (default: 3)
        template: Custom evaluation template (default: DEFAULT_SKILL_STRUCTURE_TEMPLATE)
        language: Prompt language - EN or ZH (default: LanguageEnum.EN)
        strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

    Returns:
        GraderScore object with:
            - score: Structure score [1, 5] where 5 = excellent, 1 = poor
            - reason: Summary covering knowledge delta, mindset and procedures, description
                      completeness, content layering, freedom calibration, practical usability,
                      and detected failure patterns
            - metadata: Threshold and evaluation details

    Example:
        >>> import asyncio
        >>> from openjudge.models.openai_chat_model import OpenAIChatModel
        >>> from openjudge.graders.skills.design import SkillDesignGrader
        >>>
        >>> model = OpenAIChatModel(api_key="sk-...", model="qwen3-32b")
        >>> grader = SkillDesignGrader(model=model, threshold=3)
        >>>
        >>> # Well-designed skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     skill_name="docx-editor",
        ...     skill_manifest=(
        ...         "name: docx-editor\\ndescription: Create, edit, and analyze .docx files "
        ...         "including tracked changes. Use when working with Word documents."
        ...     ),
        ...     instruction_body="# NEVER\\n- NEVER use tracked-changes blindly...\\n## Steps\\n...",
        ...     script_contents=[],
        ...     reference_contents=[],
        ... ))
        >>> print(result.score)   # 4 or 5 - Strong / Excellent
        >>>
        >>> # Poorly designed skill
        >>> result = asyncio.run(grader.aevaluate(
        ...     skill_name="helper",
        ...     skill_manifest="name: helper\\ndescription: A helpful skill for various tasks.",
        ...     instruction_body="# Helper\\nThis skill helps you do things. Be careful with errors.",
        ...     script_contents=[],
        ...     reference_contents=[],
        ... ))
        >>> print(result.score)   # 1 - Poor
        >>> print(result.reason)  # "Redundant content; description too vague..."
    """

    DEFAULT_TEMPLATE = DEFAULT_SKILL_STRUCTURE_TEMPLATE

    def __init__(
        self,
        model: BaseChatModel | dict,
        threshold: float = 3,
        template: Optional[PromptTemplate] = None,
        language: LanguageEnum = LanguageEnum.EN,
        strategy: BaseEvaluationStrategy | None = None,
    ):
        """
        Initialize SkillDesignGrader.

        Args:
            model: BaseChatModel instance or dict config for OpenAIChatModel
            threshold: Minimum score [1, 5] to pass (default: 3)
            template: PromptTemplate for evaluation prompts (default: DEFAULT_SKILL_STRUCTURE_TEMPLATE)
            language: Language for prompts (default: LanguageEnum.EN)
            strategy: The evaluation strategy to use. Defaults to DirectEvaluationStrategy.

        Raises:
            ValueError: If threshold is not in range [1, 5]
        """
        if not 1 <= threshold <= 5:
            raise ValueError(f"threshold must be in range [1, 5], got {threshold}")

        super().__init__(
            name="skill_design",
            mode=GraderMode.POINTWISE,
            description="Evaluate design quality of an AI Agent Skill across six primary dimensions (knowledge delta, mindset and procedures, specification compliance, progressive disclosure, freedom calibration, practical usability) plus anti-pattern quality as a supplementary dimension",
            model=model,
            template=template or self.DEFAULT_TEMPLATE,
            language=language,
            strategy=strategy,
        )
        self.threshold = threshold

    async def _aevaluate(
        self,
        skill_name: str,
        skill_manifest: str,
        instruction_body: str,
        script_contents: List[str],
        reference_contents: List[str],
    ) -> GraderScore:
        """
        Evaluate the structural quality of an AI Agent Skill.

        Args:
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
            GraderScore: Score in [1, 5] where:
                        5 = Excellent across all six dimensions,
                        4 = Strong with minor gaps,
                        3 = Adequate with some notable issues,
                        2 = Weak with significant gaps,
                        1 = Poor; fails most criteria.

        Example:
            >>> result = await grader.aevaluate(
            ...     skill_name="pdf-processor",
            ...     skill_manifest="name: pdf-processor\\ndescription: Extract text from PDF files.",
            ...     instruction_body="# NEVER\\n- NEVER load files > 50 MB...\\n## Steps\\n...",
            ...     script_contents=[],
            ...     reference_contents=[],
            ... )
        """
        try:
            # Kept for API parity with other skill graders; prompts currently use manifest + body only.
            _ = (script_contents, reference_contents)
            result = await super()._aevaluate(
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
            logger.exception(f"Error evaluating skill structure: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SkillDesignGrader", "DEFAULT_SKILL_STRUCTURE_TEMPLATE"]
