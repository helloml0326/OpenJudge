# Generate Graders from Data

Automatically create evaluation graders from various data sources instead of manually designing criteria. The system learns or derives rubrics from task descriptions, queries, unlabeled data, labeled preferences, external knowledge, or existing models.

**Key benefits:**

- **Save time** — Eliminate manual rubric design
- **Data-driven** — Learn criteria from actual examples
- **Consistent** — Produce reproducible evaluation standards
- **Scalable** — Quickly prototype graders for new domains

---

## When to Use This Approach

**Use grader generation when:**

- Manual rubric design is too time-consuming or subjective
- You want to quickly prototype graders for new domains
- Your evaluation criteria are implicit and hard to articulate
- You need consistent, reproducible evaluation standards
- You want to leverage existing data or knowledge sources

**Don't use grader generation when:**

- Your criteria are already well-defined and documented
- You need highly specialized domain expertise not captured in any source
- Simple rule-based evaluation is sufficient

---

## Generator Methods

GraderGenerator supports multiple generation strategies:

| Category | Method | Input | Status |
|----------|--------|-------|--------|
| **Supervised** | Auto-Rubric | Labeled preferences/scores | ✅ Available |
| **Unsupervised** | Task Description-Based | Task description text | 🔜 Planned |
| **Unsupervised** | Query-Specific-Based | Individual queries | 🔜 Planned |
| **Unsupervised** | QA-Based | Unlabeled Q&A pairs | 🔜 Planned |
| **Transfer** | GeneratePrompt-Based | Generation prompts | 🔜 Planned |
| **Knowledge** | Knowledge-Based | Documents, KGs, specs | 🔜 Planned |
| **Distillation** | Model-Based | Trained reward models | 🔜 Planned |
| **Composite** | Hybrid | Multiple sources | 🔜 Planned |

> **Note:** Currently, **Auto-Rubric** (`IterativeRubricsGenerator`) is available for supervised rubric generation from labeled data.

---

## Generate Grader from Labeled Data (Auto-Rubric)

The **Auto-Rubric** method (`IterativeRubricsGenerator`) extracts evaluation rubrics from labeled preference data using a supervised approach.

### How It Works

Auto-Rubric automatically extracts evaluation rubrics from preference data without training. Based on [Auto-Rubric: Learning to Extract Generalizable Criteria for Reward Modeling](https://arxiv.org/abs/2510.17314).

**Core concept:** Instead of manually writing criteria like "The response should be factually accurate," the system discovers these criteria by analyzing what makes labeled data good or bad.

**Key insight:** Evaluation rubrics underlying human preferences generalize strongly across diverse queries, enabling remarkable data efficiency.

**Two-stage approach:**

1. **Infer query-specific rubrics** — Uses validation-guided Propose-Evaluate-Revise pipeline for quality control
2. **Generalize to core set** — Maximizes information-theoretic coding rate (MCR²) to create compact, non-redundant rubrics

**Output:** Interpretable, hierarchical "Theme-Tips" rubric set.

**Data efficiency:** Using just 70 preference pairs (1.5% of source data), this method enables smaller models like Qwen3-8B to outperform specialized, fully-trained counterparts with full interpretability.

<div style="border: 1px solid #e5e7eb; border-radius: 0.5rem; overflow: hidden; margin: 1.5em 0;">
  <iframe src="../assets/auto_rubric_overview.pdf#view=FitH" width="100%" height="600px" style="border: none; display: block;">
    <p>Unable to display PDF. <a href="../assets/auto_rubric_overview.pdf" target="_blank">View Auto-Rubric Pipeline diagram (PDF)</a></p>
  </iframe>
</div>

---

### When to Use Auto-Rubric

**Use Auto-Rubric when:**

- You have labeled evaluation data (preference pairs or scored responses)
- You want to quickly prototype graders for new domains
- Manual rubric design is too time-consuming
- You need data-driven criteria rather than subjective opinions
- Your evaluation task has implicit criteria that are hard to articulate

**Don't use Auto-Rubric when:**

- You have no labeled data
- Your criteria are already well-defined and documented
- Your task requires domain knowledge that isn't in the data


---

## Quick Start

### Choose Your Evaluation Mode

| Mode | Config Class | Use Case | Output |
|------|--------------|----------|--------|
| **Pointwise** | `IterativePointwiseRubricsGeneratorConfig` | Score individual responses (e.g., 1-5 rating) | `score`, `reason` |
| **Listwise** | `IterativeListwiseRubricsGeneratorConfig` | Rank multiple responses (e.g., A > B > C) | `ranking` |

### Prepare Your Data

**Pointwise Evaluation:**

```python
dataset = [
    {
        "query": "What is photosynthesis?",
        "response": "Photosynthesis is the process by which plants convert sunlight into energy...",
        "label_score": 5  # Ground truth score (use "label_score" key)
    },
    {
        "query": "Explain quantum mechanics",
        "response": "It's like really small stuff that does weird things.",
        "label_score": 2  # Poor quality
    },
    # ... more examples (recommend 50-200)
]
```

**Listwise Evaluation:**

```python
dataset = [
    {
        "query": "How do I learn Python?",
        "responses": [
            "Start with online tutorials and practice daily.",
            "Python is a programming language.",
            "Buy a book and read it cover to cover."
        ],
        "label_rank": [1, 3, 2]  # Ranking labels (use "label_rank" key)
    },
    # ... more examples
]
```

### Configure the Generator

**Pointwise Configuration:**

```python
from rm_gallery.core.generator.iterative_rubric import (
    IterativeRubricsGenerator,
    IterativePointwiseRubricsGeneratorConfig
)
from rm_gallery.core.generator.iterative_rubric.query_rubric_generator import (
    POINTWISE_EVALUATION_TEMPLATE
)
from rm_gallery.core.models import OpenAIChatModel

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="my_quality_grader",
    model=OpenAIChatModel(model="qwen3-32b", api_key="your-key"),
    custom_evaluation_prompt=POINTWISE_EVALUATION_TEMPLATE,
    min_score=1,
    max_score=5,
    query_specific_generate_number=2,  # Generate 2 rubrics per sample
    enable_categorization=True,        # Group similar rubrics
    categories_number=3                # Target 3 categories
)
```

**Listwise Configuration:**

```python
from rm_gallery.core.generator.iterative_rubric import (
    IterativeRubricsGenerator,
    IterativeListwiseRubricsGeneratorConfig
)
from rm_gallery.core.generator.iterative_rubric.query_rubric_generator import (
    LISTWISE_EVALUATION_TEMPLATE
)

config = IterativeListwiseRubricsGeneratorConfig(
    grader_name="my_ranking_grader",
    model=OpenAIChatModel(model="qwen3-32b", api_key="your-key"),
    custom_evaluation_prompt=LISTWISE_EVALUATION_TEMPLATE,
    query_specific_generate_number=2,
    enable_categorization=False
)
```

### Generate the Grader

```python
generator = IterativeRubricsGenerator(config)
grader = await generator.generate(dataset)
```

### Use the Grader

**Pointwise Evaluation:**

```python
result = await grader.aevaluate(
    query="What causes seasons?",
    answer="Seasons are caused by Earth's axial tilt..."
)
print(f"Score: {result.score}")   # e.g., 4.0
print(f"Reason: {result.reason}") # Explanation based on learned rubrics
```

**Listwise Evaluation:**

```python
test_responses = [
    "Use print statements everywhere",
    "Use a debugger with breakpoints",
    "Just guess and check"
]
answer = "\n".join([f"Answer {i+1}: {resp}" for i, resp in enumerate(test_responses)])

result = await grader.aevaluate(
    query="Best way to debug code?",
    answer=answer,
    num_responses=len(test_responses)
)
print(f"Ranking: {result.ranking}")  # e.g., [2, 1, 3]
```

---

## Configuration Reference

### Core Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `grader_name` | `str` | required | Name for the generated grader |
| `model` | `BaseChatModel` | required | LLM to use for generation and evaluation |
| `language` | `LanguageEnum` | `EN` | Language for prompts (`EN` or `ZH`) |
| `enable_categorization` | `bool` | `False` | Group similar rubrics into categories |
| `categories_number` | `int` | `5` | Target number of categories |
| `query_specific_generate_number` | `int` | `1` | Rubrics to generate per training sample |

### Pointwise-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_score` | `int` | `0` | Minimum score value |
| `max_score` | `int` | `1` | Maximum score value |

### Evaluation Prompt

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `custom_evaluation_prompt` | `PromptTemplate` | `None` | Evaluation prompt template |

> **Tip:** Use built-in templates from `rm_gallery.core.generator.iterative_rubric.query_rubric_generator`:
>
> - `POINTWISE_EVALUATION_TEMPLATE` — for scoring
> - `LISTWISE_EVALUATION_TEMPLATE` — for ranking

---

## Advanced Usage

### Custom Evaluation Prompt

```python
from rm_gallery.core.models.schema.prompt_template import PromptTemplate

custom_prompt = PromptTemplate(
    template="""
Evaluate this response based on these criteria:
{rubrics}

Query: {query}
Response: {answer}

Score (1-5):
Reason:
    """,
    input_keys=["rubrics", "query", "answer"]
)

config = IterativePointwiseRubricsGeneratorConfig(
    grader_name="custom_grader",
    model=model,
    custom_evaluation_prompt=custom_prompt
)
```

---

## Best Practices

### Data Requirements

| Size | Recommendation |
|------|----------------|
| Minimum | 10-50 examples (for prototyping) |
| Recommended | 50-100 examples (all rubrics preserved) |
| Optimal | 100+ examples (MCR² auto-selection kicks in) |

### Data Quality

Data quality matters more than quantity.

**Good practices:**

- Clear preference signals (good vs. bad is obvious)
- Diverse query types covering your use case
- Consistent labeling standards

**Avoid:**

- Ambiguous cases where labels are debatable
- Noisy or contradictory labels

### Parameter Tuning

| Goal | Recommended Settings |
|------|---------------------|
| Fast prototyping | `query_specific_generate_number=1`, `enable_categorization=False` |
| Small dataset (≤100) | `query_specific_generate_number=2-3`, `enable_categorization=False` |
| Large dataset (>100) | `query_specific_generate_number=1`, `enable_categorization=True` |
| High quality | `query_specific_generate_number=3`, `enable_categorization=True` |