# Evaluate AI Agents

Assess AI agent performance at three levels: **Final Response** (end results), **Single Step** (individual actions), and **Trajectory** (execution paths). This guide helps you identify failure points, optimize costs, and improve agent reliability.

> **Note:** For detailed grader documentation, see [Built-in Graders](./built_in_graders.md).

## Why Evaluate AI Agents?

AI agents operate autonomously through complex reasoning loops, making multiple tool calls and decisions before reaching a final answer. This multi-step nature creates unique evaluation challenges—a wrong tool selection early on can cascade into complete task failure.

**Systematic evaluation enables you to:**

- **Identify failure points** — Pinpoint issues in planning, tool selection, or execution
- **Optimize costs** — Reduce unnecessary tool calls and LLM iterations
- **Ensure reliability** — Validate performance before deployment
- **Continuously improve** — Drive enhancements through data-driven insights

## Three Evaluation Granularities

| Granularity | What It Measures | Example |
|-------------|------------------|---------|
| **Final Response** | Overall task success and answer quality | "Did the agent correctly answer the user's question?" |
| **Single Step** | Individual action quality (tool calls, planning, memory) | "Did the agent select the right tool for this sub-task?" |
| **Trajectory** | Multi-step reasoning paths and execution efficiency | "Did the agent take an efficient path without loops?" |

> **Tip:** Start with **Final Response** evaluation to establish baseline success rates. When failures occur, use **Single Step** evaluation to pinpoint root causes. Use **Trajectory** evaluation to detect systemic issues like loops or inefficiencies.

---

## Evaluate Final Response

Assess the end result of agent execution to determine if the agent successfully completed the user's task.

Uses graders from `rm_gallery.core.graders.common`: `CorrectnessGrader`, `RelevanceGrader`, `HallucinationGrader`, `HarmfulnessGrader`, `InstructionFollowingGrader`.

### Example: Evaluate Correctness

```python
import asyncio
from rm_gallery.core.graders.common import CorrectnessGrader
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.statistical import DistributionAnalyzer

# Initialize model and grader
model = OpenAIChatModel(
    model="qwen3-32b",
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
grader = CorrectnessGrader(model=model)

# Prepare dataset
dataset = [
    {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "reference_response": "Paris"
    },
    {
        "query": "What is 25 + 17?",
        "response": "25 + 17 equals 42.",
        "reference_response": "42"
    }
]

# Configure runner (no mapper needed if field names match)
runner = GradingRunner(
    grader_configs={
        "correctness": GraderConfig(grader=grader)
    },
    max_concurrency=16,
    show_progress=True
)

# Run evaluation
results = asyncio.run(runner.arun(dataset=dataset))

# Analyze results
grader_results = results["correctness"]
analyzer = DistributionAnalyzer()
analysis = analyzer.analyze(dataset=dataset, grader_results=grader_results)

print(f"=== Final Response Evaluation ===")
print(f"Mean score: {analysis.mean:.2f}/5")
print(f"Median score: {analysis.median:.2f}/5")

for i, result in enumerate(grader_results):
    print(f"\nQuery: {dataset[i]['query']}")
    print(f"  Score: {result.score}/5")
    print(f"  Reason: {result.reason}")
```

**Expected Output:**

```
=== Final Response Evaluation ===
Mean score: 5.00/5
Median score: 5.00/5

Query: What is the capital of France?
  Score: 5.0/5
  Reason: The response correctly states that the capital of France is Paris, which is factually consistent with the reference response 'Paris'. The added phrasing 'The capital of France is' provides appropriate context without contradicting or distorting the reference, and the core information matches exactly.

Query: What is 25 + 17?
  Score: 5.0/5
  Reason: The response correctly states that 25 + 17 equals 42, which is factually consistent with the reference response '42'. The additional wording ('equals') does not distort or contradict the reference and appropriately answers the query.
```

---

## Evaluate Single Step

Assess individual agent decisions in isolation—one tool call, one planning step, or one memory retrieval at a time.

Uses graders from `rm_gallery.core.graders.agent`: `ToolSelectionGrader`, `ToolCallSuccessGrader`, `PlanFeasibilityGrader`, `MemoryAccuracyGrader`, `ReflectionAccuracyGrader`, etc.

### Extract Grader Inputs from Trace Data

Single Step graders require specific fields extracted from your agent traces. Use the `mapper` parameter to transform trace data into grader inputs.

**Option 1: Dictionary Mapper**

Simple field renaming when your data structure differs from grader expectations.

```python
# Your data has different field names than what the grader expects
mapper = {
    "query": "user_input",           # grader expects "query", your data has "user_input"
    "tool_calls": "agent_actions",   # grader expects "tool_calls", your data has "agent_actions"
}
```

**Option 2: Callable Mapper**

Extract fields from complex structures like OpenAI messages format.

```python
def extract_tool_inputs(data: dict) -> dict:
    """Extract fields required by tool graders from OpenAI messages."""
    messages = data["messages"]

    # Extract user query (first user message)
    query = next(
        (m["content"] for m in messages if m["role"] == "user"),
        ""
    )

    # Extract tool calls from assistant messages
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                })

    # Extract tool responses
    tool_responses = [
        {"name": m.get("name"), "content": m["content"]}
        for m in messages if m.get("role") == "tool"
    ]

    return {
        "query": query,
        "tool_definitions": data.get("available_tools", []),
        "tool_calls": tool_calls,
        "tool_responses": tool_responses
    }
```

### Example: Evaluate Tool Selection

```python
import asyncio
from rm_gallery.core.graders.agent import ToolSelectionGrader
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.statistical import DistributionAnalyzer

# Initialize model and grader
model = OpenAIChatModel(
    model="qwen3-32b",
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
grader = ToolSelectionGrader(model=model)

# Prepare dataset (raw OpenAI messages format)
dataset = [
    {
        "trace_id": "trace_001",
        "messages": [
            {"role": "user", "content": "What's 15% tip on a $45 bill?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {"name": "calculator", "arguments": '{"expression": "45 * 0.15"}'}
                }]
            }
        ],
        "available_tools": [
            {"name": "calculator", "description": "Perform mathematical calculations"},
            {"name": "search_web", "description": "Search the web for information"}
        ]
    },
    {
        "trace_id": "trace_002",
        "messages": [
            {"role": "user", "content": "What's the capital of France?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {"name": "calculator", "arguments": '{"expression": "France"}'}  # Wrong tool!
                }]
            }
        ],
        "available_tools": [
            {"name": "calculator", "description": "Perform mathematical calculations"},
            {"name": "search_web", "description": "Search the web for information"}
        ]
    }
]

# Define mapper to extract grader inputs
def extract_tool_inputs(data: dict) -> dict:
    messages = data["messages"]
    query = next((m["content"] for m in messages if m["role"] == "user"), "")
    tool_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_calls.append({
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"]
                })
    return {
        "query": query,
        "tool_definitions": data["available_tools"],
        "tool_calls": tool_calls
    }

# Configure runner with mapper
runner = GradingRunner(
    grader_configs={
        "tool_selection": GraderConfig(
            grader=grader,
            mapper=extract_tool_inputs
        )
    },
    max_concurrency=16,
    show_progress=True
)

# Run evaluation
results = asyncio.run(runner.arun(dataset=dataset))

# Analyze results
grader_results = results["tool_selection"]
analyzer = DistributionAnalyzer()
analysis = analyzer.analyze(dataset=dataset, grader_results=grader_results)

print(f"=== Single Step Evaluation (Tool Selection) ===")
print(f"Mean score: {analysis.mean:.2f}/5")

for i, result in enumerate(grader_results):
    status = "Good" if result.score >= 4 else "Poor"
    print(f"\n{dataset[i]['trace_id']}: {status} (score: {result.score}/5)")
    print(f"  Reason: {result.reason}")
```

**Expected Output:**

```
=== Single Step Evaluation (Tool Selection) ===
Mean score: 3.00/5

trace_001: Good (score: 5.0/5)
  Reason: The agent selected the 'calculator' tool with the expression '45 * 0.15', which is the most direct, efficient, and semantically relevant tool for computing a percentage-based tip. The query is purely mathematical, requiring no external information or web search. The calculator tool is fully capable of performing this arithmetic operation accurately and instantly. No other tools are necessary — selecting 'search_web' would be irrelevant and inefficient. The agent demonstrates clear understanding of both the task intent (calculating a tip) and the tool's capability (performing mathematical calculations). All evaluation rubrics are satisfied: relevance, completeness, efficiency, capability, and understanding of tool scope.

trace_002: Poor (score: 1.0/5)
  Reason: The selected tool 'calculator' is completely irrelevant to the query 'What's the capital of France?'. The calculator tool is designed for mathematical calculations, not for retrieving factual geographic information. Passing 'France' as an expression to a calculator will result in an error or meaningless output. The appropriate tool available is 'search_web', which is explicitly designed to search for information like country capitals. No other tools were selected, and no attempt was made to use the semantically relevant tool. This selection demonstrates a fundamental misunderstanding of both the query intent and the tool capabilities.
```

---

## Evaluate Trajectory

Assess the entire sequence of agent actions to determine if the agent took an optimal path without loops or redundant steps.

Uses `TrajectoryComprehensiveGrader` from `rm_gallery.core.graders.agent`.

### Example: Evaluate Execution Path

```python
import asyncio
from rm_gallery.core.graders.agent import TrajectoryComprehensiveGrader
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig
from rm_gallery.core.analyzer.statistical import DistributionAnalyzer

# Initialize model and grader
model = OpenAIChatModel(
    model="qwen3-32b",
    api_key="your-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
grader = TrajectoryComprehensiveGrader(
    model=model,
    resolution_threshold=0.8  # Scores >= 0.8 are considered "resolved"
)

# Prepare dataset (full agent trajectories)
dataset = [
    {
        "trace_id": "efficient_trace",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What's the weather in Tokyo?"},
            {
                "role": "assistant",
                "content": "I'll check the weather for you.",
                "tool_calls": [{
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"location": "Tokyo"}'}
                }]
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "get_weather",
                "content": '{"temp": 22, "condition": "sunny"}'
            },
            {
                "role": "assistant",
                "content": "The weather in Tokyo is sunny with 22°C."
            }
        ]
    },
    {
        "trace_id": "inefficient_trace",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant with tools."},
            {"role": "user", "content": "What's 2 + 2?"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {"name": "search", "arguments": '{"q": "2+2"}'}  # Unnecessary tool
                }]
            },
            {"role": "tool", "content": "No results"},
            {
                "role": "assistant",
                "tool_calls": [{
                    "function": {"name": "search", "arguments": '{"q": "two plus two"}'}  # Retry
                }]
            },
            {"role": "tool", "content": "No results"},
            {"role": "assistant", "content": "2 + 2 = 4"}  # Could have answered directly
        ]
    }
]

# Configure runner (messages field maps directly)
runner = GradingRunner(
    grader_configs={
        "trajectory": GraderConfig(
            grader=grader,
            mapper={"messages": "messages"}
        )
    },
    max_concurrency=16,
    show_progress=True
)

# Run evaluation
results = asyncio.run(runner.arun(dataset=dataset))

# Analyze results
grader_results = results["trajectory"]
analyzer = DistributionAnalyzer()
analysis = analyzer.analyze(dataset=dataset, grader_results=grader_results)

print(f"=== Trajectory Evaluation ===")
print(f"Mean score: {analysis.mean:.2f}")
print(f"Median score: {analysis.median:.2f}")

for i, result in enumerate(grader_results):
    status = "Resolved" if result.score >= 0.8 else "Needs improvement"
    print(f"\n{dataset[i]['trace_id']}: {status} (score: {result.score:.2f})")
    print(f"  Reason: {result.reason[:150]}...")

    # Access step-level details from metadata
    if hasattr(result, 'metadata') and 'step_evaluations' in result.metadata:
        print(f"  Steps evaluated: {len(result.metadata['step_evaluations'])}")
        print(f"  Avg efficiency: {result.metadata.get('avg_efficiency', 'N/A')}")
```

**Expected Output:**

```
=== Trajectory Evaluation ===
Mean score: 0.53
Median score: 0.53

efficient_trace: Resolved (score: 1.00)
  Reason: Step 0: This step directly retrieves the weather information for Tokyo, which is exactly what the user requested. The tool call is precise, targeting ...
  Steps evaluated: 1
  Avg efficiency: 1.0

inefficient_trace: Needs improvement (score: 0.06)
  Reason: Step 0: The step attempts to search for '2+2' using a tool, but the tool returns no results. Since this is a basic arithmetic question that does not r...
  Steps evaluated: 2
  Avg efficiency: 0.0
```

---

## Choose the Right Granularity

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Production monitoring | Final Response | Track overall success rate |
| Debugging failures | Single Step | Isolate the problematic component |
| A/B testing | Final Response | Compare versions on outcomes |
| Cost optimization | Trajectory | Find redundant steps |
| Prompt engineering | Single Step | Test specific component improvements |
| Training reward models | Single Step + Trajectory | Need multi-level signals |

---

## Next Steps

- **[Built-in Graders](./built_in_graders.md)** — Detailed documentation for all available graders
- **[Generate Graders from Data](./generate%20graders%20from%20data.md)** — Automatically generate custom graders from your data
