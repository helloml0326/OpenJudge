# Agent Graders

Evaluate AI agent behavior across actions, tools, memory, planning, reflection, and trajectories. These graders help you assess decision quality, detect failures, and optimize agent performance at every step.

## Overview

| Category | Grader | Purpose | Key Use Case |
|----------|--------|---------|--------------|
| **Action** | ActionAlignmentGrader | Evaluates action-plan consistency | ReAct agents, step-by-step reasoning |
| | ActionLoopDetectionGrader | Detects repetitive actions | Multi-step exploration tasks |
| **Tool** | ToolSelectionGrader | Assesses tool choice quality | Function calling agents |
| | ToolCallAccuracyGrader | Evaluates tool call accuracy | API-based assistants |
| | ToolCallSuccessGrader | Checks technical execution success | Production agent monitoring |
| | ToolParameterCheckGrader | Validates parameter correctness | Slot-filling dialogues |
| | ToolCallSequenceMatchGrader | Compares tool call sequences | Benchmark evaluation |
| **Memory** | MemoryAccuracyGrader | Validates memory factuality | Memory-augmented agents |
| | MemoryDetailPreservationGrader | Checks detail retention | Long-horizon tasks |
| | MemoryRetrievalEffectivenessGrader | Assesses memory retrieval | RAG-based agents |
| **Plan** | PlanFeasibilityGrader | Evaluates plan feasibility | Task planning agents |
| **Reflection** | ReflectionAccuracyGrader | Validates reflection accuracy | Self-correcting agents |
| | ReflectionOutcomeUnderstandingGrader | Checks outcome understanding | Error recovery scenarios |
| | ReflectionProgressAwarenessGrader | Assesses progress awareness | Goal-tracking agents |
| **Observation** | ObservationInformationGainGrader | Measures information gain | Exploration efficiency |
| **Trajectory** | TrajectoryComprehensiveGrader | Comprehensive trajectory evaluation | End-to-end agent testing |

## Performance

Benchmark results using qwen3-max on agent evaluation tasks:

| Grader | Samples | Preference Accuracy | Source |
|--------|---------|---------------------|--------|
| ActionAlignmentGrader | 14 | 79% | ALFWorld, WebShop, GAIA |
| ToolCallAccuracyGrader | 80 | 88% | API-Bank |
| ToolCallSuccessGrader | 40 | 98% | API-Bank |
| ToolParameterCheckGrader | 40 | 75% | API-Bank |
| ToolSelectionGrader | 40 | 73% | API-Bank |
| MemoryAccuracyGrader | 24 | 75% | ALFWorld, WebShop, GAIA |
| MemoryDetailPreservationGrader | 40 | 73% | ALFWorld, WebShop, GAIA |
| MemoryRetrievalEffectivenessGrader | 4 | 100% | ALFWorld |
| PlanFeasibilityGrader | 14 | 64% | ALFWorld, GAIA |
| ReflectionAccuracyGrader | 2 | 100% | ALFWorld |
| ReflectionOutcomeUnderstandingGrader | 24 | 76% | ALFWorld, GAIA |
| ReflectionProgressAwarenessGrader | 40 | 70% | ALFWorld, WebShop, GAIA |

> **Note:** Preference Accuracy measures alignment with human-annotated preference labels (positive and negative samples) on agent evaluation tasks. Higher is better.


---

## Action Graders

### ActionAlignmentGrader

Evaluates whether agent actions align with stated plans or reasoning.

**Use this grader to:**

- Verify consistency between planning and execution
- Debug agent decision-making processes
- Ensure actions follow stated intentions

**Evaluation criteria:** Direct plan implementation, correct object targeting, goal contribution, logical sequence, and constraint respect.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning statement |
| `action` | str | Yes | Agent's executed action |
| `history` | List[dict] | No | Previous step dictionaries for context |
| `context` | str | No | Task context (description, environment, available actions) |

**Scoring:**
- `1.0`: Good alignment - action follows plan logically
- `0.0`: Poor alignment - action inconsistent with plan

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ActionAlignmentGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ActionAlignmentGrader(model=model)

    result = await grader.aevaluate(
        plan="I will open drawer 1 to find the key.",
        action="open drawer 1",
        context="Task: Find the key to unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - good alignment
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ActionLoopDetectionGrader

Detects repetitive or similar actions in agent sequences.

**Use this grader to:**

- Identify when agents get stuck in loops
- Detect inefficient exploration strategies
- Debug stuck agents in multi-step tasks

**Evaluation method:** Compares all pairs of action signatures for similarity.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Message list containing agent interactions |
| `similarity_threshold` | float | No | Threshold to consider actions similar (default: 1.0) |

**Scoring:**
- `1.0`: No loops detected
- `0.0`: Many similar action pairs found
- Score computed as: `1.0 - (similar_pairs / total_pairs)`

**Example:**

```python
import asyncio
from rm_gallery.core.graders.agent import ActionLoopDetectionGrader

async def main():
    grader = ActionLoopDetectionGrader(similarity_threshold=1.0)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "search", "arguments": '{"query": "python"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Results..."},
        {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "search", "arguments": '{"query": "python"}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "Results..."},
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Score: {result.score}")   # Lower score indicates loop
    print(f"Similar pairs: {result.metadata['similar_pair_count']}")

asyncio.run(main())
```

---

## Tool Graders

### ToolSelectionGrader

Evaluates tool selection quality for addressing user queries.

**Use this grader to:**

- Assess tool choice appropriateness
- Evaluate agent decision-making quality
- Compare different agent architectures

**Evaluation criteria:** Tool relevance, selection completeness, efficiency, and understanding of tool capabilities.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | User query or conversation history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Available tool definitions |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tools actually selected by agent |

**Scoring:**
- `5`: Optimal tool selection - most direct and efficient
- `4`: Reasonable selection - can complete task but not optimal
- `3`: Acceptable - related but not direct match
- `2`: Poor - clearly mismatched with task
- `1`: Incorrect - no tool selected or completely irrelevant

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ToolSelectionGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolSelectionGrader(model=model)

    result = await grader.aevaluate(
        query="Find all Python files modified in the last week",
        tool_definitions=[
            {"name": "search_files", "description": "Search for files by pattern"},
            {"name": "git_log", "description": "Get git commit history"},
            {"name": "read_file", "description": "Read file contents"}
        ],
        tool_calls=[
            {"name": "search_files", "arguments": {"pattern": "*.py"}},
            {"name": "git_log", "arguments": {"days": 7}}
        ]
    )

    print(f"Score: {result.score}")   # 4-5 - good tool selection
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ToolCallAccuracyGrader

Evaluates tool call accuracy including parameter correctness and query relevance.

**Use this grader to:**

- Validate tool call correctness
- Assess parameter extraction accuracy
- Evaluate agent tool-use capability

**Evaluation criteria:** Tool relevance to query and parameter correctness according to definitions.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | Query or chat history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions with parameters |
| `tool_calls` | List[Dict[str, Any]] | No | Tool calls to evaluate (or provide `response`) |
| `response` | str or List[Dict] | No | Response containing tool calls |

**Scoring:**
- `5`: Fully relevant, all parameters correct
- `4`: Relevant, tools returned errors but agent retried successfully
- `3`: Relevant but unnecessary/excessive calls
- `2`: Partially relevant, insufficient tools or incorrect parameters
- `1`: Irrelevant or tool names not found in definitions

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ToolCallAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolCallAccuracyGrader(model=model)

    conversation = [
        {"role": "user", "content": "What's the weather like in New York?"}
    ]

    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information for a location",
            "parameters": {"location": "City name"}
        }
    ]

    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
    ]

    result = await grader.aevaluate(
        query=conversation,
        tool_definitions=tool_definitions,
        tool_calls=tool_calls
    )

    print(f"Score: {result.score}")   # 5.0 - accurate tool call
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ToolCallSuccessGrader

Evaluates technical execution success of tool calls (no errors, exceptions, or timeouts).

**Use this grader to:**

- Detect technical failures in tool execution
- Monitor agent reliability
- Debug tool integration issues

> **Note:** This grader only checks technical success, not business correctness.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions for context |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tool calls to evaluate (name and arguments) |
| `tool_responses` | str or List[str] | Yes | Tool responses corresponding to each tool call |

**Scoring:**
- `1.0`: All tool calls successful
- `0.0`: At least one tool call failed

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ToolCallSuccessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolCallSuccessGrader(model=model)

    tool_definitions = [
        {
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {"location": "City name"}
        }
    ]

    tool_calls = [
        {
            "name": "get_weather",
            "arguments": {"location": "New York"}
        }
    ]

    tool_responses = [
        "The weather in New York is sunny and 25 degrees Celsius."
    ]

    result = await grader.aevaluate(
        tool_definitions=tool_definitions,
        tool_calls=tool_calls,
        tool_responses=tool_responses
    )

    print(f"Score: {result.score}")   # 1.0 - successful
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ToolParameterCheckGrader

Evaluates parameter extraction accuracy from user queries.

**Use this grader to:**

- Validate parameter extraction accuracy
- Ensure grounded parameter values
- Detect hallucinated parameters

**Evaluation criteria:** Parameter completeness, accuracy, grounding, and correct mapping.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | str or List[Dict] | Yes | User query or conversation history |
| `tool_definitions` | List[Dict[str, Any]] | Yes | Tool definitions with parameter specifications |
| `tool_calls` | List[Dict[str, Any]] | Yes | Tool calls made by the agent |

**Scoring:**
- `1.0`: All parameters correct and complete
- `0.0`: Parameters have issues (missing, incorrect, or fabricated)

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ToolParameterCheckGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ToolParameterCheckGrader(model=model)

    result = await grader.aevaluate(
        query="Search for Python files in the src directory",
        tool_definitions=[
            {
                "name": "search_files",
                "parameters": {"pattern": "str", "directory": "str"}
            }
        ],
        tool_calls=[
            {
                "name": "search_files",
                "arguments": {"pattern": "*.py", "directory": "src"}
            }
        ]
    )

    print(f"Score: {result.score}")   # 1.0 - correct parameters
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ToolCallSequenceMatchGrader

Compares agent tool call sequences against reference sequences.

**Use this grader for:**

- Benchmark evaluation against ground truth
- Trajectory comparison and validation
- A/B testing different agent implementations

**Matching modes:** Strict (name + parameters) or loose (name only).

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Agent's message history with tool calls |
| `reference_tool_calls` | List[List[Dict[str, Any]]] | Yes | Ground truth reference tool sequence by steps |
| `strict_mode` | bool | No | Match name + parameters (True) or name only (False), default: True |
| `use_jaccard_similarity` | bool | No | Use Jaccard similarity ignoring order (True) or step-by-step (False), default: True |

**Scoring:**
- **Strict mode with Jaccard**: Intersection over union of (tool_name, parameters) pairs
- **Loose mode with Jaccard**: Intersection over union of tool names
- **Step-by-step mode**: Average F1 score across steps
- Range: 0.0 (no match) to 1.0 (perfect match)

**Example:**

```python
import asyncio
from rm_gallery.core.graders.agent import ToolCallSequenceMatchGrader

async def main():
    grader = ToolCallSequenceMatchGrader(
        strict_mode=True,
        use_jaccard_similarity=True
    )

    messages = [
        {"role": "assistant", "tool_calls": [
            {"id": "1", "function": {"name": "search", "arguments": '{"query": "python"}'}}
        ]},
        {"role": "tool", "tool_call_id": "1", "content": "Results..."},
    ]

    reference_tool_calls = [
        [
            {"name": "search", "arguments": {"query": "python"}}
        ]
    ]

    result = await grader.aevaluate(
        messages=messages,
        reference_tool_calls=reference_tool_calls
    )

    print(f"Score: {result.score}")   # 1.0 - perfect match
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

---

## Memory Graders

### MemoryAccuracyGrader

Evaluates accuracy and factuality of stored memory content.

**Use this grader to:**

- Validate memory system correctness
- Ensure grounded information storage
- Debug hallucination in memory

**Evaluation criteria:** Memory reflects actual observations, stores only factual details, and maintains accurate associations.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous step dictionaries |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Accurate and factual memory
- `0.0`: Inaccurate or fabricated memory

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import MemoryAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet with three drawers.",
        memory="The cabinet is closed and has three drawers.",
        context="Task: Inventory room objects"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### MemoryDetailPreservationGrader

Evaluates preservation of important details in stored memory.

**Use this grader to:**

- Validate detail retention
- Ensure actionable memory content
- Debug information loss

**Evaluation criteria:** Storage of specific details, exact locations, numerical values, and important constraints.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous step dictionaries |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Important details preserved
- `0.0`: Important details lost or generalized

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import MemoryDetailPreservationGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryDetailPreservationGrader(model=model)

    result = await grader.aevaluate(
        observation="Cabinet 1 at coordinates (3.5, 2.1) contains 5 red apples.",
        memory="Cabinet 1 at (3.5, 2.1) has 5 red apples.",
        context="Task: Inventory items with precise locations"
    )

    print(f"Score: {result.score}")   # 1.0 - details preserved
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### MemoryRetrievalEffectivenessGrader

Evaluates effectiveness of memory retrieval for planning and decision-making.

**Use this grader to:**

- Assess memory system effectiveness
- Detect failure to use available information
- Identify repetitive behavior due to poor retrieval

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning |
| `observation` | str | Yes | Current environment observation |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Effective memory retrieval
- `0.0`: Ineffective retrieval or failure to use memory

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import MemoryRetrievalEffectivenessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = MemoryRetrievalEffectivenessGrader(model=model)

    result = await grader.aevaluate(
        plan="I will use the key from drawer 1 to unlock the door.",
        observation="You are standing in the room with a locked door.",
        memory="The key was found in drawer 1 in step 3.",
        context="Task: Unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - effective retrieval
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

---

## Plan Graders

### PlanFeasibilityGrader

Evaluates logical soundness and feasibility of agent plans.

**Use this grader to:**

- Validate agent planning capability
- Ensure logical action sequences
- Debug infeasible plans

**Evaluation criteria:** Causal logic, action order feasibility, executability, and prerequisite awareness.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `plan` | str | Yes | Agent's planning/reasoning |
| `observation` | str | Yes | Current environment observation |
| `memory` | str | Yes | Agent's memory content |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Feasible and logically sound
- `0.0`: Infeasible or illogical

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import PlanFeasibilityGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = PlanFeasibilityGrader(model=model)

    result = await grader.aevaluate(
        plan="I will first open the drawer to get the key, then use it to unlock the door.",
        observation="The drawer is closed. You don't have any items.",
        memory="The key is inside the drawer.",
        context="Task: Unlock the door"
    )

    print(f"Score: {result.score}")   # 1.0 - feasible
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

---

## Reflection Graders

### ReflectionAccuracyGrader

Evaluates accuracy of agent reflections based on actual observations.

**Use this grader to:**

- Validate agent self-assessment accuracy
- Ensure grounded reflections
- Debug hallucination in reasoning

**Evaluation criteria:** Reflections only mention observed objects, states, and details.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Accurate and grounded reflection
- `0.0`: Contains fabrications

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ReflectionAccuracyGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionAccuracyGrader(model=model)

    result = await grader.aevaluate(
        observation="You see a closed cabinet.",
        reflection="I observed a closed cabinet.",
        context="Task: Find objects in the room"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ReflectionOutcomeUnderstandingGrader

Evaluates correctness of action outcome interpretation in reflections.

**Use this grader to:**

- Validate outcome interpretation accuracy
- Detect fabricated or distorted understanding
- Ensure evidence-based reasoning

**Validation:** Strict factual accuracy checking.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context |

**Scoring:**
- `1.0`: Correct understanding - reflection accurately mirrors observation
- `0.0`: Poor understanding - factual distortion, failure misinterpretation, premature conclusions, scope overreach, inference leaps, fabrication, or format misinterpretation

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ReflectionOutcomeUnderstandingGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionOutcomeUnderstandingGrader(model=model)

    result = await grader.aevaluate(
        observation="The drawer is now open. You see a key inside.",
        reflection="I successfully opened the drawer and found a key inside.",
        context="Task: Find the key"
    )

    print(f"Score: {result.score}")   # 1.0 - correct understanding
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

### ReflectionProgressAwarenessGrader

Evaluates accuracy of task progress awareness and sub-goal recognition.

**Use this grader to:**

- Assess task progress tracking
- Detect loop/stuck situations
- Validate sub-goal awareness

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `observation` | str | Yes | Agent's observation from environment |
| `reflection` | str | Yes | Agent's reflection on the situation |
| `history` | List[dict] | No | Previous steps |
| `context` | str | No | Task context (critical for sub-goal tracking) |

**Scoring:**
- `1.0`: Accurate awareness - correctly identifies accomplishments, assesses distance to goal, recognizes all sub-goals
- `0.0`: Inaccurate awareness - misjudges progress, overlooks sub-goals, claims "almost done" while major requirements unmet

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import ReflectionProgressAwarenessGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = ReflectionProgressAwarenessGrader(model=model)

    result = await grader.aevaluate(
        observation="You have collected 3 out of 5 required items.",
        reflection="Good progress! I'm about halfway through the task. Still need to find 2 more items.",
        context="Task: Collect 5 specific items from different locations"
    )

    print(f"Score: {result.score}")   # 1.0 - accurate awareness
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

---

## Observation Graders

### ObservationInformationGainGrader

Measures information gain and redundancy in observation sequences.

**Use this grader to:**

- Evaluate exploration efficiency
- Detect redundant information gathering
- Assess agent curiosity/exploration strategy

**Scoring:** Rewards novel observations, penalizes redundant ones.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Message list containing agent interactions |
| `similarity_threshold` | float | No | Redundancy threshold (default: 0.5) |

**Scoring:**
- `1.0`: High information gain, low redundancy
- `0.0`: High redundancy, low information gain
- Score based on average per-observation novelty with exponential penalty for similarity

**Example:**

```python
import asyncio
from rm_gallery.core.graders.agent import ObservationInformationGainGrader

async def main():
    grader = ObservationInformationGainGrader(similarity_threshold=0.5)

    messages = [
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "look", "arguments": '{}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "You see a red box."},
        {"role": "assistant", "tool_calls": [{"id": "2", "function": {"name": "look", "arguments": '{}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "You see a blue sphere."},
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Score: {result.score}")   # Higher = more novel observations
    print(f"Each turn similarity: {result.metadata['each_turn_similarity']}")

asyncio.run(main())
```

---

## Trajectory Graders

### TrajectoryComprehensiveGrader

Comprehensive evaluation of complete agent trajectories.

**Use this grader for:**

- End-to-end agent evaluation
- Holistic trajectory assessment
- Agent benchmark evaluation
- Production quality monitoring

**Evaluation dimensions:** Step contribution, relevance, accuracy, and efficiency.

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `messages` | List[Dict[str, Any]] | Yes | Complete message history including system, user, assistant, tool |
| `resolution_threshold` | float | No | Threshold for success determination (default: 0.8) |

**Scoring:**

Each dimension uses 1-5 scale in prompts, normalized to 0-1:
- `5` → `1.0`: Excellent
- `4` → `0.75`: Good
- `3` → `0.5`: Acceptable
- `2` → `0.25`: Poor
- `1` → `0.0`: Very poor

Overall score is the average across all steps and dimensions.

**Example:**

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import TrajectoryComprehensiveGrader

async def main():
    model = OpenAIChatModel(model="qwen3-32b")
    grader = TrajectoryComprehensiveGrader(
        model=model,
        resolution_threshold=0.75
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Find Python files modified today"},
        {"role": "assistant", "content": "I'll search for Python files.",
         "tool_calls": [{"id": "1", "function": {"name": "search_files", "arguments": '{"pattern": "*.py"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "Found: main.py, utils.py"},
        {"role": "assistant", "content": "I'll check their modification dates.",
         "tool_calls": [{"id": "2", "function": {"name": "get_file_info", "arguments": '{"files": ["main.py", "utils.py"]}'}}]},
        {"role": "tool", "tool_call_id": "2", "content": "main.py: today, utils.py: yesterday"},
        {"role": "assistant", "content": "Found 1 file modified today: main.py"}
    ]

    result = await grader.aevaluate(messages=messages)

    print(f"Overall Score: {result.score}")   # 0.0-1.0
    print(f"Is Resolved: {result.metadata['is_resolved']}")
    print(f"Avg Contribution: {result.metadata['avg_contribution']}")
    print(f"Avg Relevance: {result.metadata['avg_relevance']}")
    print(f"Avg Accuracy: {result.metadata['avg_accuracy']}")
    print(f"Avg Efficiency: {result.metadata['avg_efficiency']}")

    # Per-step details
    for step in result.metadata['step_evaluations']:
        print(f"Step {step['step_index']}: {step['step_reason']}")

asyncio.run(main())
```

---

## Combining Multiple Graders

For comprehensive agent evaluation, combine multiple graders using `GradingRunner`:

```python
import asyncio
from rm_gallery.core.models import OpenAIChatModel
from rm_gallery.core.graders.agent import (
    ActionAlignmentGrader,
    MemoryAccuracyGrader,
    PlanFeasibilityGrader,
)
from rm_gallery.core.runner.grading_runner import GradingRunner, GraderConfig

async def main():
    model = OpenAIChatModel(model="qwen3-32b")

    grader_configs = {
        "action_alignment": GraderConfig(grader=ActionAlignmentGrader(model=model)),
        "memory_accuracy": GraderConfig(grader=MemoryAccuracyGrader(model=model)),
        "plan_feasibility": GraderConfig(grader=PlanFeasibilityGrader(model=model)),
    }

    runner = GradingRunner(grader_configs=grader_configs)

    dataset = [
        {
            "plan": "I will open the drawer to find the key",
            "action": "open drawer 1",
            "observation": "The drawer is closed",
            "memory": "The key is in drawer 1",
            "query": "Find the key"
        }
    ]

    results = await runner.arun(dataset)

    print(f"Action Alignment: {results['action_alignment'][0].score}")
    print(f"Memory Accuracy: {results['memory_accuracy'][0].score}")
    print(f"Plan Feasibility: {results['plan_feasibility'][0].score}")

asyncio.run(main())
```

---

## Summary

Agent graders provide comprehensive evaluation across all aspects of agent behavior—from individual actions and tool calls to memory management, planning, reflection, and complete trajectories.

**Key capabilities:**

- **Process-level debugging** — Identify specific failure points in tool selection, parameter extraction, or reasoning
- **Outcome-level assessment** — Measure overall task success and trajectory quality
- **Systematic improvement** — Combine multiple graders to diagnose where agents fail, why they fail, and how to improve them

Build complete evaluation pipelines by combining graders from different categories to match your agent architecture and debugging needs.
