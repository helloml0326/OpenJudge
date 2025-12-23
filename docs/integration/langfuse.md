# Langfuse Integration Guide

## Overview

Langfuse is an open-source LLM application monitoring and evaluation platform that adopts a "backfill-based" integration model. RM-Gallery can be easily integrated with Langfuse, feeding results back to the Langfuse platform after evaluation.

## Integration Principles

Langfuse does not deeply integrate into the runtime flow, but instead provides flexible APIs that allow external evaluation systems to score existing Traces at any time. This loosely coupled design allows RM-Gallery to operate as an independent evaluation service.

## Quick Start: Integrating with Individual Graders

### 1. Install Dependencies

To begin integrating RM-Gallery with Langfuse, first install the required dependencies:

```bash
pip install langfuse
```

### 2. Configure Langfuse Client

Initialize the Langfuse client with your credentials. You'll need to set the appropriate environment variables for authentication:

```python
from langfuse import Langfuse
import os

# Get configuration from environment variables
# Make sure to set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and optionally LANGFUSE_HOST
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
)
```

### 3. Create RM-Gallery Evaluation Worker

Create a worker that fetches trace data from Langfuse, evaluates it using RM-Gallery graders, and sends the results back to Langfuse. This approach processes traces one by one, which is suitable for simple use cases.

```python
import asyncio
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader
from rm_gallery.core.graders.schema import GraderResult, GraderScore, GraderRank, GraderError

async def rm_gallery_eval_worker():
    """
    Worker function that evaluates Langfuse traces using RM-Gallery graders.

    This function demonstrates the basic pattern of fetching traces,
    evaluating them with individual graders, and sending results back to Langfuse.
    """
    # Initialize graders
    # These are individual RM-Gallery graders that will be used for evaluation
    similarity_grader = SimilarityGrader(algorithm="cosine")
    relevance_grader = RelevanceGrader()

    # Simulate fetching pending trace data
    # In a real implementation, you would fetch actual traces from Langfuse
    traces = fetch_pending_traces()  # This is a sample function that needs to be implemented based on actual requirements

    # Process each trace individually
    for trace in traces:
        # Extract trace information
        trace_id = trace["id"]
        input_data = trace["input"]
        output_data = trace["output"]
        expected_data = trace.get("expected")

        try:
            # Execute RM-Gallery evaluation with individual graders
            # Each grader is called separately, which is simple but not optimal for performance
            sim_result: GraderResult = await similarity_grader.aevaluate(
                prediction=output_data,
                reference=expected_data
            )

            rel_result: GraderResult = await relevance_grader.aevaluate(
                prediction=output_data,
                input=input_data
            )

            # Convert RM-Gallery results to Langfuse scores
            # Each result type needs to be handled differently
            if isinstance(sim_result, GraderScore):
                langfuse.score(
                    trace_id=trace_id,
                    name="cosine_similarity",
                    value=sim_result.score,
                    comment=getattr(sim_result, 'reason', '')
                )
            elif isinstance(sim_result, GraderError):
                langfuse.score(
                    trace_id=trace_id,
                    name="cosine_similarity",
                    value=0.0,
                    comment=f"Error: {sim_result.error}"
                )

            if isinstance(rel_result, GraderScore):
                langfuse.score(
                    trace_id=trace_id,
                    name="relevance",
                    value=rel_result.score,
                    comment=getattr(rel_result, 'reason', '')
                )
            elif isinstance(rel_result, GraderError):
                langfuse.score(
                    trace_id=trace_id,
                    name="relevance",
                    value=0.0,
                    comment=f"Error: {rel_result.error}"
                )

            print(f"Successfully evaluated trace {trace_id}")

        except Exception as e:
            # Handle any errors during evaluation
            print(f"Error evaluating trace {trace_id}: {str(e)}")
            continue

# Run evaluation worker
# This executes the evaluation process
if __name__ == "__main__":
    asyncio.run(rm_gallery_eval_worker())
```

## Advanced Usage: Integrating with GradingRunner

### Batch Evaluation with GradingRunner

For improved performance and resource efficiency, especially when dealing with many traces or graders, use RM-Gallery's GradingRunner. This approach processes multiple traces in batches, leveraging concurrent execution.

```python
import asyncio
import pandas as pd
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader
from rm_gallery.core.graders.schema import GraderResult, GraderScore, GraderRank, GraderError
from rm_gallery.core.runner.grading_runner import GradingRunner

async def rm_gallery_batch_eval_worker():
    """
    Worker function that performs batch evaluation using GradingRunner.

    This approach is more efficient than individual evaluation because:
    1. It processes multiple traces in batches
    2. It leverages concurrent execution of graders
    3. It reduces the number of API calls to Langfuse
    """
    # Initialize graders with runner for batch processing
    # The GradingRunner manages concurrent execution and batching
    runner = GradingRunner(
        grader_configs={
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=10  # Control how many graders run concurrently
    )

    # Simulate fetching pending trace data
    # In a real implementation, you would fetch actual traces from Langfuse
    traces = fetch_pending_traces()  # This is a sample function that needs to be implemented based on actual requirements

    # Prepare evaluation data in batch format
    # This transforms the trace data into the format expected by GradingRunner
    evaluation_data = []
    trace_mapping = {}  # Map data indices to trace IDs for result tracking

    for i, trace in enumerate(traces):
        trace_id = trace["id"]
        input_data = trace["input"]
        output_data = trace["output"]
        expected_data = trace.get("expected")

        # Prepare data for RM-Gallery evaluation
        # Each item needs to match the expected input format of the graders
        eval_item = {
            "input": input_data,
            "prediction": output_data,
        }

        # Add reference if available
        if expected_data:
            eval_item["reference"] = expected_data

        evaluation_data.append(eval_item)
        trace_mapping[i] = trace_id

    # Check if there's any data to evaluate
    if not evaluation_data:
        print("No traces to evaluate")
        return

    try:
        # Execute RM-Gallery batch evaluation
        # This is where the performance benefit comes from - all data is processed together
        batch_results = await runner.arun(evaluation_data)

        # Process and send results to Langfuse
        # Iterate through results and send them back to Langfuse
        for grader_name, grader_results in batch_results.items():
            for i, result in enumerate(grader_results):
                trace_id = trace_mapping[i]

                # Handle different result types appropriately
                if isinstance(result, GraderScore):
                    langfuse.score(
                        trace_id=trace_id,
                        name=grader_name,
                        value=result.score,
                        comment=getattr(result, 'reason', '')
                    )
                elif isinstance(result, GraderRank):
                    langfuse.score(
                        trace_id=trace_id,
                        name=grader_name,
                        value=getattr(result, 'rank', 0),
                        comment=getattr(result, 'reason', '')
                    )
                elif isinstance(result, GraderError):
                    langfuse.score(
                        trace_id=trace_id,
                        name=grader_name,
                        value=0.0,
                        comment=f"Error: {result.error}"
                    )

        print(f"Successfully evaluated {len(traces)} traces with {len(batch_results)} graders")

    except Exception as e:
        # Handle any errors during batch evaluation
        print(f"Error in batch evaluation: {str(e)}")

# Run batch evaluation worker
# This executes the more efficient batch evaluation process
if __name__ == "__main__":
    asyncio.run(rm_gallery_batch_eval_worker())
```

### Working with Aggregated Results

RM-Gallery's GradingRunner also supports result aggregation, which allows you to compute composite scores from multiple individual metrics. This is useful for creating overall quality measures.

```python
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.analyzer.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader

async def rm_gallery_aggregated_eval_worker():
    """
    Worker function that performs evaluation with result aggregation.

    This demonstrates how to use RM-Gallery's built-in aggregation capabilities
    to compute composite scores from multiple metrics.
    """
    # Initialize runner with graders and aggregators
    # Aggregators automatically combine results according to specified rules
    runner = GradingRunner(
        grader_configs={
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=10,
        # Add aggregators for result combination
        # This computes a weighted sum of the individual metrics
        aggregators=[
            WeightedSumAggregator(weights={"similarity": 0.6, "relevance": 0.4})
        ]
    )

    # Simulate fetching pending trace data
    traces = fetch_pending_traces()

    # Prepare evaluation data in batch format
    evaluation_data = []
    trace_mapping = {}

    for i, trace in enumerate(traces):
        trace_id = trace["id"]
        input_data = trace["input"]
        output_data = trace["output"]
        expected_data = trace.get("expected")

        # Prepare data for RM-Gallery evaluation
        eval_item = {
            "input": input_data,
            "prediction": output_data,
        }

        if expected_data:
            eval_item["reference"] = expected_data

        evaluation_data.append(eval_item)
        trace_mapping[i] = trace_id

    # Check if there's any data to evaluate
    if not evaluation_data:
        print("No traces to evaluate")
        return

    try:
        # Execute RM-Gallery batch evaluation with aggregation
        # The runner automatically applies aggregators to the results
        batch_results = await runner.arun(evaluation_data)

        # Process individual grader results
        # These are the raw results from each grader
        for grader_name, grader_results in batch_results.items():
            for i, result in enumerate(grader_results):
                trace_id = trace_mapping[i]

                if isinstance(result, (GraderScore, GraderRank)):
                    langfuse.score(
                        trace_id=trace_id,
                        name=grader_name,
                        value=getattr(result, 'score', getattr(result, 'rank', 0)),
                        comment=getattr(result, 'reason', '')
                    )
                elif isinstance(result, GraderError):
                    langfuse.score(
                        trace_id=trace_id,
                        name=grader_name,
                        value=0.0,
                        comment=f"Error: {result.error}"
                    )

        # Process aggregated results
        # These are the composite scores computed by the aggregators
        for key in batch_results.keys():
            if key.startswith("aggregated_"):
                # Extract the aggregator name from the key
                agg_name = key[len("aggregated_"):]
                for i, result in enumerate(batch_results[key]):
                    trace_id = trace_mapping[i]

                    if isinstance(result, (GraderScore, GraderRank)):
                        langfuse.score(
                            trace_id=trace_id,
                            name=agg_name,
                            value=getattr(result, 'score', getattr(result, 'rank', 0)),
                            comment=getattr(result, 'reason', '')
                        )
                    elif isinstance(result, GraderError):
                        langfuse.score(
                            trace_id=trace_id,
                            name=agg_name,
                            value=0.0,
                            comment=f"Aggregation error: {result.error}"
                        )

        print(f"Successfully evaluated {len(traces)} traces with graders and aggregators")

    except Exception as e:
        # Handle any errors during aggregated batch evaluation
        print(f"Error in aggregated batch evaluation: {str(e)}")

# Run aggregated batch evaluation worker
# Uncomment to run the aggregated evaluation
# if __name__ == "__main__":
#     asyncio.run(rm_gallery_aggregated_eval_worker())
```

### Scheduled Evaluation Task with Batch Processing

For continuous monitoring, you can set up a scheduled task that periodically evaluates new traces. This approach is useful for ongoing evaluation of production systems.

```python
import asyncio
import time
from datetime import datetime
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader

def transform_trace_data(traces):
    """
    Transform trace data to RM-Gallery evaluation format.

    This function converts Langfuse trace data into the format expected by RM-Gallery graders.
    It's a reusable utility that can be used in different parts of your evaluation pipeline.

    Args:
        traces: List of trace objects from Langfuse

    Returns:
        tuple: (evaluation_data, trace_mapping)
    """
    evaluation_data = []
    trace_mapping = {}

    for i, trace in enumerate(traces):
        # Extract relevant fields from the trace
        input_text = trace.get("input", "")
        output_text = trace.get("output", "")
        expected_output = trace.get("expected", "")

        # Create evaluation item in the format expected by RM-Gallery
        eval_item = {
            "input": input_text,
            "prediction": output_text,
        }

        # Add reference data if available
        if expected_output:
            eval_item["reference"] = expected_output

        evaluation_data.append(eval_item)
        # Keep track of which trace corresponds to which data item
        trace_mapping[i] = trace["id"]

    return evaluation_data, trace_mapping

async def scheduled_eval_worker(interval_minutes=10):
    """
    Scheduled evaluation task with batch processing.

    This function runs continuously, periodically fetching new traces and evaluating them.
    It's suitable for ongoing monitoring of production systems.

    Args:
        interval_minutes: How often to run evaluations (default: 10 minutes)
    """

    # Initialize runner once for reuse
    # Creating the runner once and reusing it is more efficient
    runner = GradingRunner(
        grader_configs={
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=15  # Higher concurrency for scheduled tasks
    )

    # Main evaluation loop
    while True:
        print(f"[{datetime.now()}] Starting scheduled batch evaluation...")
        try:
            # Fetch traces that need evaluation
            traces = fetch_pending_traces()

            # If no traces, wait and try again
            if not traces:
                print(f"[{datetime.now()}] No traces to evaluate. Waiting for next cycle.")
                await asyncio.sleep(interval_minutes * 60)
                continue

            # Transform data to RM-Gallery format
            evaluation_data, trace_mapping = transform_trace_data(traces)

            # Execute batch evaluation
            # This is where the actual evaluation happens
            batch_results = await runner.arun(evaluation_data)

            # Send results to Langfuse
            # Process and upload all results
            processed_count = 0
            for grader_name, grader_results in batch_results.items():
                for i, grader_result in enumerate(grader_results):
                    trace_id = trace_mapping[i]

                    # Handle different result types
                    if isinstance(grader_result, (GraderScore, GraderRank)):
                        value = getattr(grader_result, 'score', getattr(grader_result, 'rank', 0))
                        langfuse.score(
                            trace_id=trace_id,
                            name=grader_name,
                            value=value,
                            comment=getattr(grader_result, 'reason', '')
                        )
                        processed_count += 1
                    elif isinstance(grader_result, GraderError):
                        langfuse.score(
                            trace_id=trace_id,
                            name=grader_name,
                            value=0.0,
                            comment=f"Error: {grader_result.error}"
                        )

            print(f"[{datetime.now()}] Batch evaluation completed. Processed {processed_count} evaluations.")

        except Exception as e:
            # Handle any errors during scheduled evaluation
            print(f"[{datetime.now()}] Error in scheduled batch evaluation: {str(e)}")

        # Wait for next execution
        await asyncio.sleep(interval_minutes * 60)

# Run scheduled task
# Uncomment to run the scheduled evaluation worker
# if __name__ == "__main__":
#     asyncio.run(scheduled_eval_worker())
```

## Tips

- :bulb: **Grader Error Handling**: Always check for `GraderError` results when using graders. These indicate evaluation failures that should be handled gracefully rather than accessing non-existent attributes.

- :bulb: **Grader Result Types**: Understand the different result types (`GraderScore`, `GraderRank`, `GraderError`) that graders can return and handle each appropriately in your integration code.

- :bulb: **Runner Concurrency**: Use the `max_concurrency` parameter in `GradingRunner` to control how many graders run simultaneously, preventing resource exhaustion.

- :bulb: **Batch Processing with Runner**: For evaluating multiple data points, use `GradingRunner.arun()` to process them in batches rather than calling graders individually for better performance.

- :bulb: **Runner Aggregators**: Take advantage of built-in aggregators in `GradingRunner` rather than implementing your own result aggregation logic.

- :bulb: **Async Grader Methods**: Always use the async methods (e.g., `aevaluate`) when working with graders to ensure proper asynchronous execution.

## Related Resources

- [RM-Gallery Runner Documentation](../../../running_graders/run-tasks.md)
- [Langfuse Official Documentation](https://langfuse.com/docs)