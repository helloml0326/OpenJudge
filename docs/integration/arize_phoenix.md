# Arize Phoenix Integration Guide

## Overview

Arize Phoenix is an open-standard-based LLM observability platform that adopts a "standard-based" integration model. RM-Gallery can be integrated with Phoenix through the OpenInference specification, importing evaluation results as telemetry signals into the Phoenix platform.

## Integration Principles

Phoenix is based on the OpenInference specification, treating evaluation results as a set of ID-tagged DataFrames that align with the original Trace tree through ID mapping. This plug-in integration approach allows RM-Gallery to flexibly interface with Phoenix.

## Quick Start: Integrating with Individual Graders

### 1. Install Dependencies

To begin integrating RM-Gallery with Arize Phoenix, first install the required dependencies:

```bash
pip install phoenix-langchain pandas
```

### 2. Export Trace Data and Perform Evaluation

The first step is to export trace data from Phoenix and evaluate it using RM-Gallery graders. This example shows how to perform individual evaluations with separate graders.

```python
import phoenix as px
import pandas as pd
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader
from rm_gallery.core.graders.schema import GraderResult, GraderScore, GraderRank, GraderError
import asyncio

# Launch Phoenix (if not already started)
# This starts the Phoenix UI server
px.launch_app()

# Get trace data from Phoenix
# This retrieves the trace data as a pandas DataFrame
spans_df = px.Client().get_spans_dataframe()

# Initialize RM-Gallery graders
# These are individual graders that will be used for evaluation
similarity_grader = SimilarityGrader(algorithm="cosine")
relevance_grader = RelevanceGrader()

async def evaluate_spans(spans_df):
    """
    Evaluate spans using individual RM-Gallery graders.

    This function demonstrates the basic pattern of evaluating each span
    individually with separate graders. While simple, this approach may
    not be optimal for large datasets.

    Args:
        spans_df: DataFrame containing Phoenix trace data

    Returns:
        DataFrame with evaluation results
    """
    eval_results = []

    # Process each span individually
    for idx, row in spans_df.iterrows():
        try:
            # Extract input/output data from the span
            input_text = row.get('input.value', '')
            output_text = row.get('output.value', '')
            expected_output = row.get('expected_output', '')  # If there's expected output

            # Execute RM-Gallery evaluation with individual graders
            # Each grader is called separately
            sim_result: GraderResult = await similarity_grader.aevaluate(
                prediction=output_text,
                reference=expected_output
            )

            rel_result: GraderResult = await relevance_grader.aevaluate(
                prediction=output_text,
                input=input_text
            )

            # Convert results to dictionary format for DataFrame creation
            result_dict = {
                "span_id": str(idx),  # Use index as span_id
            }

            # Handle similarity result
            # Different result types need different handling
            if isinstance(sim_result, GraderScore):
                result_dict["cosine_similarity"] = sim_result.score
                result_dict["similarity_reasoning"] = getattr(sim_result, 'reason', '')
            elif isinstance(sim_result, GraderError):
                result_dict["cosine_similarity"] = None
                result_dict["similarity_error"] = sim_result.error

            # Handle relevance result
            if isinstance(rel_result, GraderScore):
                result_dict["relevance"] = rel_result.score
                result_dict["relevance_reasoning"] = getattr(rel_result, 'reason', '')
            elif isinstance(rel_result, GraderError):
                result_dict["relevance"] = None
                result_dict["relevance_error"] = rel_result.error

            eval_results.append(result_dict)

        except Exception as e:
            # Handle any errors during evaluation
            print(f"Error evaluating span {idx}: {str(e)}")
            eval_results.append({
                "span_id": str(idx),
                "cosine_similarity": None,
                "relevance": None,
                "error": str(e)
            })

    # Return results as a DataFrame
    return pd.DataFrame(eval_results)

# Execute evaluation
# This runs the evaluation process and prints the results
eval_df = asyncio.run(evaluate_spans(spans_df))
print("Evaluation completed. Results:")
print(eval_df.head())
```

### 3. Import Evaluation Results to Phoenix

After evaluating the spans, import the results back into Phoenix for visualization and analysis:

```python
from phoenix.trace import SpanEvaluations

# Create span evaluations for each metric
# Each metric needs to be converted to a SpanEvaluations object
if "cosine_similarity" in eval_df.columns:
    similarity_eval = SpanEvaluations(
        eval_df[["span_id", "cosine_similarity"]].rename(columns={"cosine_similarity": "score"}),
        name="Cosine Similarity"
    )

if "relevance" in eval_df.columns:
    relevance_eval = SpanEvaluations(
        eval_df[["span_id", "relevance"]].rename(columns={"relevance": "score"}),
        name="Relevance"
    )

# Log evaluations to Phoenix
# Collect all evaluation objects and log them together
evaluations_to_log = []
if 'similarity_eval' in locals():
    evaluations_to_log.append(similarity_eval)
if 'relevance_eval' in locals():
    evaluations_to_log.append(relevance_eval)

# Import evaluations to Phoenix
if evaluations_to_log:
    px.Client().log_evaluations(*evaluations_to_log)

print("Evaluations logged to Phoenix successfully!")
```

## Advanced Usage: Integrating with GradingRunner

### Batch Evaluation with GradingRunner

For improved performance and resource efficiency, especially with large datasets, use RM-Gallery's GradingRunner. This approach processes multiple spans in batches, leveraging concurrent execution.

```python
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.analyzer.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader
from rm_gallery.core.graders.schema import GraderResult

def prepare_phoenix_data_for_rm_gallery(spans_df):
    """
    Prepare Phoenix trace data for RM-Gallery evaluation.

    This function transforms Phoenix trace data into the format expected by RM-Gallery graders.
    It's a reusable utility that can be used in different parts of your evaluation pipeline.

    Args:
        spans_df: DataFrame containing Phoenix trace data

    Returns:
        tuple: (evaluation_data, span_id_mapping)
    """
    evaluation_data = []
    span_id_mapping = {}

    # Process each span to extract relevant data
    for idx, row in spans_df.iterrows():
        # Extract input/output data from Phoenix spans
        input_text = row.get('input.value', '')
        output_text = row.get('output.value', '')
        expected_output = row.get('expected_output', '')  # If there's expected output

        # Prepare evaluation item in RM-Gallery format
        eval_item = {
            "input": input_text,
            "prediction": output_text,
        }

        # Add reference data if available
        if expected_output:
            eval_item["reference"] = expected_output

        evaluation_data.append(eval_item)
        # Keep track of which span corresponds to which data item
        span_id_mapping[len(evaluation_data) - 1] = str(idx)  # Map to span ID

    return evaluation_data, span_id_mapping

async def advanced_phoenix_integration():
    """
    Advanced Phoenix integration using GradingRunner for batch processing.

    This function demonstrates how to use RM-Gallery's GradingRunner for efficient
    batch evaluation of Phoenix traces. It offers better performance than individual
    evaluation by processing multiple spans concurrently.
    """
    # Configure comprehensive evaluation runner
    # The GradingRunner manages concurrent execution and batching
    runner = GradingRunner(
        grader_configs={
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=20,  # Higher concurrency for better performance
    )

    # Get Phoenix trace data
    spans_df = px.Client().get_spans_dataframe()

    # Check if there's any data to evaluate
    if spans_df.empty:
        print("No spans to evaluate")
        return pd.DataFrame()

    # Prepare data for batch evaluation
    # This transforms the data into the format expected by GradingRunner
    evaluation_data, span_id_mapping = prepare_phoenix_data_for_rm_gallery(spans_df)

    try:
        # Execute comprehensive batch evaluation
        # This is where the performance benefit comes from - all data is processed together
        batch_results = await runner.arun(evaluation_data)

        # Process results into DataFrame format
        # Convert the batch results back to a format suitable for Phoenix
        eval_results = []

        # Iterate through the evaluation data to build results
        for i in range(len(evaluation_data)):
            span_id = span_id_mapping[i]
            result_dict = {"span_id": span_id}

            # Collect results from all graders for this span
            for grader_name, grader_results in batch_results.items():
                if i < len(grader_results):
                    result = grader_results[i]
                    # Handle different result types appropriately
                    if isinstance(result, (GraderScore, GraderRank)):
                        result_dict[grader_name] = getattr(result, 'score', getattr(result, 'rank', 0))
                        result_dict[f"{grader_name}_reasoning"] = getattr(result, 'reason', '')
                    elif isinstance(result, GraderError):
                        result_dict[grader_name] = None
                        result_dict[f"{grader_name}_error"] = result.error

            eval_results.append(result_dict)

        eval_df = pd.DataFrame(eval_results)

        # Create and import evaluation results
        # Convert results to Phoenix format and prepare for import
        evaluations_to_log = []
        for column in eval_df.columns:
            # Skip non-metric columns
            if column not in ["span_id", "error"] and not column.endswith("_reasoning") and not column.endswith("_error"):
                # Filter out rows with None values for this metric
                valid_rows = eval_df.dropna(subset=[column])
                if not valid_rows.empty:
                    # Create SpanEvaluations object for this metric
                    eval_obj = SpanEvaluations(
                        valid_rows[["span_id", column]].rename(columns={column: "score"}),
                        name=column.replace("_", " ").title()
                    )
                    evaluations_to_log.append(eval_obj)

        # Import to Phoenix
        # Send all evaluation results to Phoenix at once
        if evaluations_to_log:
            px.Client().log_evaluations(*evaluations_to_log)
            print(f"Successfully imported {len(evaluations_to_log)} evaluation metrics to Phoenix")

        return eval_df

    except Exception as e:
        # Handle any errors during advanced batch evaluation
        print(f"Error in advanced batch evaluation: {str(e)}")
        return pd.DataFrame()

# Execute advanced integration
# Uncomment to run the advanced evaluation
# eval_results_df = asyncio.run(advanced_phoenix_integration())
```

### Working with Aggregated Results

RM-Gallery's GradingRunner also supports result aggregation, which allows you to compute composite scores from multiple individual metrics. This is useful for creating overall quality measures.

```python
from rm_gallery.core.runner.grading_runner import GradingRunner
from rm_gallery.core.analyzer.aggregator.weighted_sum_aggregator import WeightedSumAggregator
from rm_gallery.core.graders.text.similarity import SimilarityGrader
from rm_gallery.core.graders.text.relevance_grader import RelevanceGrader

async def phoenix_aggregated_integration():
    """
    Phoenix integration with aggregated results.

    This function demonstrates how to use RM-Gallery's built-in aggregation capabilities
    to compute composite scores from multiple metrics, then import them to Phoenix.
    """
    # Configure runner with graders and aggregators
    # Aggregators automatically combine results according to specified rules
    runner = GradingRunner(
        grader_configs={
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=20,
        # Add aggregators for result combination
        # This computes a weighted sum of the individual metrics
        aggregators=[
            WeightedSumAggregator(weights={"similarity": 0.6, "relevance": 0.4})
        ]
    )

    # Get Phoenix trace data
    spans_df = px.Client().get_spans_dataframe()

    # Check if there's any data to evaluate
    if spans_df.empty:
        print("No spans to evaluate")
        return pd.DataFrame()

    # Prepare data for batch evaluation
    evaluation_data, span_id_mapping = prepare_phoenix_data_for_rm_gallery(spans_df)

    try:
        # Execute batch evaluation with aggregation
        # The runner automatically applies aggregators to the results
        batch_results = await runner.arun(evaluation_data)

        # Process results into DataFrame format
        # Convert both individual and aggregated results to DataFrame format
        eval_results = []

        # Iterate through the evaluation data to build results
        for i in range(len(evaluation_data)):
            span_id = span_id_mapping[i]
            result_dict = {"span_id": span_id}

            # Collect results from all graders for this span
            for grader_name, grader_results in batch_results.items():
                if i < len(grader_results):
                    result = grader_results[i]
                    # Handle different result types appropriately
                    if isinstance(result, (GraderScore, GraderRank)):
                        result_dict[grader_name] = getattr(result, 'score', getattr(result, 'rank', 0))
                        result_dict[f"{grader_name}_reasoning"] = getattr(result, 'reason', '')
                    elif isinstance(result, GraderError):
                        result_dict[grader_name] = None
                        result_dict[f"{grader_name}_error"] = result.error

            eval_results.append(result_dict)

        eval_df = pd.DataFrame(eval_results)

        # Create and import evaluation results (including aggregated ones)
        # Both individual and aggregated metrics are imported to Phoenix
        evaluations_to_log = []
        for column in eval_df.columns:
            # Skip non-metric columns
            if column not in ["span_id", "error"] and not column.endswith("_reasoning") and not column.endswith("_error"):
                # Filter out rows with None values for this metric
                valid_rows = eval_df.dropna(subset=[column])
                if not valid_rows.empty:
                    # Create SpanEvaluations object for this metric
                    eval_obj = SpanEvaluations(
                        valid_rows[["span_id", column]].rename(columns={column: "score"}),
                        name=column.replace("_", " ").title()
                    )
                    evaluations_to_log.append(eval_obj)

        # Import to Phoenix
        # Send all evaluation results to Phoenix at once
        if evaluations_to_log:
            px.Client().log_evaluations(*evaluations_to_log)
            print(f"Successfully imported {len(evaluations_to_log)} evaluation metrics to Phoenix")

        return eval_df

    except Exception as e:
        # Handle any errors during aggregated batch evaluation
        print(f"Error in aggregated batch evaluation: {str(e)}")
        return pd.DataFrame()

# Execute aggregated integration
# Uncomment to run the aggregated evaluation
# aggregated_eval_df = asyncio.run(phoenix_aggregated_integration())
```

### Custom Evaluation Metrics with Batch Processing

You can also create custom evaluation metrics and use them with the batch processing capabilities:

```python
from rm_gallery.core.graders.base_grader import BaseGrader
from rm_gallery.core.graders.schema import GraderInput, GraderResult, GraderScore, GraderError

class CustomPhoenixGrader(BaseGrader):
    """Custom Phoenix grader for specialized evaluation metrics."""

    def __init__(self):
        """
        Initialize the custom grader.

        This demonstrates how to create a custom evaluation metric for use with Phoenix.
        """
        super().__init__(
            name="custom_metric",
            description="Custom evaluation metric"
        )

    async def aevaluate(self, **kwargs) -> GraderResult:
        """
        Custom evaluation logic.

        This is where you implement your custom evaluation logic.
        This example implements a simple length-based evaluation.

        Args:
            **kwargs: Input data for evaluation

        Returns:
            Evaluation result
        """
        prediction = kwargs.get("prediction", "")
        reference = kwargs.get("reference", "")
        context = kwargs.get("context", "")

        # Implement custom evaluation logic
        # This is just an example, actual implementation can be customized based on requirements
        if reference and prediction:
            # Simple length match evaluation
            length_ratio = min(len(prediction), len(reference)) / max(len(prediction), len(reference))
            score = length_ratio if length_ratio > 0.5 else 0.0
        else:
            score = 0.0

        return GraderScore(
            name=self.name,
            score=score,
            reason=f"Length ratio: {length_ratio:.2f}" if 'length_ratio' in locals() else "Missing prediction or reference"
        )

# Use custom grader in batch processing
async def custom_evaluation_with_batch_processing(spans_df):
    """
    Custom evaluation with batch processing.

    This function demonstrates how to use a custom grader with the batch processing capabilities.
    It shows how to integrate specialized evaluation metrics into the Phoenix workflow.

    Args:
        spans_df: DataFrame containing Phoenix trace data

    Returns:
        DataFrame with custom evaluation results
    """
    # Create runner with custom grader
    # This combines the custom grader with standard graders
    custom_runner = GradingRunner(
        grader_configs={
            "custom_metric": CustomPhoenixGrader(),
            "similarity": SimilarityGrader(algorithm="cosine"),
            "relevance": RelevanceGrader(),
        },
        max_concurrency=15  # Adjust concurrency based on custom grader characteristics
    )

    # Check if there's any data to evaluate
    if spans_df.empty:
        print("No spans to evaluate")
        return pd.DataFrame()

    # Prepare data using the same transformation function
    evaluation_data, span_id_mapping = prepare_phoenix_data_for_rm_gallery(spans_df)

    try:
        # Execute batch evaluation with custom grader
        # The custom grader is processed alongside standard graders
        batch_results = await custom_runner.arun(evaluation_data)

        # Process results
        # Convert batch results to DataFrame format
        eval_results = []
        for i in range(len(evaluation_data)):
            span_id = span_id_mapping[i]
            result_dict = {"span_id": span_id}

            # Process results from all graders including the custom one
            for grader_name, grader_results in batch_results.items():
                if i < len(grader_results):
                    result = grader_results[i]
                    # Handle different result types appropriately
                    if isinstance(result, (GraderScore, GraderRank)):
                        result_dict[grader_name] = getattr(result, 'score', getattr(result, 'rank', 0))
                        result_dict[f"{grader_name}_reasoning"] = getattr(result, 'reason', '')
                    elif isinstance(result, GraderError):
                        result_dict[grader_name] = None
                        result_dict[f"{grader_name}_error"] = result.error

            eval_results.append(result_dict)

        # Return results as DataFrame
        return pd.DataFrame(eval_results)

    except Exception as e:
        # Handle any errors during custom batch evaluation
        print(f"Error in custom batch evaluation: {str(e)}")
        return pd.DataFrame()

# Execute custom evaluation
# Uncomment to run the custom evaluation
# custom_eval_df = asyncio.run(custom_evaluation_with_batch_processing(spans_df))
```

## Tips

- :bulb: **Grader Error Handling**: Always check for `GraderError` results when using graders. These indicate evaluation failures that should be handled gracefully rather than accessing non-existent attributes.

- :bulb: **Grader Result Types**: Understand the different result types (`GraderScore`, `GraderRank`, `GraderError`) that graders can return and handle each appropriately in your integration code.

- :bulb: **Runner Concurrency**: Use the `max_concurrency` parameter in `GradingRunner` to control how many graders run simultaneously, preventing resource exhaustion.

- :bulb: **Batch Processing with Runner**: For evaluating multiple data points, use `GradingRunner.arun()` to process them in batches rather than calling graders individually for better performance.

- :bulb: **Runner Aggregators**: Take advantage of built-in aggregators in `GradingRunner` rather than implementing your own result aggregation logic.

- :bulb: **Async Grader Methods**: Always use the async methods (e.g., `aevaluate`) when working with graders to ensure proper asynchronous execution.

## Related Resources

- [RM-Gallery Grader Development Guide](../../../building-graders/overview.md)
- [Arize Phoenix Official Documentation](https://docs.arize.com/phoenix)
- [OpenInference Specification](https://github.com/Arize-ai/openinference)