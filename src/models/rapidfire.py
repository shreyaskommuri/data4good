"""
RapidFire AI Execution Framework

A hyper-parallelization framework for testing multiple prompts, 
model configurations, and classification strategies simultaneously.

Features:
- Shard-based data scheduling
- Parallel prompt/model testing
- Automatic result aggregation
- Performance metrics collection
- Best classifier selection

Author: Coastal Labor-Resilience Engine Team
"""

import os
import time
import logging
import hashlib
from enum import Enum
from typing import Optional, List, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import threading
import queue

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """Execution mode for parallel processing."""
    SEQUENTIAL = "sequential"
    THREADED = "threaded"
    PROCESS = "process"


class ShardStrategy(Enum):
    """Strategy for sharding data."""
    ROUND_ROBIN = "round_robin"
    HASH_BASED = "hash_based"
    SIZE_BALANCED = "size_balanced"
    RANDOM = "random"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    prompt_template: str
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 150
    batch_size: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_id(self) -> str:
        """Generate unique ID for this configuration."""
        content = f"{self.name}:{self.model_name}:{self.temperature}:{self.prompt_template[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class ShardResult:
    """Result from processing a single shard."""
    shard_id: int
    experiment_id: str
    results: List[Dict[str, Any]]
    processing_time: float
    success_count: int
    error_count: int
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Aggregated result for an experiment."""
    experiment_id: str
    config: ExperimentConfig
    shard_results: List[ShardResult]
    total_time: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    @property
    def total_processed(self) -> int:
        return sum(s.success_count for s in self.shard_results)
    
    @property
    def total_errors(self) -> int:
        return sum(s.error_count for s in self.shard_results)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_name": self.config.name,
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "total_time": self.total_time,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score
        }


class DataSharder:
    """
    Shards data for parallel processing.
    
    Divides large datasets into smaller chunks that can be
    processed in parallel across multiple workers.
    """
    
    def __init__(
        self,
        num_shards: int = 4,
        strategy: ShardStrategy = ShardStrategy.ROUND_ROBIN
    ):
        """
        Initialize the data sharder.
        
        Args:
            num_shards: Number of shards to create
            strategy: Sharding strategy to use
        """
        self.num_shards = num_shards
        self.strategy = strategy
    
    def shard(
        self,
        data: List[Any],
        key_func: Optional[Callable[[Any], str]] = None
    ) -> List[List[Any]]:
        """
        Shard data into multiple chunks.
        
        Args:
            data: List of items to shard
            key_func: Optional function to extract key for hash-based sharding
            
        Returns:
            List of shards (each shard is a list of items)
        """
        if not data:
            return [[] for _ in range(self.num_shards)]
        
        if self.strategy == ShardStrategy.ROUND_ROBIN:
            return self._round_robin_shard(data)
        elif self.strategy == ShardStrategy.HASH_BASED:
            return self._hash_shard(data, key_func)
        elif self.strategy == ShardStrategy.SIZE_BALANCED:
            return self._size_balanced_shard(data)
        elif self.strategy == ShardStrategy.RANDOM:
            return self._random_shard(data)
        else:
            return self._round_robin_shard(data)
    
    def _round_robin_shard(self, data: List[Any]) -> List[List[Any]]:
        """Distribute items in round-robin fashion."""
        shards = [[] for _ in range(self.num_shards)]
        for i, item in enumerate(data):
            shards[i % self.num_shards].append(item)
        return shards
    
    def _hash_shard(
        self,
        data: List[Any],
        key_func: Optional[Callable[[Any], str]] = None
    ) -> List[List[Any]]:
        """Shard based on hash of key."""
        shards = [[] for _ in range(self.num_shards)]
        
        for item in data:
            if key_func:
                key = key_func(item)
            else:
                key = str(item)
            
            hash_val = int(hashlib.md5(key.encode()).hexdigest(), 16)
            shard_idx = hash_val % self.num_shards
            shards[shard_idx].append(item)
        
        return shards
    
    def _size_balanced_shard(self, data: List[Any]) -> List[List[Any]]:
        """Balance shards to have similar sizes."""
        # Sort by estimated processing complexity (string length as proxy)
        sorted_data = sorted(data, key=lambda x: len(str(x)), reverse=True)
        
        shards = [[] for _ in range(self.num_shards)]
        shard_sizes = [0] * self.num_shards
        
        for item in sorted_data:
            # Add to smallest shard
            min_idx = shard_sizes.index(min(shard_sizes))
            shards[min_idx].append(item)
            shard_sizes[min_idx] += len(str(item))
        
        return shards
    
    def _random_shard(self, data: List[Any]) -> List[List[Any]]:
        """Randomly distribute items."""
        shards = [[] for _ in range(self.num_shards)]
        indices = np.random.randint(0, self.num_shards, size=len(data))
        
        for item, idx in zip(data, indices):
            shards[idx].append(item)
        
        return shards


class RapidFireScheduler:
    """
    Shard-based scheduler for parallel experiment execution.
    
    Coordinates the execution of multiple experiments across
    multiple data shards with configurable parallelism.
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        mode: ExecutionMode = ExecutionMode.THREADED,
        sharder: Optional[DataSharder] = None
    ):
        """
        Initialize the scheduler.
        
        Args:
            max_workers: Maximum parallel workers
            mode: Execution mode (threaded, process, sequential)
            sharder: DataSharder instance
        """
        self.max_workers = max_workers
        self.mode = mode
        self.sharder = sharder or DataSharder(num_shards=max_workers)
        
        self._results_queue = queue.Queue()
        self._progress_lock = threading.Lock()
        self._progress = {}
    
    def schedule_experiment(
        self,
        config: ExperimentConfig,
        data: List[Any],
        classifier_factory: Callable[[ExperimentConfig], Any],
        classify_func: Callable[[Any, Any], Dict[str, Any]]
    ) -> ExperimentResult:
        """
        Schedule and run a single experiment across shards.
        
        Args:
            config: Experiment configuration
            data: Data to classify
            classifier_factory: Function to create classifier from config
            classify_func: Function to classify single item
            
        Returns:
            ExperimentResult with aggregated metrics
        """
        start_time = time.time()
        experiment_id = config.get_id()
        
        logger.info(f"Starting experiment: {config.name} ({experiment_id})")
        
        # Shard the data
        shards = self.sharder.shard(data)
        logger.info(f"Created {len(shards)} shards with sizes: {[len(s) for s in shards]}")
        
        # Create classifier
        classifier = classifier_factory(config)
        
        # Process shards
        shard_results = []
        
        if self.mode == ExecutionMode.SEQUENTIAL:
            for shard_id, shard_data in enumerate(shards):
                result = self._process_shard(
                    shard_id, shard_data, classifier, classify_func, experiment_id
                )
                shard_results.append(result)
        else:
            executor_class = (
                ThreadPoolExecutor if self.mode == ExecutionMode.THREADED
                else ProcessPoolExecutor
            )
            
            with executor_class(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._process_shard,
                        shard_id, shard_data, classifier, classify_func, experiment_id
                    ): shard_id
                    for shard_id, shard_data in enumerate(shards)
                }
                
                for future in as_completed(futures):
                    shard_id = futures[future]
                    try:
                        result = future.result()
                        shard_results.append(result)
                        logger.info(f"Shard {shard_id} completed: {result.success_count} items")
                    except Exception as e:
                        logger.error(f"Shard {shard_id} failed: {e}")
        
        total_time = time.time() - start_time
        
        return ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            shard_results=shard_results,
            total_time=total_time
        )
    
    def _process_shard(
        self,
        shard_id: int,
        shard_data: List[Any],
        classifier: Any,
        classify_func: Callable[[Any, Any], Dict[str, Any]],
        experiment_id: str
    ) -> ShardResult:
        """Process a single shard of data."""
        start_time = time.time()
        results = []
        success_count = 0
        error_count = 0
        
        for item in shard_data:
            try:
                result = classify_func(classifier, item)
                results.append(result)
                success_count += 1
            except Exception as e:
                error_count += 1
                results.append({
                    "item": str(item),
                    "error": str(e),
                    "category": "error"
                })
        
        processing_time = time.time() - start_time
        
        return ShardResult(
            shard_id=shard_id,
            experiment_id=experiment_id,
            results=results,
            processing_time=processing_time,
            success_count=success_count,
            error_count=error_count
        )


class RapidFireEngine:
    """
    Main engine for running parallel classification experiments.
    
    Orchestrates multiple experiments with different configurations
    to find the best classifier for job title climate classification.
    """
    
    def __init__(
        self,
        num_shards: int = 4,
        max_workers: int = 4,
        mode: ExecutionMode = ExecutionMode.THREADED
    ):
        """
        Initialize the RapidFire engine.
        
        Args:
            num_shards: Number of data shards
            max_workers: Maximum parallel workers
            mode: Execution mode
        """
        self.sharder = DataSharder(num_shards=num_shards)
        self.scheduler = RapidFireScheduler(
            max_workers=max_workers,
            mode=mode,
            sharder=self.sharder
        )
        self.experiments: List[ExperimentConfig] = []
        self.results: List[ExperimentResult] = []
    
    def add_experiment(self, config: ExperimentConfig) -> None:
        """Add an experiment configuration."""
        self.experiments.append(config)
        logger.info(f"Added experiment: {config.name}")
    
    def add_experiments_from_prompts(
        self,
        prompts: List[str],
        model_name: str = "gpt-4o-mini",
        temperatures: List[float] = None
    ) -> None:
        """
        Generate experiments from prompt templates.
        
        Args:
            prompts: List of prompt templates
            model_name: Model to use
            temperatures: List of temperatures to test
        """
        temperatures = temperatures or [0.0, 0.1, 0.3]
        
        for i, prompt in enumerate(prompts):
            for temp in temperatures:
                config = ExperimentConfig(
                    name=f"prompt_{i+1}_temp_{temp}",
                    prompt_template=prompt,
                    model_name=model_name,
                    temperature=temp
                )
                self.add_experiment(config)
    
    def run_all(
        self,
        data: List[Any],
        classifier_factory: Callable[[ExperimentConfig], Any],
        classify_func: Callable[[Any, Any], Dict[str, Any]],
        ground_truth: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Run all experiments and return comparison.
        
        Args:
            data: Data to classify
            classifier_factory: Creates classifier from config
            classify_func: Classification function
            ground_truth: Optional ground truth labels for evaluation
            
        Returns:
            DataFrame comparing experiment results
        """
        logger.info(f"Running {len(self.experiments)} experiments on {len(data)} items")
        
        self.results = []
        
        for config in self.experiments:
            result = self.scheduler.schedule_experiment(
                config, data, classifier_factory, classify_func
            )
            
            # Calculate metrics if ground truth available
            if ground_truth:
                result = self._calculate_metrics(result, ground_truth)
            
            self.results.append(result)
            logger.info(f"Completed: {config.name} in {result.total_time:.2f}s")
        
        return self.get_comparison_df()
    
    def _calculate_metrics(
        self,
        result: ExperimentResult,
        ground_truth: List[str]
    ) -> ExperimentResult:
        """Calculate accuracy, precision, recall, F1 for an experiment."""
        # Flatten all predictions
        predictions = []
        for shard_result in result.shard_results:
            for r in shard_result.results:
                predictions.append(r.get("category", "error"))
        
        if len(predictions) != len(ground_truth):
            logger.warning(f"Prediction count ({len(predictions)}) != ground truth ({len(ground_truth)})")
            return result
        
        # Calculate accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        result.accuracy = correct / len(predictions) if predictions else 0
        
        # For binary classification, calculate precision/recall
        # Assuming "climate_sensitive" is the positive class
        positive_class = "climate_sensitive"
        
        true_positives = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p == positive_class and g == positive_class
        )
        false_positives = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p == positive_class and g != positive_class
        )
        false_negatives = sum(
            1 for p, g in zip(predictions, ground_truth)
            if p != positive_class and g == positive_class
        )
        
        result.precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0 else 0
        )
        result.recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0 else 0
        )
        result.f1_score = (
            2 * (result.precision * result.recall) / (result.precision + result.recall)
            if (result.precision + result.recall) > 0 else 0
        )
        
        return result
    
    def get_comparison_df(self) -> pd.DataFrame:
        """Get DataFrame comparing all experiment results."""
        return pd.DataFrame([r.to_dict() for r in self.results])
    
    def get_best_experiment(
        self,
        metric: str = "accuracy"
    ) -> Optional[ExperimentResult]:
        """Get the best performing experiment by metric."""
        if not self.results:
            return None
        
        return max(
            self.results,
            key=lambda r: getattr(r, metric) or 0
        )
    
    def save_results(self, output_path: str) -> None:
        """Save experiment results to CSV."""
        df = self.get_comparison_df()
        df.to_csv(output_path, index=False)
        logger.info(f"Saved results to {output_path}")


# ============================================================================
# Prompt Templates for Testing
# ============================================================================

CLASSIFICATION_PROMPTS = {
    "concise": """Classify job as CLIMATE_SENSITIVE or CLIMATE_RESILIENT.
Job: {title}
Output JSON: {{"category": "...", "confidence": 0.0-1.0}}""",

    "detailed": """You are a climate impact analyst. Classify this job based on weather vulnerability.

CLIMATE_SENSITIVE: Outdoor work, agriculture, fishing, tourism, construction
CLIMATE_RESILIENT: Indoor/remote work, technology, professional services

Job Title: {title}
{industry_context}

Respond with JSON: {{"category": "CLIMATE_SENSITIVE" or "CLIMATE_RESILIENT", "confidence": 0.0-1.0, "reasoning": "brief"}}""",

    "example_based": """Classify job by climate impact:

Examples of CLIMATE_SENSITIVE: Surf Instructor, Farmer, Fishing Boat Captain, Lifeguard
Examples of CLIMATE_RESILIENT: Software Engineer, Accountant, Data Analyst, Remote Manager

Job: {title}

JSON response: {{"category": "...", "confidence": 0.0-1.0}}""",

    "chain_of_thought": """Analyze this job for climate impact step by step:

1. Is it primarily outdoor or indoor work?
2. Does weather affect daily operations?
3. Can it be done remotely?
4. Is it in agriculture, fishing, tourism, or construction?

Job Title: {title}

Based on analysis, classify as CLIMATE_SENSITIVE or CLIMATE_RESILIENT.
JSON: {{"category": "...", "confidence": 0.0-1.0, "reasoning": "..."}}""",

    "binary": """Job: {title}
Climate-sensitive (outdoor/weather-dependent) or Climate-resilient (indoor/remote)?
Answer JSON: {{"category": "climate_sensitive" or "climate_resilient"}}"""
}


def create_prompt_experiments(
    models: List[str] = None,
    temperatures: List[float] = None
) -> List[ExperimentConfig]:
    """
    Create experiment configurations for all prompt/model combinations.
    
    Args:
        models: List of model names to test
        temperatures: List of temperatures to test
        
    Returns:
        List of ExperimentConfig objects
    """
    models = models or ["gpt-4o-mini"]
    temperatures = temperatures or [0.0, 0.1, 0.3]
    
    experiments = []
    
    for prompt_name, prompt_template in CLASSIFICATION_PROMPTS.items():
        for model in models:
            for temp in temperatures:
                config = ExperimentConfig(
                    name=f"{prompt_name}_{model.replace('-', '_')}_t{temp}",
                    prompt_template=prompt_template,
                    model_name=model,
                    temperature=temp,
                    metadata={"prompt_style": prompt_name}
                )
                experiments.append(config)
    
    return experiments


# ============================================================================
# Main - Demo RapidFire Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RapidFire AI Execution Framework - Demo")
    print("=" * 60)
    
    # Sample data
    test_titles = [
        "Surf Instructor",
        "Commercial Fisherman",
        "Software Engineer",
        "Remote Project Manager",
        "Farm Manager",
        "Data Scientist",
        "Beach Lifeguard",
        "Accountant",
        "Fishing Boat Captain",
        "Cybersecurity Analyst",
    ]
    
    # Ground truth labels
    ground_truth = [
        "climate_sensitive",   # Surf Instructor
        "climate_sensitive",   # Commercial Fisherman
        "climate_resilient",   # Software Engineer
        "climate_resilient",   # Remote Project Manager
        "climate_sensitive",   # Farm Manager
        "climate_resilient",   # Data Scientist
        "climate_sensitive",   # Beach Lifeguard
        "climate_resilient",   # Accountant
        "climate_sensitive",   # Fishing Boat Captain
        "climate_resilient",   # Cybersecurity Analyst
    ]
    
    # Test with rule-based classifier (no API needed)
    from src.models.job_classifier import RuleBasedClassifier, ExperimentConfig
    
    print("\n📊 Testing Data Sharder:")
    print("-" * 40)
    sharder = DataSharder(num_shards=3, strategy=ShardStrategy.ROUND_ROBIN)
    shards = sharder.shard(test_titles)
    for i, shard in enumerate(shards):
        print(f"  Shard {i}: {shard}")
    
    print("\n🚀 Testing RapidFire Scheduler:")
    print("-" * 40)
    
    # Create rule-based experiments with different thresholds
    engine = RapidFireEngine(num_shards=2, max_workers=2)
    
    # Add experiments with different confidence thresholds
    for threshold in [0.5, 0.6, 0.7, 0.8]:
        config = ExperimentConfig(
            name=f"rule_based_thresh_{threshold}",
            prompt_template="N/A",
            model_name="rule_based",
            temperature=threshold,  # reusing as threshold
            metadata={"confidence_threshold": threshold}
        )
        engine.add_experiment(config)
    
    # Define classifier factory and classify function
    def classifier_factory(config: ExperimentConfig):
        threshold = config.metadata.get("confidence_threshold", 0.6)
        return RuleBasedClassifier(confidence_threshold=threshold)
    
    def classify_func(classifier, title):
        result = classifier.classify(title)
        return result.to_dict()
    
    # Run experiments
    results_df = engine.run_all(
        data=test_titles,
        classifier_factory=classifier_factory,
        classify_func=classify_func,
        ground_truth=ground_truth
    )
    
    print("\n📈 Experiment Results:")
    print(results_df[["config_name", "total_processed", "accuracy", "f1_score", "total_time"]].to_string(index=False))
    
    # Best experiment
    best = engine.get_best_experiment(metric="accuracy")
    if best:
        print(f"\n✅ Best Experiment: {best.config.name}")
        print(f"   Accuracy: {best.accuracy:.2%}")
        print(f"   F1 Score: {best.f1_score:.2%}")
