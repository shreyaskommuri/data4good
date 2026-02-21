"""
Evaluation Pipeline for Job Title Climate Classification

Provides comprehensive metrics and reports for comparing
classifier performance across different models and configurations.

Features:
- Accuracy, Precision, Recall, F1 metrics
- Confusion matrix visualization
- Per-class performance analysis
- Confidence calibration analysis
- Cross-validation support
- Report generation

Author: Coastal Labor-Resilience Engine Team
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ClassificationMetrics:
    """Metrics for a classification evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    support: int  # Number of samples
    
    # Per-class metrics
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_support: Dict[str, int] = field(default_factory=dict)
    
    # Confusion matrix
    confusion_matrix: Optional[np.ndarray] = None
    class_labels: List[str] = field(default_factory=list)
    
    # Additional stats
    avg_confidence: float = 0.0
    uncertain_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "support": self.support,
            "per_class_precision": self.per_class_precision,
            "per_class_recall": self.per_class_recall,
            "per_class_f1": self.per_class_f1,
            "per_class_support": self.per_class_support,
            "avg_confidence": self.avg_confidence,
            "uncertain_count": self.uncertain_count,
            "error_count": self.error_count
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result."""
    model_name: str
    config_name: str
    metrics: ClassificationMetrics
    predictions: List[str]
    ground_truth: List[str]
    confidences: List[float]
    evaluation_time: float
    
    # Error analysis
    misclassified_indices: List[int] = field(default_factory=list)
    misclassified_titles: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "config_name": self.config_name,
            "metrics": self.metrics.to_dict(),
            "evaluation_time": self.evaluation_time,
            "num_misclassified": len(self.misclassified_indices)
        }


class MetricsCalculator:
    """
    Calculates classification metrics from predictions and ground truth.
    """
    
    def __init__(self, positive_class: str = "climate_sensitive"):
        """
        Initialize calculator.
        
        Args:
            positive_class: Class to consider as positive for binary metrics
        """
        self.positive_class = positive_class
    
    def calculate(
        self,
        predictions: List[str],
        ground_truth: List[str],
        confidences: Optional[List[float]] = None
    ) -> ClassificationMetrics:
        """
        Calculate all metrics.
        
        Args:
            predictions: Predicted labels
            ground_truth: True labels
            confidences: Optional prediction confidences
            
        Returns:
            ClassificationMetrics instance
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")
        
        n_samples = len(predictions)
        
        # Get unique classes
        all_classes = sorted(set(ground_truth) | set(predictions))
        
        # Overall accuracy
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / n_samples if n_samples > 0 else 0
        
        # Per-class metrics
        per_class_precision = {}
        per_class_recall = {}
        per_class_f1 = {}
        per_class_support = {}
        
        for cls in all_classes:
            tp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g == cls)
            fp = sum(1 for p, g in zip(predictions, ground_truth) if p == cls and g != cls)
            fn = sum(1 for p, g in zip(predictions, ground_truth) if p != cls and g == cls)
            support = sum(1 for g in ground_truth if g == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            per_class_precision[cls] = precision
            per_class_recall[cls] = recall
            per_class_f1[cls] = f1
            per_class_support[cls] = support
        
        # Macro-averaged metrics
        macro_precision = np.mean(list(per_class_precision.values()))
        macro_recall = np.mean(list(per_class_recall.values()))
        macro_f1 = np.mean(list(per_class_f1.values()))
        
        # Binary metrics for positive class
        bin_precision = per_class_precision.get(self.positive_class, 0)
        bin_recall = per_class_recall.get(self.positive_class, 0)
        bin_f1 = per_class_f1.get(self.positive_class, 0)
        
        # Confusion matrix
        class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
        n_classes = len(all_classes)
        cm = np.zeros((n_classes, n_classes), dtype=int)
        
        for pred, true in zip(predictions, ground_truth):
            if pred in class_to_idx and true in class_to_idx:
                cm[class_to_idx[true], class_to_idx[pred]] += 1
        
        # Confidence stats
        avg_confidence = np.mean(confidences) if confidences else 0
        uncertain_count = sum(1 for p in predictions if p == "uncertain")
        
        return ClassificationMetrics(
            accuracy=accuracy,
            precision=bin_precision,
            recall=bin_recall,
            f1_score=bin_f1,
            support=n_samples,
            per_class_precision=per_class_precision,
            per_class_recall=per_class_recall,
            per_class_f1=per_class_f1,
            per_class_support=per_class_support,
            confusion_matrix=cm,
            class_labels=all_classes,
            avg_confidence=avg_confidence,
            uncertain_count=uncertain_count
        )


class ClassifierEvaluator:
    """
    Evaluates job title classifiers with comprehensive metrics.
    """
    
    def __init__(
        self,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            metrics_calculator: MetricsCalculator instance
        """
        self.calculator = metrics_calculator or MetricsCalculator()
        self.results: List[EvaluationResult] = []
    
    def evaluate(
        self,
        classifier: Any,
        test_titles: List[str],
        ground_truth: List[str],
        model_name: str = "unknown",
        config_name: str = "default"
    ) -> EvaluationResult:
        """
        Evaluate a classifier on test data.
        
        Args:
            classifier: Classifier with classify_batch method
            test_titles: List of job titles to classify
            ground_truth: True labels
            model_name: Name of the model
            config_name: Name of the configuration
            
        Returns:
            EvaluationResult with all metrics
        """
        import time
        start_time = time.time()
        
        # Get predictions
        results = classifier.classify_batch(test_titles)
        
        # Extract predictions and confidences
        predictions = [r.category.value if hasattr(r.category, 'value') else str(r.category) for r in results]
        confidences = [r.confidence for r in results]
        
        evaluation_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self.calculator.calculate(predictions, ground_truth, confidences)
        
        # Find misclassified examples
        misclassified_indices = [
            i for i, (p, g) in enumerate(zip(predictions, ground_truth))
            if p != g
        ]
        misclassified_titles = [test_titles[i] for i in misclassified_indices]
        
        result = EvaluationResult(
            model_name=model_name,
            config_name=config_name,
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            confidences=confidences,
            evaluation_time=evaluation_time,
            misclassified_indices=misclassified_indices,
            misclassified_titles=misclassified_titles
        )
        
        self.results.append(result)
        
        return result
    
    def cross_validate(
        self,
        classifier_factory: Callable[[], Any],
        titles: List[str],
        labels: List[str],
        n_folds: int = 5,
        model_name: str = "unknown"
    ) -> Dict[str, List[float]]:
        """
        Perform k-fold cross-validation.
        
        Args:
            classifier_factory: Function that creates a new classifier
            titles: All titles
            labels: All labels
            n_folds: Number of folds
            model_name: Name of the model
            
        Returns:
            Dict mapping metric names to lists of per-fold values
        """
        indices = list(range(len(titles)))
        np.random.shuffle(indices)
        
        fold_size = len(indices) // n_folds
        
        metrics_per_fold = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1_score": []
        }
        
        for fold in range(n_folds):
            # Split data
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < n_folds - 1 else len(indices)
            
            test_indices = indices[test_start:test_end]
            train_indices = indices[:test_start] + indices[test_end:]
            
            test_titles = [titles[i] for i in test_indices]
            test_labels = [labels[i] for i in test_indices]
            
            # Evaluate
            classifier = classifier_factory()
            result = self.evaluate(
                classifier, test_titles, test_labels,
                model_name=model_name,
                config_name=f"fold_{fold+1}"
            )
            
            metrics_per_fold["accuracy"].append(result.metrics.accuracy)
            metrics_per_fold["precision"].append(result.metrics.precision)
            metrics_per_fold["recall"].append(result.metrics.recall)
            metrics_per_fold["f1_score"].append(result.metrics.f1_score)
        
        # Calculate mean and std
        for metric in metrics_per_fold:
            values = metrics_per_fold[metric]
            mean = np.mean(values)
            std = np.std(values)
            logger.info(f"CV {metric}: {mean:.3f} (±{std:.3f})")
        
        return metrics_per_fold
    
    def compare_classifiers(
        self,
        classifiers: Dict[str, Any],
        test_titles: List[str],
        ground_truth: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple classifiers on the same test set.
        
        Args:
            classifiers: Dict mapping names to classifier instances
            test_titles: Test job titles
            ground_truth: True labels
            
        Returns:
            DataFrame comparing all classifiers
        """
        results_data = []
        
        for name, classifier in classifiers.items():
            result = self.evaluate(
                classifier, test_titles, ground_truth,
                model_name=name, config_name="default"
            )
            
            results_data.append({
                "model": name,
                "accuracy": result.metrics.accuracy,
                "precision": result.metrics.precision,
                "recall": result.metrics.recall,
                "f1_score": result.metrics.f1_score,
                "avg_confidence": result.metrics.avg_confidence,
                "uncertain_count": result.metrics.uncertain_count,
                "time_seconds": result.evaluation_time
            })
        
        return pd.DataFrame(results_data)
    
    def generate_report(
        self,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a text report of all evaluations.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        lines = [
            "=" * 70,
            "JOB TITLE CLIMATE CLASSIFICATION - EVALUATION REPORT",
            "=" * 70,
            ""
        ]
        
        for result in self.results:
            lines.extend([
                f"Model: {result.model_name} ({result.config_name})",
                "-" * 50,
                f"  Accuracy:    {result.metrics.accuracy:.3f}",
                f"  Precision:   {result.metrics.precision:.3f}",
                f"  Recall:      {result.metrics.recall:.3f}",
                f"  F1 Score:    {result.metrics.f1_score:.3f}",
                f"  Support:     {result.metrics.support}",
                f"  Avg Conf:    {result.metrics.avg_confidence:.3f}",
                f"  Uncertain:   {result.metrics.uncertain_count}",
                f"  Eval Time:   {result.evaluation_time:.2f}s",
                "",
                "  Per-class F1:",
            ])
            
            for cls, f1 in result.metrics.per_class_f1.items():
                support = result.metrics.per_class_support.get(cls, 0)
                lines.append(f"    {cls}: {f1:.3f} (n={support})")
            
            if result.misclassified_titles[:5]:
                lines.extend([
                    "",
                    "  Sample Misclassifications:",
                ])
                for i, title in enumerate(result.misclassified_titles[:5]):
                    idx = result.misclassified_indices[i]
                    pred = result.predictions[idx]
                    true = result.ground_truth[idx]
                    lines.append(f"    '{title}': predicted={pred}, true={true}")
            
            lines.append("")
        
        report = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(report)
            logger.info(f"Report saved to {output_path}")
        
        return report


def run_full_evaluation(
    classifiers: Dict[str, Any],
    output_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run a complete evaluation of classifiers on training data.
    
    Args:
        classifiers: Dict mapping names to classifier instances
        output_dir: Optional directory to save results
        
    Returns:
        DataFrame with comparison results
    """
    from .training_data import get_benchmark_dataset
    
    # Get benchmark data
    titles, labels = get_benchmark_dataset()
    
    # Evaluate
    evaluator = ClassifierEvaluator()
    comparison_df = evaluator.compare_classifiers(classifiers, titles, labels)
    
    # Generate report
    report = evaluator.generate_report()
    print(report)
    
    # Save results
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        comparison_df.to_csv(output_path / "comparison.csv", index=False)
        (output_path / "report.txt").write_text(report)
        
        logger.info(f"Results saved to {output_path}")
    
    return comparison_df


# ============================================================================
# Main - Demo Evaluation
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Job Title Classifier Evaluation Pipeline")
    print("=" * 60)
    
    from .job_classifier import RuleBasedClassifier
    from .training_data import get_benchmark_dataset
    
    # Get benchmark data
    titles, labels = get_benchmark_dataset()
    print(f"\n📊 Benchmark Dataset: {len(titles)} samples")
    
    # Create classifiers with different thresholds
    classifiers = {
        "rule_thresh_0.5": RuleBasedClassifier(confidence_threshold=0.5),
        "rule_thresh_0.6": RuleBasedClassifier(confidence_threshold=0.6),
        "rule_thresh_0.7": RuleBasedClassifier(confidence_threshold=0.7),
        "rule_thresh_0.8": RuleBasedClassifier(confidence_threshold=0.8),
    }
    
    # Evaluate
    evaluator = ClassifierEvaluator()
    comparison = evaluator.compare_classifiers(classifiers, titles, labels)
    
    print("\n📈 Classifier Comparison:")
    print("-" * 70)
    print(comparison.to_string(index=False))
    
    # Best model
    best_idx = comparison["f1_score"].idxmax()
    best_model = comparison.loc[best_idx, "model"]
    best_f1 = comparison.loc[best_idx, "f1_score"]
    
    print(f"\n✅ Best Model: {best_model} (F1={best_f1:.3f})")
    
    # Generate report
    print("\n" + evaluator.generate_report())
