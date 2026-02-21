"""
Models package for Coastal Labor-Resilience Engine.

Phase 2: NLP Classification
- Job title climate classification
- RapidFire parallel execution framework
- Training data and evaluation pipeline

Phase 3: Mathematical Engine
- Markov transition matrix for labor state dynamics
- Resilience ODE solver with EJ friction
- Coupled Markov-ODE simulation

Submodules:
- job_classifier: Rule-based and LLM-based job classifiers
- rapidfire: Hyper-parallelization framework for model testing
- training_data: Labeled dataset for training and evaluation
- evaluation: Metrics calculation and comparison tools
- markov_chain: Markov transition matrix for labor categories
- resilience_ode: Differential equation solver for labor dynamics
"""

from .job_classifier import (
    ClimateCategory,
    ClassificationResult,
    RuleBasedClassifier,
    LLMClassifier,
    LLMClassifierConfig,
    EnsembleClassifier,
    classify_job_titles
)

from .rapidfire import (
    RapidFireEngine,
    RapidFireScheduler,
    DataSharder,
    ExperimentConfig,
    ExperimentResult,
    ShardStrategy,
    ExecutionMode,
    CLASSIFICATION_PROMPTS,
    create_prompt_experiments
)

from .training_data import (
    LabeledJobTitle,
    CLIMATE_SENSITIVE_JOBS,
    CLIMATE_RESILIENT_JOBS,
    get_all_training_data,
    get_training_df,
    get_evaluation_split,
    get_benchmark_dataset
)

from .evaluation import (
    ClassificationMetrics,
    EvaluationResult,
    MetricsCalculator,
    ClassifierEvaluator,
    run_full_evaluation
)

# Phase 3: Mathematical Engine
from .markov_chain import (
    LaborState,
    STATE_INDEX,
    INDEX_STATE,
    TransitionData,
    ShockParameters,
    MarkovAnalysisResult,
    MarkovTransitionMatrix,
    create_regional_chain
)

from .resilience_ode import (
    EJScreenData,
    ODEParameters,
    ClimateShock,
    ODESolution,
    ResilienceODE,
    CoupledMarkovODE,
    create_shock_scenario,
    create_ej_tract
)

__all__ = [
    # Classification
    "ClimateCategory",
    "ClassificationResult",
    "RuleBasedClassifier",
    "LLMClassifier",
    "LLMClassifierConfig",
    "EnsembleClassifier",
    "classify_job_titles",
    
    # RapidFire Framework
    "RapidFireEngine",
    "RapidFireScheduler",
    "DataSharder",
    "ExperimentConfig",
    "ExperimentResult",
    "ShardStrategy",
    "ExecutionMode",
    "CLASSIFICATION_PROMPTS",
    "create_prompt_experiments",
    
    # Training Data
    "LabeledJobTitle",
    "CLIMATE_SENSITIVE_JOBS",
    "CLIMATE_RESILIENT_JOBS",
    "get_all_training_data",
    "get_training_df",
    "get_evaluation_split",
    "get_benchmark_dataset",
    
    # Evaluation
    "ClassificationMetrics",
    "EvaluationResult",
    "MetricsCalculator",
    "ClassifierEvaluator",
    "run_full_evaluation",
    
    # Phase 3: Markov Chain
    "LaborState",
    "STATE_INDEX",
    "INDEX_STATE",
    "TransitionData",
    "ShockParameters",
    "MarkovAnalysisResult",
    "MarkovTransitionMatrix",
    "create_regional_chain",
    
    # Phase 3: Resilience ODE
    "EJScreenData",
    "ODEParameters",
    "ClimateShock",
    "ODESolution",
    "ResilienceODE",
    "CoupledMarkovODE",
    "create_shock_scenario",
    "create_ej_tract",
]
