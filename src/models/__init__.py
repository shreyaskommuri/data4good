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

Phase 4: ML Prediction Models
- Displacement risk prediction (XGBoost binary classifier)
- Economic impact scoring (XGBoost regression)

Submodules:
- job_classifier: Rule-based and LLM-based job classifiers
- rapidfire: Hyper-parallelization framework for model testing
- training_data: Labeled dataset for training and evaluation
- evaluation: Metrics calculation and comparison tools
- markov_chain: Markov transition matrix for labor categories
- resilience_ode: Differential equation solver for labor dynamics
- displacement_features: Feature engineering for displacement risk
- displacement_model: XGBoost model for displacement prediction
- displacement_predictor: Production inference for displacement risk
- economic_impact_features: Feature engineering for tract vulnerability
- economic_impact_model: XGBoost regression for vulnerability scoring
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

# Phase 4: ML Prediction Models
from .displacement_features import (
    DisplacementFeatures,
    DisplacementFeatureExtractor,
)

from .displacement_model import (
    DisplacementRiskModel,
    train_displacement_model,
    cross_validate_model,
)

from .displacement_predictor import (
    DisplacementPredictor,
    get_predictor,
    predict_displacement_risk,
)

from .economic_impact_features import (
    EconomicImpactFeatures,
    EconomicImpactFeatureExtractor,
)

from .economic_impact_model import (
    EconomicImpactModel,
    train_economic_impact_model,
    cross_validate_economic_model,
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
    
    # Phase 4: ML Models
    "DisplacementFeatures",
    "DisplacementFeatureExtractor",
    "DisplacementRiskModel",
    "train_displacement_model",
    "cross_validate_model",
    "DisplacementPredictor",
    "get_predictor",
    "predict_displacement_risk",
    "EconomicImpactFeatures",
    "EconomicImpactFeatureExtractor",
    "EconomicImpactModel",
    "train_economic_impact_model",
    "cross_validate_economic_model",

    "ODEParameters",
    "ClimateShock",
    "ODESolution",
    "ResilienceODE",
    "CoupledMarkovODE",
    "create_shock_scenario",
    "create_ej_tract",
]
