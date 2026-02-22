"""
Worker Displacement Risk - XGBoost Model Training
-------------------------------------------------
Trains XGBoost classifier to predict worker displacement risk
from climate shocks and sea level rise.

Model predicts probability that a worker in a climate-sensitive
industry will be displaced (leave coastal counties or industry).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import json
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from .displacement_features import prepare_training_data


class DisplacementRiskModel:
    """XGBoost-based displacement risk prediction model."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        min_child_weight: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            min_child_weight: Minimum sum of instance weight in a child
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            scale_pos_weight: Balancing of positive/negative weights (auto if None)
            random_state: Random seed
        """
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False,
        )
        self.feature_names = None
        self.training_metrics = {}
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        verbose: bool = True
    ):
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            verbose: Print training progress
        """
        self.feature_names = list(X_train.columns)
        
        # Auto-calculate scale_pos_weight if not set
        if self.model.scale_pos_weight is None:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            self.model.scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            if verbose:
                print(f"Auto scale_pos_weight: {self.model.scale_pos_weight:.2f}")
        
        # Train with early stopping if validation set provided
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=verbose
            )
        else:
            self.model.fit(X_train, y_train, verbose=verbose)
        
        if verbose:
            print("\nTraining complete!")
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test"
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset (for reporting)
            
        Returns:
            Dict of metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_proba) if len(np.unique(y)) > 1 else 0.0,
        }
        
        print(f"\n{dataset_name.upper()} SET METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  ROC AUC:   {metrics['roc_auc']:.4f}")
        
        # Check if we have both classes in the data
        if len(np.unique(y)) < 2:
            print(f"\n⚠️  Warning: Only one class present in {dataset_name} set")
            print(f"   Cannot compute full classification report")
            print(f"   Class distribution: {np.unique(y, return_counts=True)}")
        else:
            print(f"\nClassification Report:")
            try:
                print(classification_report(y, y_pred, target_names=['Retained', 'Displaced'], zero_division=0))
            except ValueError as e:
                print(f"  Could not generate report: {e}")
            
            print(f"\nConfusion Matrix:")
            cm = confusion_matrix(y, y_pred)
            print(cm)
            if cm.shape == (2, 2):
                print(f"  True Negatives:  {cm[0,0]}")
                print(f"  False Positives: {cm[0,1]}")
                print(f"  False Negatives: {cm[1,0]}")
                print(f"  True Positives:  {cm[1,1]}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with features and importance scores
        """
        importance = self.model.feature_importances_
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_df.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 6)):
        """Plot feature importance."""
        importance_df = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance (XGBoost)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, X: pd.DataFrame, y: pd.Series, figsize: Tuple[int, int] = (8, 6)):
        """Plot ROC curve."""
        y_proba = self.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - Displacement Risk Model')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DisplacementRiskModel':
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls()
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_metrics = model_data.get('training_metrics', {})
        
        print(f"Model loaded from: {filepath}")
        return instance


def train_displacement_model(
    data_path: str,
    max_records: Optional[int] = None,
    test_size: float = 0.2,
    val_size: float = 0.1,
    save_path: Optional[str] = None,
    random_state: int = 42
) -> DisplacementRiskModel:
    """
    Complete training pipeline for displacement risk model.
    
    Args:
        data_path: Path to Live Data JSONL files
        max_records: Maximum records to use
        test_size: Fraction for test set
        val_size: Fraction for validation set
        save_path: Path to save model (.pkl file)
        random_state: Random seed
        
    Returns:
        Trained DisplacementRiskModel
    """
    print("=" * 60)
    print("WORKER DISPLACEMENT RISK MODEL TRAINING")
    print("=" * 60)
    
    # Load and prepare data
    print("\n1. Loading and preparing training data...")
    X, y = prepare_training_data(data_path, max_records=max_records, min_jobs=2)
    
    # Split data
    print(f"\n2. Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
    )
    
    print(f"  Train: {len(X_train)} samples ({y_train.sum()} displaced)")
    print(f"  Val:   {len(X_val)} samples ({y_val.sum()} displaced)")
    print(f"  Test:  {len(X_test)} samples ({y_test.sum()} displaced)")
    
    # Initialize and train model
    print(f"\n3. Training XGBoost model...")
    model = DisplacementRiskModel(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state
    )
    
    model.train(X_train, y_train, X_val, y_val, verbose=True)
    
    # Evaluate
    print(f"\n4. Evaluating model...")
    train_metrics = model.evaluate(X_train, y_train, "train")
    val_metrics = model.evaluate(X_val, y_val, "validation")
    test_metrics = model.evaluate(X_test, y_test, "test")
    
    model.training_metrics = {
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
    }
    
    # Feature importance
    print(f"\n5. Feature importance:")
    importance_df = model.get_feature_importance(top_n=10)
    print(importance_df.to_string(index=False))
    
    # Save model
    if save_path:
        model.save(save_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return model


def cross_validate_model(
    data_path: str,
    max_records: Optional[int] = None,
    n_folds: int = 5,
    random_state: int = 42
):
    """
    Perform cross-validation to assess model stability.
    
    Args:
        data_path: Path to Live Data JSONL files
        max_records: Maximum records to use
        n_folds: Number of CV folds
        random_state: Random seed
    """
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION")
    print("=" * 60)
    
    # Load data
    X, y = prepare_training_data(data_path, max_records=max_records, min_jobs=2)
    
    # Create model
    model = DisplacementRiskModel(random_state=random_state)
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print(f"\nRunning {n_folds}-fold cross-validation...")
    scores = cross_val_score(model.model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    print(f"\nROC AUC Scores per fold:")
    for i, score in enumerate(scores, 1):
        print(f"  Fold {i}: {score:.4f}")
    
    print(f"\nMean ROC AUC: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Configuration
    DATA_DIR = Path(__file__).parent.parent.parent / "data" / "live_data"
    MODEL_DIR = Path(__file__).parent.parent.parent / "models"
    MODEL_PATH = MODEL_DIR / "displacement_risk_model.pkl"
    
    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        print("Please ensure Live Data Technologies JSONL files are in data/live_data/")
        sys.exit(1)
    
    # Train model
    model = train_displacement_model(
        data_path=str(DATA_DIR),
        max_records=5000,  # Use subset for faster training
        save_path=str(MODEL_PATH)
    )
    
    # Cross-validation
    print("\n\nRunning cross-validation for robustness check...")
    cross_validate_model(
        data_path=str(DATA_DIR),
        max_records=5000,
        n_folds=5
    )
