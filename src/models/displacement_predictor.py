"""
Displacement Risk Prediction Service
------------------------------------
Provides real-time displacement risk scoring for workers
using trained XGBoost model.
"""

from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
import pandas as pd

from .displacement_model import DisplacementRiskModel
from .displacement_features import DisplacementFeatureExtractor
from ..data.live_data_loader import PersonRecord


class DisplacementPredictor:
    """Production service for displacement risk prediction."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize predictor with trained model.
        
        Args:
            model_path: Path to saved model .pkl file
        """
        self.model = None
        self.feature_extractor = DisplacementFeatureExtractor()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        
    def load_model(self, model_path: str):
        """Load trained model from file."""
        self.model = DisplacementRiskModel.load(model_path)
        print(f"✓ Displacement risk model loaded")
        
    def predict_person(self, person: PersonRecord) -> Dict:
        """
        Predict displacement risk for a single person.
        
        Args:
            person: PersonRecord with job history
            
        Returns:
            Dict with risk score and explanation
        """
        if not self.model:
            return {
                'person_id': person.id,
                'risk_score': 0.5,
                'risk_level': 'unknown',
                'confidence': 0.0,
                'error': 'Model not loaded'
            }
        
        try:
            # Extract features
            features = self.feature_extractor.extract_features(person)
            feature_dict = features.to_dict()
            
            # Remove non-feature columns
            X = pd.DataFrame([feature_dict]).drop(['person_id', 'displaced'], axis=1)
            
            # Ensure feature order matches training
            X = X[self.model.feature_names]
            
            # Predict
            risk_prob = float(self.model.predict_proba(X)[0, 1])
            
            # Categorize risk
            if risk_prob < 0.3:
                risk_level = "low"
            elif risk_prob < 0.6:
                risk_level = "moderate"
            elif risk_prob < 0.8:
                risk_level = "high"
            else:
                risk_level = "critical"
            
            # Generate explanation
            explanation = self._generate_explanation(feature_dict, risk_prob)
            
            return {
                'person_id': person.id,
                'risk_score': round(risk_prob, 3),
                'risk_level': risk_level,
                'confidence': round(1.0 - abs(risk_prob - 0.5) * 2, 3),  # Higher at extremes
                'explanation': explanation,
                'features': {
                    'climate_sensitive': bool(feature_dict['current_climate_sensitive']),
                    'coastal': bool(feature_dict['current_coastal']),
                    'tenure_months': feature_dict['current_tenure_months'],
                    'job_stability': round(feature_dict['job_stability_score'], 2),
                }
            }
            
        except Exception as e:
            return {
                'person_id': person.id,
                'risk_score': 0.5,
                'risk_level': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def predict_batch(self, persons: List[PersonRecord]) -> List[Dict]:
        """Predict displacement risk for multiple persons."""
        return [self.predict_person(p) for p in persons]
    
    def _generate_explanation(self, features: Dict, risk_score: float) -> str:
        """Generate human-readable explanation of risk factors."""
        factors = []
        
        # Industry risk
        if features['current_climate_sensitive']:
            factors.append("works in climate-sensitive industry")
        
        # Geographic risk
        if features['current_coastal']:
            factors.append("located in coastal county")
        
        # Job stability
        if features['job_stability_score'] < 12:
            factors.append("low job tenure")
        elif features['recent_transitions'] > 2:
            factors.append("frequent job changes")
        
        # History of displacement
        if features['left_coastal_county']:
            factors.append("previously left coastal area")
        
        # Seniority protection
        if features['is_senior'] or features['is_manager']:
            factors.append("senior/management role (protective)")
        
        if not factors:
            return "Average displacement risk based on job profile"
        
        return f"Risk factors: {', '.join(factors[:3])}"
    
    def get_model_info(self) -> Dict:
        """Get information about loaded model."""
        if not self.model:
            return {'loaded': False}
        
        return {
            'loaded': True,
            'features': self.model.feature_names,
            'n_features': len(self.model.feature_names),
            'metrics': self.model.training_metrics,
        }


# Global predictor instance (lazy loaded)
_predictor_instance: Optional[DisplacementPredictor] = None


def get_predictor(model_path: Optional[str] = None) -> DisplacementPredictor:
    """
    Get or create global predictor instance.
    
    Args:
        model_path: Path to model file (only used on first call)
        
    Returns:
        DisplacementPredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        _predictor_instance = DisplacementPredictor(model_path)
    
    return _predictor_instance


def predict_displacement_risk(
    person: PersonRecord,
    model_path: Optional[str] = None
) -> Dict:
    """
    Convenience function to predict displacement risk.
    
    Args:
        person: PersonRecord to score
        model_path: Optional path to model file
        
    Returns:
        Risk prediction dict
    """
    predictor = get_predictor(model_path)
    return predictor.predict_person(person)
