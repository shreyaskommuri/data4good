"""
Job Title Climate Classification Module

Phase 2: NLP Labeling
Categorizes job titles as "Climate-Sensitive" vs "Climate-Resilient"
using multiple classification approaches.

Climate-Sensitive: Jobs directly impacted by weather/climate events
  Examples: Surf Instructor, Commercial Fisherman, Outdoor Guide
  
Climate-Resilient: Jobs with minimal climate disruption
  Examples: Software Engineer, Remote Project Manager, Data Analyst

Author: Coastal Labor-Resilience Engine Team
"""

import re
import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ClimateCategory(Enum):
    """Job climate sensitivity categories."""
    CLIMATE_SENSITIVE = "climate_sensitive"
    CLIMATE_RESILIENT = "climate_resilient"
    UNCERTAIN = "uncertain"


@dataclass
class ClassificationResult:
    """Result of job title classification."""
    title: str
    category: ClimateCategory
    confidence: float
    reasoning: str
    matched_keywords: List[str] = field(default_factory=list)
    model_used: str = "rule_based"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "category": self.category.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "matched_keywords": self.matched_keywords,
            "model_used": self.model_used
        }


# ============================================================================
# Keyword-Based Classification (Baseline)
# ============================================================================

# Keywords indicating climate-sensitive jobs
CLIMATE_SENSITIVE_KEYWORDS = {
    # Outdoor/Nature-dependent
    "outdoor": 0.8,
    "surf": 0.95,
    "fishing": 0.95,
    "fisherman": 0.95,
    "fishery": 0.9,
    "marine": 0.85,
    "ocean": 0.85,
    "beach": 0.9,
    "coastal": 0.85,
    "harbor": 0.8,
    "port": 0.7,
    "dock": 0.85,
    "boat": 0.9,
    "captain": 0.75,
    "deckhand": 0.95,
    "lifeguard": 0.95,
    "diving": 0.9,
    "scuba": 0.9,
    
    # Agriculture
    "farm": 0.9,
    "farmer": 0.95,
    "ranch": 0.9,
    "rancher": 0.95,
    "agriculture": 0.9,
    "agricultural": 0.9,
    "vineyard": 0.9,
    "winery": 0.8,
    "harvest": 0.9,
    "crop": 0.9,
    "irrigation": 0.85,
    "livestock": 0.9,
    "orchard": 0.9,
    
    # Tourism/Hospitality (outdoor-focused)
    "tour guide": 0.85,
    "hiking": 0.9,
    "kayak": 0.9,
    "sailing": 0.9,
    "charter": 0.85,
    "whale watching": 0.95,
    "ecotourism": 0.9,
    "adventure": 0.75,
    "campground": 0.9,
    "ski": 0.9,
    "snowboard": 0.9,
    "golf course": 0.8,
    "groundskeeper": 0.85,
    "landscap": 0.8,
    
    # Construction/Outdoor labor
    "construction worker": 0.8,
    "roofer": 0.85,
    "roofing": 0.85,
    "paving": 0.8,
    "road crew": 0.85,
    "lineman": 0.8,
    "utility worker": 0.75,
    "tree service": 0.85,
    "arborist": 0.85,
    
    # Emergency/Weather-dependent
    "firefighter": 0.7,
    "wildfire": 0.9,
    "emergency response": 0.7,
    "flood": 0.85,
    "storm": 0.85,
    
    # Transportation (outdoor)
    "delivery driver": 0.65,
    "truck driver": 0.6,
    "pilot": 0.6,
    "flight": 0.55,
    
    # Oil/Gas
    "oil rig": 0.85,
    "offshore": 0.9,
    "drilling": 0.8,
    "petroleum": 0.75,
}

# Keywords indicating climate-resilient jobs
CLIMATE_RESILIENT_KEYWORDS = {
    # Technology/Remote work
    "software": 0.95,
    "developer": 0.9,
    "engineer": 0.75,  # Could be field engineer
    "programmer": 0.95,
    "data scientist": 0.95,
    "data analyst": 0.9,
    "machine learning": 0.95,
    "ai ": 0.9,
    "artificial intelligence": 0.95,
    "cloud": 0.85,
    "devops": 0.9,
    "cybersecurity": 0.95,
    "it support": 0.85,
    "network admin": 0.85,
    "database": 0.9,
    "web developer": 0.95,
    "frontend": 0.95,
    "backend": 0.95,
    "full stack": 0.95,
    
    # Remote-friendly roles
    "remote": 0.9,
    "virtual": 0.85,
    "work from home": 0.95,
    "telecommute": 0.95,
    "distributed": 0.8,
    
    # Office/Professional services
    "accountant": 0.9,
    "accounting": 0.85,
    "lawyer": 0.9,
    "attorney": 0.9,
    "legal": 0.8,
    "paralegal": 0.9,
    "consultant": 0.8,
    "analyst": 0.75,
    "project manager": 0.8,
    "product manager": 0.85,
    "marketing": 0.75,
    "digital marketing": 0.9,
    "copywriter": 0.85,
    "content writer": 0.85,
    "editor": 0.85,
    "designer": 0.75,
    "ux": 0.9,
    "ui": 0.9,
    "graphic design": 0.85,
    
    # Healthcare (indoor)
    "nurse": 0.8,
    "doctor": 0.85,
    "physician": 0.85,
    "surgeon": 0.9,
    "pharmacist": 0.9,
    "therapist": 0.85,
    "psychologist": 0.9,
    "psychiatrist": 0.9,
    "dentist": 0.9,
    "radiologist": 0.95,
    "lab tech": 0.9,
    "medical record": 0.95,
    
    # Finance
    "banker": 0.9,
    "financial": 0.85,
    "investment": 0.85,
    "insurance": 0.8,
    "actuary": 0.95,
    "underwriter": 0.9,
    "loan officer": 0.9,
    
    # Education (indoor)
    "professor": 0.85,
    "teacher": 0.75,  # Some outdoor teachers
    "tutor": 0.9,
    "instructor": 0.6,  # Depends on type
    "librarian": 0.95,
    
    # Government/Admin
    "administrator": 0.85,
    "clerk": 0.9,
    "receptionist": 0.9,
    "secretary": 0.9,
    "hr ": 0.85,
    "human resources": 0.85,
    "recruiter": 0.85,
    
    # Retail (indoor)
    "cashier": 0.8,
    "retail": 0.7,
    "store manager": 0.75,
    "customer service": 0.8,
    "call center": 0.95,
}

# Industry modifiers
CLIMATE_SENSITIVE_INDUSTRIES = {
    "agriculture": 0.9,
    "fishing": 0.95,
    "tourism": 0.7,
    "hospitality": 0.6,
    "construction": 0.7,
    "transportation": 0.5,
    "oil and gas": 0.7,
    "marine services": 0.9,
    "outdoor recreation": 0.95,
}

CLIMATE_RESILIENT_INDUSTRIES = {
    "technology": 0.9,
    "software": 0.95,
    "finance": 0.85,
    "healthcare": 0.75,
    "professional services": 0.8,
    "education": 0.7,
    "government": 0.8,
    "insurance": 0.85,
}


class RuleBasedClassifier:
    """
    Keyword and rule-based job title classifier.
    
    Fast baseline classifier using pattern matching and
    weighted keyword scoring.
    """
    
    def __init__(
        self,
        sensitive_keywords: Optional[Dict[str, float]] = None,
        resilient_keywords: Optional[Dict[str, float]] = None,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize the rule-based classifier.
        
        Args:
            sensitive_keywords: Climate-sensitive keyword -> weight mapping
            resilient_keywords: Climate-resilient keyword -> weight mapping
            confidence_threshold: Minimum confidence for classification
        """
        self.sensitive_keywords = sensitive_keywords or CLIMATE_SENSITIVE_KEYWORDS
        self.resilient_keywords = resilient_keywords or CLIMATE_RESILIENT_KEYWORDS
        self.confidence_threshold = confidence_threshold
    
    def classify(
        self,
        title: str,
        industry: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a job title.
        
        Args:
            title: Job title to classify
            industry: Optional industry for context
            
        Returns:
            ClassificationResult with category and confidence
        """
        title_lower = title.lower().strip()
        
        # Score for each category
        sensitive_score = 0.0
        resilient_score = 0.0
        sensitive_matches = []
        resilient_matches = []
        
        # Check sensitive keywords
        for keyword, weight in self.sensitive_keywords.items():
            if keyword in title_lower:
                sensitive_score += weight
                sensitive_matches.append(keyword)
        
        # Check resilient keywords
        for keyword, weight in self.resilient_keywords.items():
            if keyword in title_lower:
                resilient_score += weight
                resilient_matches.append(keyword)
        
        # Apply industry modifier if provided
        if industry:
            industry_lower = industry.lower()
            for ind, modifier in CLIMATE_SENSITIVE_INDUSTRIES.items():
                if ind in industry_lower:
                    sensitive_score *= (1 + modifier * 0.3)
            for ind, modifier in CLIMATE_RESILIENT_INDUSTRIES.items():
                if ind in industry_lower:
                    resilient_score *= (1 + modifier * 0.3)
        
        # Normalize scores to confidence
        total_score = sensitive_score + resilient_score
        
        if total_score == 0:
            return ClassificationResult(
                title=title,
                category=ClimateCategory.UNCERTAIN,
                confidence=0.0,
                reasoning="No matching keywords found",
                matched_keywords=[],
                model_used="rule_based"
            )
        
        # Determine category
        if sensitive_score > resilient_score:
            confidence = sensitive_score / (sensitive_score + resilient_score + 0.5)
            category = ClimateCategory.CLIMATE_SENSITIVE
            matches = sensitive_matches
            reasoning = f"Matched sensitive keywords: {', '.join(matches)}"
        elif resilient_score > sensitive_score:
            confidence = resilient_score / (sensitive_score + resilient_score + 0.5)
            category = ClimateCategory.CLIMATE_RESILIENT
            matches = resilient_matches
            reasoning = f"Matched resilient keywords: {', '.join(matches)}"
        else:
            confidence = 0.5
            category = ClimateCategory.UNCERTAIN
            matches = sensitive_matches + resilient_matches
            reasoning = "Equal scores for both categories"
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            category = ClimateCategory.UNCERTAIN
        
        return ClassificationResult(
            title=title,
            category=category,
            confidence=min(confidence, 1.0),
            reasoning=reasoning,
            matched_keywords=matches,
            model_used="rule_based"
        )
    
    def classify_batch(
        self,
        titles: List[str],
        industries: Optional[List[str]] = None
    ) -> List[ClassificationResult]:
        """
        Classify multiple job titles.
        
        Args:
            titles: List of job titles
            industries: Optional list of corresponding industries
            
        Returns:
            List of ClassificationResults
        """
        if industries is None:
            industries = [None] * len(titles)
        
        return [
            self.classify(title, industry)
            for title, industry in zip(titles, industries)
        ]


class LLMClassifierConfig:
    """Configuration for LLM-based classification."""
    
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 150,
        prompt_template: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        
        self.prompt_template = prompt_template or self._default_prompt()
    
    def _default_prompt(self) -> str:
        return """You are a job classification expert analyzing climate impact on employment.

Classify the following job title into one of two categories:

1. CLIMATE_SENSITIVE: Jobs directly impacted by weather events, storms, or climate conditions
   - Outdoor work, agriculture, fishing, tourism, construction, transportation
   - Examples: Surf Instructor, Commercial Fisherman, Vineyard Manager, Lifeguard

2. CLIMATE_RESILIENT: Jobs with minimal disruption from weather or climate events  
   - Indoor work, remote-capable, technology, professional services
   - Examples: Software Engineer, Accountant, Remote Project Manager, Data Analyst

Job Title: {title}
{industry_context}

Respond with EXACTLY this JSON format:
{{"category": "CLIMATE_SENSITIVE" or "CLIMATE_RESILIENT", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""
    
    def format_prompt(self, title: str, industry: Optional[str] = None) -> str:
        """Format the prompt with job details."""
        industry_context = f"Industry: {industry}" if industry else ""
        return self.prompt_template.format(
            title=title,
            industry_context=industry_context
        )


class LLMClassifier:
    """
    LLM-based job title classifier using OpenAI API.
    
    Provides more nuanced classification for titles that
    rule-based methods struggle with.
    """
    
    def __init__(self, config: Optional[LLMClassifierConfig] = None):
        """
        Initialize the LLM classifier.
        
        Args:
            config: LLMClassifierConfig instance
        """
        self.config = config or LLMClassifierConfig()
        self._client = None
    
    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                import openai
                import os
                
                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OpenAI API key not configured")
                
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")
        
        return self._client
    
    def classify(
        self,
        title: str,
        industry: Optional[str] = None
    ) -> ClassificationResult:
        """
        Classify a job title using LLM.
        
        Args:
            title: Job title to classify
            industry: Optional industry context
            
        Returns:
            ClassificationResult
        """
        import json
        
        client = self._get_client()
        prompt = self.config.format_prompt(title, industry)
        
        try:
            response = client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": "You are a job classification expert."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            try:
                # Handle potential markdown code blocks
                if "```" in content:
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                
                result = json.loads(content)
                
                category_str = result.get("category", "").upper()
                if "SENSITIVE" in category_str:
                    category = ClimateCategory.CLIMATE_SENSITIVE
                elif "RESILIENT" in category_str:
                    category = ClimateCategory.CLIMATE_RESILIENT
                else:
                    category = ClimateCategory.UNCERTAIN
                
                return ClassificationResult(
                    title=title,
                    category=category,
                    confidence=float(result.get("confidence", 0.8)),
                    reasoning=result.get("reasoning", "LLM classification"),
                    model_used=f"llm:{self.config.model_name}"
                )
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response: {content}")
                return ClassificationResult(
                    title=title,
                    category=ClimateCategory.UNCERTAIN,
                    confidence=0.0,
                    reasoning=f"Parse error: {content[:100]}",
                    model_used=f"llm:{self.config.model_name}"
                )
                
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return ClassificationResult(
                title=title,
                category=ClimateCategory.UNCERTAIN,
                confidence=0.0,
                reasoning=f"API error: {str(e)}",
                model_used=f"llm:{self.config.model_name}"
            )
    
    def classify_batch(
        self,
        titles: List[str],
        industries: Optional[List[str]] = None
    ) -> List[ClassificationResult]:
        """Classify multiple titles (sequential for now)."""
        if industries is None:
            industries = [None] * len(titles)
        
        results = []
        for title, industry in zip(titles, industries):
            results.append(self.classify(title, industry))
        
        return results


class EnsembleClassifier:
    """
    Ensemble classifier combining multiple classification methods.
    
    Uses rule-based for fast baseline, with LLM for uncertain cases.
    """
    
    def __init__(
        self,
        rule_classifier: Optional[RuleBasedClassifier] = None,
        llm_classifier: Optional[LLMClassifier] = None,
        fallback_to_llm_threshold: float = 0.7
    ):
        """
        Initialize ensemble classifier.
        
        Args:
            rule_classifier: RuleBasedClassifier instance
            llm_classifier: LLMClassifier instance
            fallback_to_llm_threshold: Use LLM if rule confidence below this
        """
        self.rule_classifier = rule_classifier or RuleBasedClassifier()
        self.llm_classifier = llm_classifier
        self.fallback_threshold = fallback_to_llm_threshold
    
    def classify(
        self,
        title: str,
        industry: Optional[str] = None,
        force_llm: bool = False
    ) -> ClassificationResult:
        """
        Classify using ensemble approach.
        
        Args:
            title: Job title
            industry: Optional industry
            force_llm: Always use LLM
            
        Returns:
            ClassificationResult
        """
        # Get rule-based result first (fast)
        rule_result = self.rule_classifier.classify(title, industry)
        
        # If confident enough or no LLM available, return rule result
        if not force_llm:
            if rule_result.confidence >= self.fallback_threshold:
                return rule_result
            if self.llm_classifier is None:
                return rule_result
        
        # Fallback to LLM for uncertain cases
        if self.llm_classifier is not None:
            llm_result = self.llm_classifier.classify(title, industry)
            
            # If LLM is confident, use it
            if llm_result.confidence > rule_result.confidence:
                llm_result.reasoning = f"LLM: {llm_result.reasoning} (Rule: {rule_result.reasoning})"
                return llm_result
        
        return rule_result


def classify_job_titles(
    titles: List[str],
    industries: Optional[List[str]] = None,
    method: str = "rule"
) -> pd.DataFrame:
    """
    Convenience function to classify a list of job titles.
    
    Args:
        titles: List of job titles
        industries: Optional list of industries
        method: Classification method ('rule', 'llm', 'ensemble')
        
    Returns:
        DataFrame with classification results
    """
    if method == "rule":
        classifier = RuleBasedClassifier()
    elif method == "llm":
        classifier = LLMClassifier()
    elif method == "ensemble":
        classifier = EnsembleClassifier()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    results = classifier.classify_batch(titles, industries)
    
    return pd.DataFrame([r.to_dict() for r in results])


# ============================================================================
# Main - Test Classification
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Job Title Climate Classification")
    print("=" * 60)
    
    # Test titles
    test_titles = [
        "Surf Instructor",
        "Commercial Fisherman",
        "Vineyard Manager",
        "Beach Lifeguard",
        "Fishing Boat Captain",
        "Farm Equipment Operator",
        "Software Engineer",
        "Remote Project Manager",
        "Data Scientist",
        "Accountant",
        "Digital Marketing Manager",
        "Cybersecurity Analyst",
        "Restaurant Server",  # Ambiguous
        "Hotel Concierge",    # Ambiguous
        "Uber Driver",        # Moderate
    ]
    
    classifier = RuleBasedClassifier()
    
    print("\n📊 Classification Results:\n")
    print(f"{'Title':<30} {'Category':<20} {'Confidence':<12} {'Keywords'}")
    print("-" * 90)
    
    for title in test_titles:
        result = classifier.classify(title)
        keywords = ", ".join(result.matched_keywords[:3])
        print(f"{title:<30} {result.category.value:<20} {result.confidence:.2f}         {keywords}")
    
    # Summary stats
    results = classifier.classify_batch(test_titles)
    categories = [r.category for r in results]
    
    print("\n📈 Summary:")
    print(f"  Climate Sensitive: {categories.count(ClimateCategory.CLIMATE_SENSITIVE)}")
    print(f"  Climate Resilient: {categories.count(ClimateCategory.CLIMATE_RESILIENT)}")
    print(f"  Uncertain: {categories.count(ClimateCategory.UNCERTAIN)}")
