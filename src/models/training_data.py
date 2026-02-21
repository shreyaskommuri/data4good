"""
Training Data for Job Title Climate Classification

Curated dataset of job titles with ground truth labels for:
- Training classifiers
- Evaluating model performance
- Benchmarking prompt configurations

Categories:
- climate_sensitive: Jobs directly impacted by weather/climate
- climate_resilient: Jobs with minimal climate disruption

Author: Coastal Labor-Resilience Engine Team
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class LabeledJobTitle:
    """A job title with ground truth label."""
    title: str
    category: str  # "climate_sensitive" or "climate_resilient"
    industry: str
    reasoning: str
    confidence: float = 1.0  # 1.0 = certain, <1.0 = ambiguous


# ============================================================================
# Climate-Sensitive Job Titles (Weather/Outdoor Dependent)
# ============================================================================

CLIMATE_SENSITIVE_JOBS: List[LabeledJobTitle] = [
    # Marine/Fishing
    LabeledJobTitle(
        title="Commercial Fisherman",
        category="climate_sensitive",
        industry="Fishing",
        reasoning="Directly depends on ocean conditions, weather windows, and marine ecosystems",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Fishing Boat Captain",
        category="climate_sensitive",
        industry="Fishing",
        reasoning="Cannot operate in storms, depends on fish migration patterns",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Deckhand",
        category="climate_sensitive",
        industry="Fishing",
        reasoning="Works on fishing vessels, weather-dependent",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Lobster Fisherman",
        category="climate_sensitive",
        industry="Fishing",
        reasoning="Marine-based work affected by storms and ocean conditions",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Oyster Farmer",
        category="climate_sensitive",
        industry="Aquaculture",
        reasoning="Aquaculture affected by water temperature, storm surges",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Marine Biologist - Field Research",
        category="climate_sensitive",
        industry="Marine Services",
        reasoning="Field work dependent on ocean access and conditions",
        confidence=0.9
    ),
    
    # Beach/Coastal Tourism
    LabeledJobTitle(
        title="Surf Instructor",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Requires beach access and suitable wave conditions",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Beach Lifeguard",
        category="climate_sensitive",
        industry="Public Safety",
        reasoning="Cannot work during storms, beach closures",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Kayak Tour Guide",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Water-based tourism directly affected by weather",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Whale Watching Captain",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Cannot operate in rough seas or poor visibility",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Scuba Diving Instructor",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Diving conditions dependent on weather and visibility",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Parasailing Operator",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Wind-dependent, cannot operate in storms",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Beach Resort Activities Coordinator",
        category="climate_sensitive",
        industry="Hospitality",
        reasoning="Outdoor activities canceled during inclement weather",
        confidence=0.9
    ),
    
    # Agriculture
    LabeledJobTitle(
        title="Vineyard Manager",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Crop health directly affected by weather, frost, heat waves",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Farm Equipment Operator",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Cannot work fields in rain, depends on growing conditions",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Strawberry Picker",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Harvest timing critical, weather-dependent",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Ranch Hand",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Outdoor livestock work in all conditions",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Agricultural Irrigation Specialist",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Work affected by drought, rainfall patterns",
        confidence=0.95
    ),
    LabeledJobTitle(
        title="Citrus Grove Manager",
        category="climate_sensitive",
        industry="Agriculture",
        reasoning="Frost damage, drought impact crop management",
        confidence=1.0
    ),
    
    # Construction/Outdoor Labor
    LabeledJobTitle(
        title="Roofer",
        category="climate_sensitive",
        industry="Construction",
        reasoning="Cannot work in rain, high winds, or extreme heat",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Road Construction Worker",
        category="climate_sensitive",
        industry="Construction",
        reasoning="Paving requires dry conditions, work stops in rain",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Utility Line Worker",
        category="climate_sensitive",
        industry="Utilities",
        reasoning="Outdoor work, emergency response during storms",
        confidence=0.9
    ),
    LabeledJobTitle(
        title="Landscaper",
        category="climate_sensitive",
        industry="Services",
        reasoning="Outdoor work affected by rain, extreme temperatures",
        confidence=0.95
    ),
    LabeledJobTitle(
        title="Tree Trimmer",
        category="climate_sensitive",
        industry="Services",
        reasoning="Cannot work in wind, rain, or storms",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Golf Course Groundskeeper",
        category="climate_sensitive",
        industry="Recreation",
        reasoning="Outdoor maintenance affected by weather",
        confidence=0.95
    ),
    
    # Oil/Gas/Energy
    LabeledJobTitle(
        title="Offshore Oil Rig Worker",
        category="climate_sensitive",
        industry="Oil and Gas",
        reasoning="Platform evacuations during hurricanes/storms",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Solar Panel Installer",
        category="climate_sensitive",
        industry="Energy",
        reasoning="Rooftop work affected by weather conditions",
        confidence=0.9
    ),
    LabeledJobTitle(
        title="Wind Turbine Technician",
        category="climate_sensitive",
        industry="Energy",
        reasoning="Cannot climb turbines in high winds or storms",
        confidence=0.95
    ),
    
    # Transportation
    LabeledJobTitle(
        title="Ferry Captain",
        category="climate_sensitive",
        industry="Transportation",
        reasoning="Service suspended during storms and high seas",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Harbor Pilot",
        category="climate_sensitive",
        industry="Transportation",
        reasoning="Ship guidance affected by visibility, weather",
        confidence=0.95
    ),
    
    # Recreation/Sports
    LabeledJobTitle(
        title="Ski Instructor",
        category="climate_sensitive",
        industry="Recreation",
        reasoning="Depends on snowfall and ski resort operations",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="River Rafting Guide",
        category="climate_sensitive",
        industry="Tourism",
        reasoning="Water levels depend on rainfall, dam releases",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Outdoor Photographer",
        category="climate_sensitive",
        industry="Creative",
        reasoning="Shoots affected by lighting, weather conditions",
        confidence=0.85
    ),
]


# ============================================================================
# Climate-Resilient Job Titles (Indoor/Remote/Weather-Independent)
# ============================================================================

CLIMATE_RESILIENT_JOBS: List[LabeledJobTitle] = [
    # Technology/Software
    LabeledJobTitle(
        title="Software Engineer",
        category="climate_resilient",
        industry="Technology",
        reasoning="Indoor work, easily done remotely",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Data Scientist",
        category="climate_resilient",
        industry="Technology",
        reasoning="Computer-based analysis, remote-capable",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="DevOps Engineer",
        category="climate_resilient",
        industry="Technology",
        reasoning="Cloud infrastructure management, remote work",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Cybersecurity Analyst",
        category="climate_resilient",
        industry="Technology",
        reasoning="Digital security monitoring, fully remote",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Machine Learning Engineer",
        category="climate_resilient",
        industry="Technology",
        reasoning="Model development done on computers",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Web Developer",
        category="climate_resilient",
        industry="Technology",
        reasoning="Indoor coding work, remote-friendly",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Database Administrator",
        category="climate_resilient",
        industry="Technology",
        reasoning="System management from any location",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="UX Designer",
        category="climate_resilient",
        industry="Technology",
        reasoning="Design work on computer, no outdoor component",
        confidence=1.0
    ),
    
    # Remote/Virtual Work
    LabeledJobTitle(
        title="Remote Project Manager",
        category="climate_resilient",
        industry="Professional Services",
        reasoning="Coordination via digital tools, location-independent",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Virtual Assistant",
        category="climate_resilient",
        industry="Services",
        reasoning="Administrative support from anywhere",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Remote Customer Support Specialist",
        category="climate_resilient",
        industry="Services",
        reasoning="Phone/chat support from home office",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Telehealth Nurse",
        category="climate_resilient",
        industry="Healthcare",
        reasoning="Patient consultations via video",
        confidence=0.95
    ),
    
    # Finance/Professional Services
    LabeledJobTitle(
        title="Accountant",
        category="climate_resilient",
        industry="Finance",
        reasoning="Indoor office work, can be done remotely",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Financial Analyst",
        category="climate_resilient",
        industry="Finance",
        reasoning="Data analysis and reporting, office-based",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Tax Preparer",
        category="climate_resilient",
        industry="Finance",
        reasoning="Desk work, no weather dependency",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Insurance Underwriter",
        category="climate_resilient",
        industry="Insurance",
        reasoning="Risk assessment done at desk",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Actuary",
        category="climate_resilient",
        industry="Insurance",
        reasoning="Statistical analysis, fully indoor",
        confidence=1.0
    ),
    
    # Legal
    LabeledJobTitle(
        title="Corporate Lawyer",
        category="climate_resilient",
        industry="Legal",
        reasoning="Office and courtroom work, indoor",
        confidence=0.95
    ),
    LabeledJobTitle(
        title="Paralegal",
        category="climate_resilient",
        industry="Legal",
        reasoning="Document preparation, office-based",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Legal Document Reviewer",
        category="climate_resilient",
        industry="Legal",
        reasoning="Computer-based document analysis",
        confidence=1.0
    ),
    
    # Healthcare (Indoor)
    LabeledJobTitle(
        title="Radiologist",
        category="climate_resilient",
        industry="Healthcare",
        reasoning="Image analysis in hospital/office, can be remote",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Medical Records Technician",
        category="climate_resilient",
        industry="Healthcare",
        reasoning="Administrative health records work",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Pharmacist",
        category="climate_resilient",
        industry="Healthcare",
        reasoning="Indoor pharmacy work",
        confidence=0.95
    ),
    LabeledJobTitle(
        title="Clinical Psychologist",
        category="climate_resilient",
        industry="Healthcare",
        reasoning="Office-based therapy, can do teletherapy",
        confidence=1.0
    ),
    
    # Education (Indoor)
    LabeledJobTitle(
        title="University Professor",
        category="climate_resilient",
        industry="Education",
        reasoning="Classroom teaching, can switch to online",
        confidence=0.9
    ),
    LabeledJobTitle(
        title="Online Course Instructor",
        category="climate_resilient",
        industry="Education",
        reasoning="Fully virtual teaching",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Librarian",
        category="climate_resilient",
        industry="Education",
        reasoning="Indoor library work",
        confidence=1.0
    ),
    
    # Marketing/Creative (Indoor)
    LabeledJobTitle(
        title="Digital Marketing Manager",
        category="climate_resilient",
        industry="Marketing",
        reasoning="Online campaign management, remote-capable",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Content Writer",
        category="climate_resilient",
        industry="Media",
        reasoning="Writing work from anywhere",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Graphic Designer",
        category="climate_resilient",
        industry="Creative",
        reasoning="Computer-based design work",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Video Editor",
        category="climate_resilient",
        industry="Media",
        reasoning="Post-production work on computer",
        confidence=1.0
    ),
    
    # Administrative/Office
    LabeledJobTitle(
        title="HR Manager",
        category="climate_resilient",
        industry="Services",
        reasoning="Office-based personnel management",
        confidence=0.95
    ),
    LabeledJobTitle(
        title="Executive Assistant",
        category="climate_resilient",
        industry="Services",
        reasoning="Administrative support, office/remote",
        confidence=1.0
    ),
    LabeledJobTitle(
        title="Call Center Representative",
        category="climate_resilient",
        industry="Services",
        reasoning="Phone support from office or home",
        confidence=1.0
    ),
]


# ============================================================================
# Ambiguous Job Titles (Context-Dependent)
# ============================================================================

AMBIGUOUS_JOBS: List[LabeledJobTitle] = [
    LabeledJobTitle(
        title="Restaurant Server",
        category="climate_sensitive",  # Beach/outdoor restaurants affected
        industry="Hospitality",
        reasoning="Depends on restaurant type - outdoor/beach restaurants are sensitive",
        confidence=0.6
    ),
    LabeledJobTitle(
        title="Hotel Concierge",
        category="climate_resilient",
        industry="Hospitality",
        reasoning="Indoor work, but coastal hotels may close during storms",
        confidence=0.7
    ),
    LabeledJobTitle(
        title="Delivery Driver",
        category="climate_sensitive",
        industry="Transportation",
        reasoning="Must drive in weather, but still works in most conditions",
        confidence=0.65
    ),
    LabeledJobTitle(
        title="Real Estate Agent",
        category="climate_resilient",
        industry="Real Estate",
        reasoning="Property showings affected by weather, but mostly resilient",
        confidence=0.75
    ),
    LabeledJobTitle(
        title="Event Photographer",
        category="climate_sensitive",
        industry="Creative",
        reasoning="Outdoor events canceled, but indoor events continue",
        confidence=0.7
    ),
    LabeledJobTitle(
        title="Uber Driver",
        category="climate_sensitive",
        industry="Transportation",
        reasoning="Driving affected by severe weather",
        confidence=0.6
    ),
    LabeledJobTitle(
        title="Yoga Instructor",
        category="climate_resilient",
        industry="Wellness",
        reasoning="Usually indoor, can teach online, but outdoor classes affected",
        confidence=0.75
    ),
    LabeledJobTitle(
        title="Wedding Planner",
        category="climate_sensitive",
        industry="Events",
        reasoning="Outdoor weddings heavily weather-dependent",
        confidence=0.7
    ),
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_all_training_data() -> List[LabeledJobTitle]:
    """Get all labeled job titles."""
    return CLIMATE_SENSITIVE_JOBS + CLIMATE_RESILIENT_JOBS + AMBIGUOUS_JOBS


def get_training_df() -> pd.DataFrame:
    """Get training data as a DataFrame."""
    data = get_all_training_data()
    return pd.DataFrame([
        {
            "title": j.title,
            "category": j.category,
            "industry": j.industry,
            "reasoning": j.reasoning,
            "confidence": j.confidence
        }
        for j in data
    ])


def get_evaluation_split(
    test_ratio: float = 0.2,
    include_ambiguous: bool = False,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Split data into train/test sets.
    
    Returns:
        (train_titles, train_labels, test_titles, test_labels)
    """
    import random
    random.seed(random_seed)
    
    # Get high-confidence data
    if include_ambiguous:
        data = get_all_training_data()
    else:
        data = [j for j in get_all_training_data() if j.confidence >= 0.9]
    
    # Shuffle
    random.shuffle(data)
    
    # Split
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    train_titles = [j.title for j in train_data]
    train_labels = [j.category for j in train_data]
    test_titles = [j.title for j in test_data]
    test_labels = [j.category for j in test_data]
    
    return train_titles, train_labels, test_titles, test_labels


def get_benchmark_dataset() -> Tuple[List[str], List[str]]:
    """
    Get a fixed benchmark dataset for comparing models.
    
    Returns:
        (titles, labels) with balanced classes
    """
    # Use equal samples from each class
    n_per_class = min(
        len(CLIMATE_SENSITIVE_JOBS),
        len(CLIMATE_RESILIENT_JOBS)
    )
    
    sensitive = CLIMATE_SENSITIVE_JOBS[:n_per_class]
    resilient = CLIMATE_RESILIENT_JOBS[:n_per_class]
    
    all_jobs = sensitive + resilient
    
    return (
        [j.title for j in all_jobs],
        [j.category for j in all_jobs]
    )


# ============================================================================
# Main - Show Training Data Statistics
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Job Title Climate Classification - Training Data")
    print("=" * 60)
    
    df = get_training_df()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total labeled titles: {len(df)}")
    print(f"  Climate Sensitive: {len(CLIMATE_SENSITIVE_JOBS)}")
    print(f"  Climate Resilient: {len(CLIMATE_RESILIENT_JOBS)}")
    print(f"  Ambiguous: {len(AMBIGUOUS_JOBS)}")
    
    print(f"\n📈 By Industry:")
    print(df['industry'].value_counts().head(10).to_string())
    
    print(f"\n🎯 Sample Entries:")
    print(df[['title', 'category', 'confidence']].head(10).to_string(index=False))
    
    # Train/test split
    train_t, train_l, test_t, test_l = get_evaluation_split()
    print(f"\n✂️  Train/Test Split:")
    print(f"  Train: {len(train_t)} titles")
    print(f"  Test: {len(test_t)} titles")
    
    # Benchmark dataset
    bench_t, bench_l = get_benchmark_dataset()
    print(f"\n🏆 Benchmark Dataset:")
    print(f"  {len(bench_t)} balanced samples ({len(bench_t)//2} per class)")
