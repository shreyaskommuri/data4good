"""
Markov Transition Matrix for Labor Category Dynamics

This module models workforce transitions between labor states
following climate shocks using discrete-time Markov chains.

States:
- COASTAL: Jobs directly dependent on coastal conditions
- INLAND: Jobs not dependent on coastal/weather conditions  
- UNEMPLOYED: Out of workforce
- TRANSITIONING: In job search/retraining

Theory:
- P_ij represents probability of moving from state i to state j
- Climate shocks modify transition probabilities
- Steady-state distribution indicates long-term workforce composition
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
from numpy.typing import NDArray


class LaborState(Enum):
    """Labor market states for Markov chain."""
    COASTAL = "coastal"           # Climate-sensitive coastal jobs
    INLAND = "inland"             # Climate-resilient inland jobs
    UNEMPLOYED = "unemployed"     # Out of workforce
    TRANSITIONING = "transitioning"  # Active job search/retraining


# State indices for matrix operations
STATE_INDEX = {
    LaborState.COASTAL: 0,
    LaborState.INLAND: 1,
    LaborState.UNEMPLOYED: 2,
    LaborState.TRANSITIONING: 3,
}

INDEX_STATE = {v: k for k, v in STATE_INDEX.items()}


@dataclass
class TransitionData:
    """Raw transition counts from job-change data."""
    from_state: LaborState
    to_state: LaborState
    count: int
    climate_shock: bool = False
    shock_severity: float = 0.0  # 0-1 scale


@dataclass
class ShockParameters:
    """Parameters defining a climate shock event."""
    severity: float  # 0-1 scale (0=minor, 1=catastrophic)
    duration_days: int
    affected_industries: List[str] = field(default_factory=list)
    geographic_radius_km: float = 50.0
    
    def __post_init__(self):
        self.severity = max(0.0, min(1.0, self.severity))


@dataclass
class MarkovAnalysisResult:
    """Results from Markov chain analysis."""
    transition_matrix: NDArray[np.float64]
    steady_state: NDArray[np.float64]
    mean_first_passage: NDArray[np.float64]
    entropy_rate: float
    mixing_time: int
    eigenvalues: NDArray[np.complex128]
    is_ergodic: bool


class MarkovTransitionMatrix:
    """
    Markov chain model for labor market transitions.
    
    Models workforce dynamics across labor states, with support
    for climate shock modifications to transition probabilities.
    """
    
    # Default baseline transition probabilities (no shock)
    # Rows: from_state, Columns: to_state
    # Order: COASTAL, INLAND, UNEMPLOYED, TRANSITIONING
    DEFAULT_BASELINE = np.array([
        [0.85, 0.08, 0.04, 0.03],  # From COASTAL
        [0.05, 0.88, 0.03, 0.04],  # From INLAND
        [0.10, 0.15, 0.55, 0.20],  # From UNEMPLOYED
        [0.20, 0.35, 0.10, 0.35],  # From TRANSITIONING
    ])
    
    # Shock modifier matrix (additive changes during shock)
    # Increases coastal→unemployed, decreases stability
    DEFAULT_SHOCK_MODIFIER = np.array([
        [-0.25, 0.05, 0.15, 0.05],   # COASTAL becomes unstable
        [0.02, -0.05, 0.02, 0.01],   # INLAND slightly affected
        [0.00, -0.05, 0.05, 0.00],   # Harder to leave unemployment
        [-0.05, -0.10, 0.05, 0.10],  # Harder to find new jobs
    ])
    
    def __init__(
        self,
        baseline_matrix: Optional[NDArray[np.float64]] = None,
        shock_modifier: Optional[NDArray[np.float64]] = None,
    ):
        """
        Initialize Markov transition matrix.
        
        Args:
            baseline_matrix: 4x4 transition probability matrix (default provided)
            shock_modifier: 4x4 additive modifier during climate shocks
        """
        self.n_states = len(LaborState)
        
        # Initialize matrices
        if baseline_matrix is not None:
            self._validate_matrix(baseline_matrix)
            self.baseline = baseline_matrix.copy()
        else:
            self.baseline = self.DEFAULT_BASELINE.copy()
            
        if shock_modifier is not None:
            self._validate_modifier(shock_modifier)
            self.shock_modifier = shock_modifier.copy()
        else:
            self.shock_modifier = self.DEFAULT_SHOCK_MODIFIER.copy()
            
        # Current active matrix
        self.current_matrix = self.baseline.copy()
        
    def _validate_matrix(self, matrix: NDArray[np.float64]) -> None:
        """Validate transition probability matrix."""
        if matrix.shape != (self.n_states, self.n_states):
            raise ValueError(f"Matrix must be {self.n_states}x{self.n_states}")
        if not np.allclose(matrix.sum(axis=1), 1.0):
            raise ValueError("Rows must sum to 1 (stochastic matrix)")
        if np.any(matrix < 0) or np.any(matrix > 1):
            raise ValueError("Probabilities must be in [0, 1]")
            
    def _validate_modifier(self, modifier: NDArray[np.float64]) -> None:
        """Validate shock modifier matrix."""
        if modifier.shape != (self.n_states, self.n_states):
            raise ValueError(f"Modifier must be {self.n_states}x{self.n_states}")
        if not np.allclose(modifier.sum(axis=1), 0.0, atol=0.01):
            raise ValueError("Modifier rows should sum to ~0 (redistribute probability)")
            
    def apply_shock(self, shock: ShockParameters) -> NDArray[np.float64]:
        """
        Apply climate shock to transition matrix.
        
        Args:
            shock: ShockParameters defining the event
            
        Returns:
            Modified transition matrix
        """
        # Scale modifier by severity
        scaled_modifier = self.shock_modifier * shock.severity
        
        # Apply modifier
        modified = self.baseline + scaled_modifier
        
        # Ensure valid probabilities
        modified = np.clip(modified, 0.001, 0.999)
        
        # Renormalize rows to sum to 1
        modified = modified / modified.sum(axis=1, keepdims=True)
        
        self.current_matrix = modified
        return modified
        
    def reset_to_baseline(self) -> None:
        """Reset to baseline (non-shock) matrix."""
        self.current_matrix = self.baseline.copy()
        
    def from_transition_data(
        self,
        transitions: List[TransitionData],
        smoothing: float = 1.0,
    ) -> NDArray[np.float64]:
        """
        Estimate transition matrix from observed job-change data.
        
        Args:
            transitions: List of observed transitions
            smoothing: Laplace smoothing parameter (default=1)
            
        Returns:
            Estimated transition matrix
        """
        # Initialize count matrix with smoothing
        counts = np.full((self.n_states, self.n_states), smoothing)
        
        # Accumulate counts
        for t in transitions:
            i = STATE_INDEX[t.from_state]
            j = STATE_INDEX[t.to_state]
            counts[i, j] += t.count
            
        # Convert to probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        matrix = counts / row_sums
        
        self.baseline = matrix
        self.current_matrix = matrix.copy()
        return matrix
        
    def get_transition_prob(
        self,
        from_state: LaborState,
        to_state: LaborState,
    ) -> float:
        """Get probability of transitioning between specific states."""
        i = STATE_INDEX[from_state]
        j = STATE_INDEX[to_state]
        return float(self.current_matrix[i, j])
        
    def simulate_trajectory(
        self,
        initial_state: LaborState,
        n_steps: int,
        shock_schedule: Optional[Dict[int, ShockParameters]] = None,
        random_seed: Optional[int] = None,
    ) -> List[LaborState]:
        """
        Simulate a single worker's trajectory through states.
        
        Args:
            initial_state: Starting labor state
            n_steps: Number of time steps to simulate
            shock_schedule: Dict mapping timestep → shock parameters
            random_seed: Random seed for reproducibility
            
        Returns:
            List of states visited
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        shock_schedule = shock_schedule or {}
        trajectory = [initial_state]
        current_state = initial_state
        
        for t in range(n_steps):
            # Apply shock if scheduled
            if t in shock_schedule:
                self.apply_shock(shock_schedule[t])
            elif t - 1 in shock_schedule:
                # Reset after shock (simplified recovery)
                self.reset_to_baseline()
                
            # Get transition probabilities from current state
            i = STATE_INDEX[current_state]
            probs = self.current_matrix[i]
            
            # Sample next state
            next_idx = np.random.choice(self.n_states, p=probs)
            current_state = INDEX_STATE[next_idx]
            trajectory.append(current_state)
            
        return trajectory
        
    def simulate_population(
        self,
        initial_distribution: NDArray[np.float64],
        n_steps: int,
        shock_schedule: Optional[Dict[int, ShockParameters]] = None,
    ) -> NDArray[np.float64]:
        """
        Simulate population distribution evolution over time.
        
        Args:
            initial_distribution: Starting distribution over states [4,]
            n_steps: Number of time steps
            shock_schedule: Dict mapping timestep → shock parameters
            
        Returns:
            Distribution history [n_steps+1, 4]
        """
        shock_schedule = shock_schedule or {}
        
        # Validate initial distribution
        initial = np.array(initial_distribution, dtype=np.float64)
        if initial.shape != (self.n_states,):
            raise ValueError(f"Initial distribution must be [{self.n_states}]")
        initial = initial / initial.sum()  # Normalize
        
        # Track distribution over time
        history = np.zeros((n_steps + 1, self.n_states))
        history[0] = initial
        current = initial.copy()
        
        for t in range(n_steps):
            # Apply shock if scheduled
            if t in shock_schedule:
                self.apply_shock(shock_schedule[t])
            elif t - 1 in shock_schedule:
                self.reset_to_baseline()
                
            # Evolution: π(t+1) = π(t) @ P
            current = current @ self.current_matrix
            history[t + 1] = current
            
        return history
        
    def compute_steady_state(
        self,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Compute stationary distribution (eigenvector for eigenvalue 1).
        
        Args:
            matrix: Transition matrix (uses current_matrix if None)
            
        Returns:
            Steady-state distribution [4,]
        """
        P = matrix if matrix is not None else self.current_matrix
        
        # Solve π = πP ⟺ π(P - I) = 0 with constraint Σπ = 1
        # Equivalent: (P^T - I)^T π^T = 0
        A = P.T - np.eye(self.n_states)
        
        # Add normalization constraint
        A = np.vstack([A, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1.0
        
        # Least squares solution
        steady_state, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        # Ensure valid probability distribution
        steady_state = np.clip(steady_state, 0, 1)
        steady_state = steady_state / steady_state.sum()
        
        return steady_state
        
    def compute_mean_first_passage(
        self,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        """
        Compute mean first passage times between all state pairs.
        
        M_ij = expected steps to reach state j starting from state i
        
        Returns:
            Mean first passage time matrix [4, 4]
        """
        P = matrix if matrix is not None else self.current_matrix
        n = self.n_states
        
        # Compute fundamental matrix Z = (I - P + W)^(-1)
        # where W is matrix with steady-state in all rows
        steady_state = self.compute_steady_state(P)
        W = np.tile(steady_state, (n, 1))
        
        try:
            Z = np.linalg.inv(np.eye(n) - P + W)
        except np.linalg.LinAlgError:
            # Matrix is singular, return inf
            return np.full((n, n), np.inf)
            
        # Mean first passage: M_ij = (Z_jj - Z_ij) / π_j
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i == j:
                    M[i, j] = 0.0
                elif steady_state[j] > 1e-10:
                    M[i, j] = (Z[j, j] - Z[i, j]) / steady_state[j]
                else:
                    M[i, j] = np.inf
                    
        return M
        
    def compute_entropy_rate(
        self,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> float:
        """
        Compute entropy rate of the Markov chain.
        
        H = -Σ_i π_i Σ_j P_ij log(P_ij)
        
        Returns:
            Entropy rate in bits
        """
        P = matrix if matrix is not None else self.current_matrix
        steady_state = self.compute_steady_state(P)
        
        entropy = 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                if P[i, j] > 1e-10:
                    entropy -= steady_state[i] * P[i, j] * np.log2(P[i, j])
                    
        return entropy
        
    def compute_mixing_time(
        self,
        epsilon: float = 0.01,
        max_steps: int = 1000,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> int:
        """
        Estimate mixing time (steps to reach near-equilibrium).
        
        Args:
            epsilon: Distance threshold from steady-state
            max_steps: Maximum steps to check
            matrix: Transition matrix
            
        Returns:
            Mixing time in steps
        """
        P = matrix if matrix is not None else self.current_matrix
        steady_state = self.compute_steady_state(P)
        
        # Start from worst-case distribution (concentrated in one state)
        P_power = P.copy()
        
        for t in range(1, max_steps + 1):
            # Check if all rows are close to steady-state
            max_deviation = 0.0
            for i in range(self.n_states):
                deviation = np.abs(P_power[i] - steady_state).sum() / 2
                max_deviation = max(max_deviation, deviation)
                
            if max_deviation < epsilon:
                return t
                
            P_power = P_power @ P
            
        return max_steps
        
    def is_ergodic(
        self,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> bool:
        """Check if Markov chain is ergodic (irreducible and aperiodic)."""
        P = matrix if matrix is not None else self.current_matrix
        n = self.n_states
        
        # Check irreducibility: can reach all states from all states
        # Use matrix power to check reachability
        reach = P.copy()
        for _ in range(n):
            reach = reach + reach @ P
        reach = (reach > 0).astype(float)
        
        if not np.all(reach > 0):
            return False
            
        # Check aperiodicity: GCD of return times is 1
        # Sufficient condition: self-loops exist
        if np.any(np.diag(P) > 0):
            return True
            
        # More thorough check using eigenvalues
        eigenvalues = np.linalg.eigvals(P)
        unit_eigenvalues = np.abs(np.abs(eigenvalues) - 1) < 1e-10
        return np.sum(unit_eigenvalues) == 1  # Only eigenvalue 1 on unit circle
        
    def full_analysis(
        self,
        matrix: Optional[NDArray[np.float64]] = None,
    ) -> MarkovAnalysisResult:
        """
        Perform complete Markov chain analysis.
        
        Returns:
            MarkovAnalysisResult with all computed metrics
        """
        P = matrix if matrix is not None else self.current_matrix
        
        return MarkovAnalysisResult(
            transition_matrix=P.copy(),
            steady_state=self.compute_steady_state(P),
            mean_first_passage=self.compute_mean_first_passage(P),
            entropy_rate=self.compute_entropy_rate(P),
            mixing_time=self.compute_mixing_time(matrix=P),
            eigenvalues=np.linalg.eigvals(P),
            is_ergodic=self.is_ergodic(P),
        )
        
    def compare_shock_impact(
        self,
        shock: ShockParameters,
    ) -> Dict[str, any]:
        """
        Compare baseline vs shock-modified chain properties.
        
        Returns:
            Dictionary with comparison metrics
        """
        baseline_analysis = self.full_analysis(self.baseline)
        
        shock_matrix = self.apply_shock(shock)
        shock_analysis = self.full_analysis(shock_matrix)
        
        self.reset_to_baseline()
        
        return {
            "baseline": {
                "steady_state": baseline_analysis.steady_state.tolist(),
                "entropy_rate": baseline_analysis.entropy_rate,
                "mixing_time": baseline_analysis.mixing_time,
            },
            "shock": {
                "steady_state": shock_analysis.steady_state.tolist(),
                "entropy_rate": shock_analysis.entropy_rate,
                "mixing_time": shock_analysis.mixing_time,
            },
            "impact": {
                "steady_state_shift": (
                    shock_analysis.steady_state - baseline_analysis.steady_state
                ).tolist(),
                "entropy_change": (
                    shock_analysis.entropy_rate - baseline_analysis.entropy_rate
                ),
                "mixing_time_change": (
                    shock_analysis.mixing_time - baseline_analysis.mixing_time
                ),
                "coastal_unemployment_risk_increase": (
                    self.get_transition_prob(LaborState.COASTAL, LaborState.UNEMPLOYED)
                    - baseline_analysis.transition_matrix[
                        STATE_INDEX[LaborState.COASTAL],
                        STATE_INDEX[LaborState.UNEMPLOYED]
                    ]
                ),
            },
        }
        
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            "baseline": self.baseline.tolist(),
            "shock_modifier": self.shock_modifier.tolist(),
            "current_matrix": self.current_matrix.tolist(),
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> "MarkovTransitionMatrix":
        """Deserialize from dictionary."""
        instance = cls(
            baseline_matrix=np.array(data["baseline"]),
            shock_modifier=np.array(data["shock_modifier"]),
        )
        instance.current_matrix = np.array(data["current_matrix"])
        return instance
        
    def __repr__(self) -> str:
        steady = self.compute_steady_state()
        return (
            f"MarkovTransitionMatrix(\n"
            f"  states={[s.value for s in LaborState]},\n"
            f"  steady_state=[{', '.join(f'{s:.3f}' for s in steady)}],\n"
            f"  ergodic={self.is_ergodic()}\n"
            f")"
        )


def create_regional_chain(
    region_type: str = "coastal",
    base_unemployment_rate: float = 0.05,
) -> MarkovTransitionMatrix:
    """
    Create a pre-configured Markov chain for a region type.
    
    Args:
        region_type: "coastal", "mixed", or "inland"
        base_unemployment_rate: Regional unemployment rate
        
    Returns:
        Configured MarkovTransitionMatrix
    """
    # Adjust baseline based on region type
    if region_type == "coastal":
        # Higher coastal employment, more vulnerable to shocks
        baseline = np.array([
            [0.82, 0.10, 0.05, 0.03],
            [0.08, 0.85, 0.03, 0.04],
            [0.15, 0.10, 0.50, 0.25],
            [0.25, 0.30, 0.10, 0.35],
        ])
        shock_mod = np.array([
            [-0.30, 0.08, 0.17, 0.05],
            [0.03, -0.06, 0.02, 0.01],
            [-0.05, -0.03, 0.08, 0.00],
            [-0.08, -0.12, 0.08, 0.12],
        ])
    elif region_type == "inland":
        # More resilient to coastal shocks
        baseline = np.array([
            [0.80, 0.12, 0.04, 0.04],
            [0.03, 0.92, 0.02, 0.03],
            [0.08, 0.20, 0.52, 0.20],
            [0.15, 0.45, 0.08, 0.32],
        ])
        shock_mod = np.array([
            [-0.15, 0.08, 0.05, 0.02],
            [0.01, -0.02, 0.01, 0.00],
            [0.00, -0.02, 0.02, 0.00],
            [-0.02, -0.05, 0.02, 0.05],
        ])
    else:  # mixed
        baseline = MarkovTransitionMatrix.DEFAULT_BASELINE.copy()
        shock_mod = MarkovTransitionMatrix.DEFAULT_SHOCK_MODIFIER.copy()
        
    # Adjust for regional unemployment rate
    unemployment_factor = base_unemployment_rate / 0.05
    baseline[:, STATE_INDEX[LaborState.UNEMPLOYED]] *= unemployment_factor
    baseline = baseline / baseline.sum(axis=1, keepdims=True)
    
    return MarkovTransitionMatrix(
        baseline_matrix=baseline,
        shock_modifier=shock_mod,
    )
