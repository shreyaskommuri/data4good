"""
Resilience ODE Solver for Labor Market Dynamics

This module implements the labor resilience differential equation:

    dL/dt = rL(1 - L/K) - β(EJ)

Where:
- L(t): Labor force participation at time t
- r: Intrinsic growth rate (job creation rate)
- K: Carrying capacity (maximum sustainable employment)
- β(EJ): Environmental justice friction coefficient

The β(EJ) term captures how environmental justice factors
(pollution burden, demographic vulnerability, infrastructure)
create "friction" that inhibits workforce recovery.

Theory:
- Logistic growth models natural labor market recovery
- EJ friction slows recovery in vulnerable communities
- Climate shocks can temporarily reduce K or increase β
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import odeint, solve_ivp
from scipy.optimize import minimize_scalar


@dataclass
class EJScreenData:
    """Environmental Justice Screen data for a census tract."""
    tract_id: str
    
    # Environmental indicators (percentiles 0-100)
    pm25_percentile: float = 50.0          # Particulate matter
    ozone_percentile: float = 50.0         # Ozone concentration
    traffic_percentile: float = 50.0       # Traffic proximity
    superfund_percentile: float = 50.0     # Superfund site proximity
    wastewater_percentile: float = 50.0    # Wastewater discharge
    
    # Demographic indicators (percentiles 0-100)
    low_income_percentile: float = 50.0    # Low income population
    minority_percentile: float = 50.0      # Minority population
    linguistic_iso_percentile: float = 50.0  # Linguistic isolation
    less_than_hs_percentile: float = 50.0    # Less than high school
    unemployment_percentile: float = 50.0     # Unemployment rate
    
    # Supplemental indices
    ej_index: float = 50.0                 # Combined EJ index (0-100)
    climate_vulnerability: float = 50.0    # Climate vulnerability score
    
    def compute_beta(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Compute β(EJ) friction coefficient from EJScreen data.
        
        Higher β = more friction = slower labor market recovery.
        
        Args:
            weights: Custom weights for each factor (default provided)
            
        Returns:
            β coefficient in range [0.01, 1.0]
        """
        default_weights = {
            "environmental": 0.3,
            "demographic": 0.4,
            "ej_index": 0.2,
            "climate": 0.1,
        }
        w = weights or default_weights
        
        # Environmental burden (average of indicators)
        env_burden = np.mean([
            self.pm25_percentile,
            self.ozone_percentile,
            self.traffic_percentile,
            self.superfund_percentile,
            self.wastewater_percentile,
        ]) / 100.0
        
        # Demographic vulnerability
        demo_vuln = np.mean([
            self.low_income_percentile,
            self.minority_percentile,
            self.linguistic_iso_percentile,
            self.less_than_hs_percentile,
            self.unemployment_percentile,
        ]) / 100.0
        
        # Combined score
        beta_raw = (
            w["environmental"] * env_burden +
            w["demographic"] * demo_vuln +
            w["ej_index"] * (self.ej_index / 100.0) +
            w["climate"] * (self.climate_vulnerability / 100.0)
        )
        
        # Scale to [0.01, 1.0] - never zero (always some friction)
        beta = 0.01 + 0.99 * beta_raw
        
        return beta


@dataclass 
class ODEParameters:
    """Parameters for the resilience ODE."""
    r: float = 0.1        # Growth rate (recovery speed)
    K: float = 1.0        # Carrying capacity (max employment ratio)
    beta: float = 0.05    # EJ friction coefficient
    
    # Optional time-varying parameters
    r_func: Optional[Callable[[float], float]] = None
    K_func: Optional[Callable[[float], float]] = None
    beta_func: Optional[Callable[[float], float]] = None
    
    def get_r(self, t: float) -> float:
        """Get growth rate at time t."""
        if self.r_func is not None:
            return self.r_func(t)
        return self.r
        
    def get_K(self, t: float) -> float:
        """Get carrying capacity at time t."""
        if self.K_func is not None:
            return self.K_func(t)
        return self.K
        
    def get_beta(self, t: float) -> float:
        """Get friction coefficient at time t."""
        if self.beta_func is not None:
            return self.beta_func(t)
        return self.beta


@dataclass
class ClimateShock:
    """Definition of a climate shock event."""
    start_time: float         # When shock begins
    duration: float           # How long shock lasts
    severity: float = 0.5     # 0-1 scale
    K_reduction: float = 0.3  # Fractional reduction in carrying capacity
    beta_increase: float = 0.2  # Additive increase in friction
    
    def is_active(self, t: float) -> bool:
        """Check if shock is active at time t."""
        return self.start_time <= t < self.start_time + self.duration
        
    def modify_params(
        self,
        params: ODEParameters,
        t: float,
    ) -> Tuple[float, float, float]:
        """
        Modify ODE parameters during shock.
        
        Returns:
            (r, K, beta) tuple with shock modifications
        """
        r = params.get_r(t)
        K = params.get_K(t)
        beta = params.get_beta(t)
        
        if self.is_active(t):
            # Reduce carrying capacity
            K *= (1 - self.K_reduction * self.severity)
            # Increase friction
            beta += self.beta_increase * self.severity
            
        return r, K, beta


@dataclass
class ODESolution:
    """Result of solving the resilience ODE."""
    t: NDArray[np.float64]           # Time points
    L: NDArray[np.float64]           # Labor force values
    params: ODEParameters
    shocks: List[ClimateShock]
    
    # Computed metrics
    recovery_time: Optional[float] = None
    min_labor_force: Optional[float] = None
    equilibrium: Optional[float] = None
    resilience_score: Optional[float] = None
    
    def compute_metrics(self, threshold: float = 0.95) -> None:
        """Compute recovery and resilience metrics.

        Recovery time: days from the shock trough until employment returns to
        within ``threshold`` of the pre-shock level.  Falls back to measuring
        against the post-shock equilibrium when the initial level is
        unattainable.  A minimum of 1 day is returned whenever a shock caused
        any measurable dip, so the UI never shows a confusing ``0d``.

        Resilience score: 0-100 composite combining recovery speed, shock
        depth, and long-run equilibrium ratio.
        """
        self.min_labor_force = float(np.min(self.L))
        self.equilibrium = float(self.L[-1])
        initial = self.L[0]
        sim_span = float(self.t[-1] - self.t[0])

        # Use the higher of (initial, equilibrium) × threshold as recovery
        # target so that even mild shocks with a small dip report a nonzero
        # recovery time.
        recovery_target = max(initial, self.equilibrium) * threshold

        # If the shock's trough is already above the target the series
        # essentially never "broke" — but we still want to report a small
        # recovery time proportional to the dip depth.
        trough_idx = int(np.argmin(self.L))
        dip_depth = initial - self.min_labor_force

        below = np.where(self.L < recovery_target)[0]
        if len(below) > 0:
            last_below = below[-1]
            if last_below < len(self.L) - 1:
                self.recovery_time = float(self.t[last_below + 1] - self.t[0])
            else:
                self.recovery_time = sim_span
        elif dip_depth > 1e-4:
            # Shock caused a measurable dip but never crossed the 95% line.
            # Estimate recovery as the time from trough back to 99% of initial.
            near_initial = np.where(self.L[trough_idx:] >= initial * 0.99)[0]
            if len(near_initial) > 0:
                self.recovery_time = float(
                    self.t[trough_idx + near_initial[0]] - self.t[0]
                )
            else:
                self.recovery_time = float(self.t[trough_idx] - self.t[0]) + 1.0
            self.recovery_time = max(self.recovery_time, 1.0)
        else:
            self.recovery_time = 0.0

        # Clamp to simulation span
        self.recovery_time = min(self.recovery_time, sim_span)

        # --- Resilience score (0-100) ---
        # Recovery speed: faster is better (baseline 90 days = midpoint)
        recovery_factor = 1.0 / (1.0 + self.recovery_time / 90.0)
        # Shock depth: how little employment dropped
        depth_factor = (self.min_labor_force / initial) if initial > 0 else 0
        # Equilibrium ratio: how much capacity was retained
        equilibrium_factor = (self.equilibrium / initial) if initial > 0 else 0

        raw = (
            0.3 * recovery_factor
            + 0.3 * depth_factor
            + 0.4 * equilibrium_factor
        )
        self.resilience_score = float(np.clip(raw * 100.0, 0.0, 100.0))


class ResilienceODE:
    """
    Solver for the labor market resilience differential equation.
    
    dL/dt = rL(1 - L/K) - β(EJ)
    
    This logistic growth model with friction captures:
    - Natural recovery (logistic term)
    - Environmental justice barriers (friction term)
    - Climate shock impacts (parameter modifications)
    """
    
    def __init__(
        self,
        params: Optional[ODEParameters] = None,
        ej_data: Optional[EJScreenData] = None,
    ):
        """
        Initialize the ODE solver.
        
        Args:
            params: ODE parameters (r, K, β)
            ej_data: EJScreen data for computing β
        """
        self.params = params or ODEParameters()
        self.ej_data = ej_data
        self.shocks: List[ClimateShock] = []
        
        # If EJ data provided, compute β from it
        if ej_data is not None:
            self.params.beta = ej_data.compute_beta()
            
    def add_shock(self, shock: ClimateShock) -> None:
        """Add a climate shock event."""
        self.shocks.append(shock)
        
    def clear_shocks(self) -> None:
        """Clear all shock events."""
        self.shocks = []
        
    def _ode_func(
        self,
        L: float,
        t: float,
    ) -> float:
        """
        The ODE right-hand side: dL/dt = rL(1 - L/K) - β
        
        Args:
            L: Current labor force level
            t: Current time
            
        Returns:
            dL/dt
        """
        # Get base parameters
        r = self.params.get_r(t)
        K = self.params.get_K(t)
        beta = self.params.get_beta(t)
        
        # Apply any active shocks
        for shock in self.shocks:
            r, K, beta = shock.modify_params(self.params, t)
            
        # Logistic growth with friction
        # dL/dt = rL(1 - L/K) - β
        logistic_term = r * L * (1 - L / K) if K > 0 else 0
        friction_term = beta
        
        dL_dt = logistic_term - friction_term
        
        return dL_dt
        
    def _ode_func_ivp(
        self,
        t: float,
        L: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Wrapper for solve_ivp (different argument order)."""
        return np.array([self._ode_func(L[0], t)])
        
    def solve(
        self,
        L0: float,
        t_span: Tuple[float, float],
        n_points: int = 100,
        method: str = "ivp",
    ) -> ODESolution:
        """
        Solve the resilience ODE.
        
        Args:
            L0: Initial labor force level (0-1 scale)
            t_span: (t_start, t_end) time interval
            n_points: Number of output time points
            method: "odeint" or "ivp"
            
        Returns:
            ODESolution with time series and metrics
        """
        t = np.linspace(t_span[0], t_span[1], n_points)
        
        if method == "odeint":
            # scipy.integrate.odeint
            L = odeint(self._ode_func, L0, t)
            L = L.flatten()
        else:
            # scipy.integrate.solve_ivp — handles discontinuous shocks reliably
            min_shock_dur = min((s.duration for s in self.shocks), default=10.0)
            sol = solve_ivp(
                self._ode_func_ivp,
                t_span,
                [L0],
                t_eval=t,
                method="RK45",
                max_step=max(0.5, min_shock_dur / 4.0),
            )
            L = sol.y.flatten()
            
        # Ensure L stays in valid range
        L = np.clip(L, 0, 1)
        
        solution = ODESolution(
            t=t,
            L=L,
            params=self.params,
            shocks=self.shocks.copy(),
        )
        solution.compute_metrics()
        
        return solution
        
    def solve_with_ej(
        self,
        L0: float,
        t_span: Tuple[float, float],
        ej_data: EJScreenData,
        n_points: int = 100,
    ) -> ODESolution:
        """
        Solve ODE using EJScreen data for β coefficient.
        
        Args:
            L0: Initial labor force
            t_span: Time interval
            ej_data: EJScreen data for the census tract
            n_points: Number of output points
            
        Returns:
            ODESolution
        """
        # Update β from EJ data
        self.ej_data = ej_data
        self.params.beta = ej_data.compute_beta()
        
        return self.solve(L0, t_span, n_points)
        
    def compare_tracts(
        self,
        L0: float,
        t_span: Tuple[float, float],
        tracts: List[EJScreenData],
        n_points: int = 100,
    ) -> Dict[str, ODESolution]:
        """
        Compare recovery dynamics across multiple census tracts.
        
        Args:
            L0: Initial labor force (same for all)
            t_span: Time interval
            tracts: List of EJScreen data for different tracts
            n_points: Number of output points
            
        Returns:
            Dict mapping tract_id to ODESolution
        """
        solutions = {}
        
        for tract in tracts:
            self.params.beta = tract.compute_beta()
            solution = self.solve(L0, t_span, n_points)
            solutions[tract.tract_id] = solution
            
        return solutions
        
    def find_critical_beta(
        self,
        L0: float = 0.8,
        t_end: float = 365,
        target_equilibrium: float = 0.5,
    ) -> float:
        """
        Find critical β where equilibrium drops below target.
        
        This identifies the EJ threshold where labor market 
        becomes unsustainable.
        
        Args:
            L0: Initial labor force
            t_end: Time horizon
            target_equilibrium: Target minimum equilibrium
            
        Returns:
            Critical β value
        """
        def objective(beta: float) -> float:
            self.params.beta = beta
            solution = self.solve(L0, (0, t_end), n_points=50)
            return abs(solution.equilibrium - target_equilibrium)
            
        result = minimize_scalar(
            objective,
            bounds=(0.01, 1.0),
            method="bounded",
        )
        
        return result.x
        
    def sensitivity_analysis(
        self,
        L0: float,
        t_span: Tuple[float, float],
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 10,
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Analyze sensitivity of equilibrium to parameter changes.
        
        Args:
            L0: Initial labor force
            t_span: Time interval
            param_ranges: Dict of param name to (min, max) range
            n_samples: Number of samples per parameter
            
        Returns:
            Dict mapping param name to list of (value, equilibrium) pairs
        """
        results = {}
        original_params = (
            self.params.r,
            self.params.K,
            self.params.beta,
        )
        
        for param_name, (p_min, p_max) in param_ranges.items():
            values = np.linspace(p_min, p_max, n_samples)
            param_results = []
            
            for val in values:
                # Set parameter
                if param_name == "r":
                    self.params.r = val
                elif param_name == "K":
                    self.params.K = val
                elif param_name == "beta":
                    self.params.beta = val
                    
                solution = self.solve(L0, t_span, n_points=50)
                param_results.append((val, solution.equilibrium))
                
            results[param_name] = param_results
            
            # Reset parameters
            self.params.r, self.params.K, self.params.beta = original_params
            
        return results


def create_shock_scenario(
    scenario: str = "moderate_storm",
    start_time: float = 30.0,
) -> ClimateShock:
    """
    Create a pre-defined climate shock scenario.
    
    Args:
        scenario: One of "minor", "moderate_storm", "major_hurricane", "catastrophic"
        start_time: When shock begins (days)
        
    Returns:
        Configured ClimateShock
    """
    scenarios = {
        "minor": ClimateShock(
            start_time=start_time,
            duration=7.0,
            severity=0.2,
            K_reduction=0.1,
            beta_increase=0.05,
        ),
        "moderate_storm": ClimateShock(
            start_time=start_time,
            duration=14.0,
            severity=0.5,
            K_reduction=0.25,
            beta_increase=0.15,
        ),
        "major_hurricane": ClimateShock(
            start_time=start_time,
            duration=30.0,
            severity=0.7,
            K_reduction=0.4,
            beta_increase=0.25,
        ),
        "catastrophic": ClimateShock(
            start_time=start_time,
            duration=60.0,
            severity=0.9,
            K_reduction=0.6,
            beta_increase=0.4,
        ),
    }
    
    return scenarios.get(scenario, scenarios["moderate_storm"])


def create_ej_tract(
    profile: str = "average",
    tract_id: str = "000000",
) -> EJScreenData:
    """
    Create pre-defined EJScreen tract profiles.
    
    Args:
        profile: One of "low_burden", "average", "high_burden", "extreme_burden"
        tract_id: Census tract ID
        
    Returns:
        Configured EJScreenData
    """
    profiles = {
        "low_burden": EJScreenData(
            tract_id=tract_id,
            pm25_percentile=20,
            ozone_percentile=25,
            traffic_percentile=15,
            superfund_percentile=10,
            wastewater_percentile=20,
            low_income_percentile=20,
            minority_percentile=25,
            linguistic_iso_percentile=10,
            less_than_hs_percentile=15,
            unemployment_percentile=20,
            ej_index=18,
            climate_vulnerability=22,
        ),
        "average": EJScreenData(
            tract_id=tract_id,
            pm25_percentile=50,
            ozone_percentile=50,
            traffic_percentile=50,
            superfund_percentile=50,
            wastewater_percentile=50,
            low_income_percentile=50,
            minority_percentile=50,
            linguistic_iso_percentile=50,
            less_than_hs_percentile=50,
            unemployment_percentile=50,
            ej_index=50,
            climate_vulnerability=50,
        ),
        "high_burden": EJScreenData(
            tract_id=tract_id,
            pm25_percentile=75,
            ozone_percentile=70,
            traffic_percentile=80,
            superfund_percentile=65,
            wastewater_percentile=75,
            low_income_percentile=80,
            minority_percentile=75,
            linguistic_iso_percentile=70,
            less_than_hs_percentile=75,
            unemployment_percentile=78,
            ej_index=76,
            climate_vulnerability=72,
        ),
        "extreme_burden": EJScreenData(
            tract_id=tract_id,
            pm25_percentile=92,
            ozone_percentile=88,
            traffic_percentile=95,
            superfund_percentile=85,
            wastewater_percentile=90,
            low_income_percentile=95,
            minority_percentile=90,
            linguistic_iso_percentile=88,
            less_than_hs_percentile=92,
            unemployment_percentile=94,
            ej_index=91,
            climate_vulnerability=89,
        ),
    }
    
    return profiles.get(profile, profiles["average"])


def optimize_policy_allocation(
    current_beta: float,
    ej_percentile: float,
    sensitive_worker_count: int,
    r: float = 0.10,
    K: float = 0.95,
    shock_severity: float = 0.5,
    shock_duration: float = 21.0,
    shock_start: float = 30.0,
    reskill_target_pct: float = 0.10,
    cost_per_worker: float = 12_000.0,
    sim_days: int = 365,
) -> Dict[str, float]:
    """
    CAP Measure CP-1 "Green Technology Workforce" optimizer.

    Computes the budget to re-skill a fraction of climate-sensitive workers
    into resilient sectors and quantifies the resulting beta (EJ friction)
    reduction and recovery-time improvement.

    Uses scipy.optimize.minimize_scalar to find the optimal reskill
    percentage in [0.05, 0.30] that minimises recovery time.

    Args:
        current_beta:           Current EJ friction coefficient.
        ej_percentile:          Tract EJ percentile (0-100).
        sensitive_worker_count: Workers in climate-sensitive industries.
        r:                      ODE recovery rate.
        K:                      ODE carrying capacity.
        shock_severity:         Climate shock severity (0-1).
        shock_duration:         Shock duration in days.
        shock_start:            Shock start day.
        reskill_target_pct:     Default fraction of workers to reskill.
        cost_per_worker:        Unit cost of green re-skilling (USD).
        sim_days:               Simulation horizon in days.

    Returns:
        Dict with budget, beta changes, recovery-time changes.
    """
    from scipy.optimize import minimize_scalar

    def _recovery_time_for_pct(pct: float) -> float:
        """Run ODE with a reduced beta and return recovery time."""
        reduction_factor = 1.0 - pct * (ej_percentile / 100.0)
        new_beta = current_beta * max(reduction_factor, 0.1)
        params = ODEParameters(r=r, K=K, beta=new_beta)
        ode = ResilienceODE(params)
        ode.add_shock(ClimateShock(
            start_time=shock_start,
            duration=shock_duration,
            severity=shock_severity,
            K_reduction=0.3 * shock_severity,
            beta_increase=0.02 * shock_severity,
        ))
        sol = ode.solve(L0=0.92, t_span=(0, sim_days), n_points=200)
        return sol.recovery_time if sol.recovery_time is not None else float(sim_days)

    # --- baseline recovery time ---
    recovery_before = _recovery_time_for_pct(0.0)

    # --- find optimal reskill percentage ---
    result = minimize_scalar(
        _recovery_time_for_pct,
        bounds=(0.05, 0.30),
        method="bounded",
    )
    optimal_pct = float(result.x)
    recovery_after_optimal = float(result.fun)

    # --- also compute the result at the requested default pct ---
    recovery_after_target = _recovery_time_for_pct(reskill_target_pct)

    workers_to_reskill = int(sensitive_worker_count * optimal_pct)
    total_budget = workers_to_reskill * cost_per_worker

    beta_after = current_beta * max(1.0 - optimal_pct * (ej_percentile / 100.0), 0.1)
    beta_reduction_pct = (1.0 - beta_after / current_beta) * 100.0 if current_beta > 0 else 0.0

    return {
        "optimal_reskill_pct": round(optimal_pct, 4),
        "workers_to_reskill": workers_to_reskill,
        "total_budget": round(total_budget, 2),
        "cost_per_worker": cost_per_worker,
        "beta_before": round(current_beta, 6),
        "beta_after": round(beta_after, 6),
        "beta_reduction_pct": round(beta_reduction_pct, 1),
        "recovery_time_before": round(recovery_before, 1),
        "recovery_time_after": round(recovery_after_optimal, 1),
        "recovery_improvement_days": round(recovery_before - recovery_after_optimal, 1),
        "recovery_at_default_target": round(recovery_after_target, 1),
    }


class CoupledMarkovODE:
    """
    Coupled Markov chain + ODE model.
    
    Combines discrete state transitions (Markov) with
    continuous labor force dynamics (ODE) for richer modeling.
    """
    
    def __init__(
        self,
        markov_chain,  # MarkovTransitionMatrix
        ode_solver: ResilienceODE,
    ):
        """
        Initialize coupled model.
        
        Args:
            markov_chain: MarkovTransitionMatrix for state transitions
            ode_solver: ResilienceODE for continuous dynamics
        """
        self.markov = markov_chain
        self.ode = ode_solver
        
    def simulate_coupled(
        self,
        initial_distribution: NDArray[np.float64],
        initial_labor_force: float,
        n_days: int,
        shock_day: Optional[int] = None,
        shock_severity: float = 0.5,
    ) -> Dict[str, NDArray]:
        """
        Run coupled simulation.
        
        The Markov chain determines workforce distribution across states,
        while the ODE determines total labor force evolution.
        
        Args:
            initial_distribution: Initial distribution over Markov states [4,]
            initial_labor_force: Initial total labor force (0-1)
            n_days: Number of days to simulate
            shock_day: Day when climate shock occurs (optional)
            shock_severity: Severity of shock
            
        Returns:
            Dict with 'markov_history', 'labor_force', 'combined'
        """
        from .markov_chain import ShockParameters
        
        # Setup shock schedules
        shock_schedule = {}
        if shock_day is not None:
            shock_schedule[shock_day] = ShockParameters(
                severity=shock_severity,
                duration_days=14,
            )
            
        # Run Markov simulation
        markov_history = self.markov.simulate_population(
            initial_distribution,
            n_days,
            shock_schedule=shock_schedule,
        )
        
        # Run ODE simulation
        if shock_day is not None:
            self.ode.clear_shocks()
            self.ode.add_shock(ClimateShock(
                start_time=float(shock_day),
                duration=14.0,
                severity=shock_severity,
            ))
            
        ode_solution = self.ode.solve(
            initial_labor_force,
            (0, n_days),
            n_points=n_days + 1,
        )
        
        # Combine: actual workers in each state = distribution × total labor force
        combined = markov_history * ode_solution.L.reshape(-1, 1)
        
        return {
            "markov_history": markov_history,
            "labor_force": ode_solution.L,
            "combined": combined,
            "ode_solution": ode_solution,
        }
