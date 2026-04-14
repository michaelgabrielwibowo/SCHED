"""
NSGA-III Multi-Objective Evolutionary Algorithm

Evidence Status:
- NSGA-III algorithm: CONFIRMED from Deb & Jain (2014)
- Application to scheduling: CONFIRMED from literature
- Application to SFJSSP: PROPOSED (this work)

Reference:
Deb, K., & Jain, H. (2014). An evolutionary many-objective optimization
algorithm using reference-point-based nondominated sorting approach.
IEEE Transactions on Evolutionary Computation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any, Sequence
from dataclasses import dataclass, field
import copy


NSGA3_CONSTRAINT_HANDLING_POLICIES: Tuple[str, ...] = (
    "legacy_scalar_penalized_objectives",
    "objective_only",
    "feasibility_first_constrained_domination",
    "feasibility_first_lexicographic",
    "hard_feasible_first_soft_penalties",
)
NSGA3_DEFAULT_CONSTRAINT_HANDLING = "legacy_scalar_penalized_objectives"
NSGA3_PARENT_SELECTION_POLICIES: Tuple[str, ...] = (
    "random_pairing",
    "feasible_tardiness_tournament",
)
NSGA3_LOCAL_IMPROVEMENT_POLICIES: Tuple[str, ...] = (
    "none",
    "tardiness_sequence_repair",
)
NSGA3_DEFAULT_LOCAL_IMPROVEMENT = "none"
NSGA3_SEQUENCE_MUTATION_POLICIES: Tuple[str, ...] = (
    "legacy_random_swap",
    "urgent_pull_forward",
)
NSGA3_DEFAULT_SEQUENCE_MUTATION = "legacy_random_swap"
NSGA3_IMMIGRANT_POLICIES: Tuple[str, ...] = (
    "none",
    "feasible_tardiness_archive",
)
NSGA3_DEFAULT_IMMIGRANT_POLICY = "none"
NSGA3_CROSSOVER_POLICIES: Tuple[str, ...] = (
    "legacy_pox_uniform_assignments",
    "urgent_prefix_merge",
)
NSGA3_DEFAULT_CROSSOVER_POLICY = "legacy_pox_uniform_assignments"
NSGA3_POLICIES_REQUIRING_DETAILED_EVALUATION = frozenset(
    {
        "objective_only",
        "feasibility_first_constrained_domination",
        "feasibility_first_lexicographic",
        "hard_feasible_first_soft_penalties",
    }
)


def nsga3_policy_requires_detailed_evaluation(policy: str) -> bool:
    """Return whether a policy needs detailed raw/penalty evaluation payloads."""
    return policy in NSGA3_POLICIES_REQUIRING_DETAILED_EVALUATION


def nsga3_policy_uses_penalized_objectives(policy: str) -> bool:
    """Return whether a policy ranks candidates on scalar-penalized objectives."""
    return policy in {
        "legacy_scalar_penalized_objectives",
        "hard_feasible_first_soft_penalties",
    }


def _empty_local_improvement_stats() -> Dict[str, int]:
    """Return zeroed counters for local-improvement diagnostics."""
    return {
        "repair_attempted_children": 0,
        "repair_accepted_children": 0,
        "repair_improved_n_tardy_jobs_count": 0,
        "repair_improved_weighted_tardiness_count": 0,
        "repair_rejected_due_to_makespan_cap_count": 0,
    }


def _merge_local_improvement_stats(
    target: Dict[str, int],
    delta: Dict[str, int],
) -> Dict[str, int]:
    """Accumulate one local-improvement counter payload into another."""
    for key, value in delta.items():
        target[key] = int(target.get(key, 0)) + int(value)
    return target


def _empty_sequence_mutation_stats() -> Dict[str, int]:
    """Return zeroed counters for sequence-mutation diagnostics."""
    return {
        "urgent_sequence_mutation_attempts": 0,
        "urgent_sequence_mutation_applied": 0,
        "urgent_sequence_mutation_fallback_random_swap_count": 0,
        "urgent_sequence_mutation_noop_count": 0,
        "urgent_sequence_mutation_changed_position_distance_sum": 0,
    }


def _merge_sequence_mutation_stats(
    target: Dict[str, int],
    delta: Dict[str, int],
) -> Dict[str, int]:
    """Accumulate one sequence-mutation counter payload into another."""
    for key, value in delta.items():
        target[key] = int(target.get(key, 0)) + int(value)
    return target


def _empty_immigrant_stats() -> Dict[str, int]:
    """Return zeroed counters for immigrant-archive diagnostics."""
    return {
        "immigrant_archive_admission_count": 0,
        "immigrant_archive_replaced_count": 0,
        "immigrant_injection_events": 0,
        "immigrant_injected_individuals": 0,
        "immigrant_skipped_duplicate_count": 0,
        "immigrant_survivors_in_next_population": 0,
    }


def _merge_immigrant_stats(
    target: Dict[str, int],
    delta: Dict[str, int],
) -> Dict[str, int]:
    """Accumulate one immigrant-diagnostic payload into another."""
    for key, value in delta.items():
        target[key] = int(target.get(key, 0)) + int(value)
    return target


def _empty_crossover_stats() -> Dict[str, int]:
    """Return zeroed counters for crossover diagnostics."""
    return {
        "urgent_crossover_attempts": 0,
        "urgent_crossover_applied": 0,
        "urgent_crossover_fallback_legacy_count": 0,
        "urgent_crossover_prefix_total": 0,
        "urgent_crossover_children_from_tardiness_better_parent_count": 0,
    }


def _merge_crossover_stats(
    target: Dict[str, int],
    delta: Dict[str, int],
) -> Dict[str, int]:
    """Accumulate one crossover-diagnostic payload into another."""
    for key, value in delta.items():
        target[key] = int(target.get(key, 0)) + int(value)
    return target


def _empty_child_lineage_stats() -> Dict[str, Any]:
    """Return zeroed counters for child-vs-parent lineage diagnostics."""
    return {
        "children_evaluated_count": 0,
        "children_hard_feasible_count": 0,
        "children_zero_tardy_count": 0,
        "children_improve_both_parents_n_tardy_jobs_count": 0,
        "children_improve_both_parents_weighted_tardiness_count": 0,
        "children_improve_both_parents_makespan_count": 0,
        "best_child_weighted_tardiness": None,
        "best_child_generation": None,
        "best_child_n_tardy_jobs": None,
        "best_child_makespan": None,
    }


def _lineage_best_child_key(stats: Dict[str, Any]) -> Optional[Tuple[float, float, float, int]]:
    """Build a comparable key for the best hard-feasible child stored in one lineage payload."""
    if stats.get("best_child_weighted_tardiness") is None:
        return None
    generation = stats.get("best_child_generation")
    return (
        float(stats.get("best_child_n_tardy_jobs", float("inf"))),
        float(stats.get("best_child_weighted_tardiness", float("inf"))),
        float(stats.get("best_child_makespan", float("inf"))),
        int(generation) if generation is not None else int(1e9),
    )


def _merge_child_lineage_stats(
    target: Dict[str, Any],
    delta: Dict[str, Any],
) -> Dict[str, Any]:
    """Accumulate one child-lineage payload into another, preserving the best child summary."""
    count_keys = (
        "children_evaluated_count",
        "children_hard_feasible_count",
        "children_zero_tardy_count",
        "children_improve_both_parents_n_tardy_jobs_count",
        "children_improve_both_parents_weighted_tardiness_count",
        "children_improve_both_parents_makespan_count",
    )
    for key in count_keys:
        target[key] = int(target.get(key, 0)) + int(delta.get(key, 0))

    delta_best = _lineage_best_child_key(delta)
    if delta_best is None:
        return target

    target_best = _lineage_best_child_key(target)
    if target_best is None or delta_best < target_best:
        target["best_child_weighted_tardiness"] = float(delta["best_child_weighted_tardiness"])
        target["best_child_generation"] = int(delta["best_child_generation"])
        target["best_child_n_tardy_jobs"] = float(delta["best_child_n_tardy_jobs"])
        target["best_child_makespan"] = float(delta["best_child_makespan"])
    return target


def _details_are_hard_feasible(details: Dict[str, Any]) -> bool:
    """Return whether a detailed-evaluation payload is hard-feasible."""
    penalties = details.get("penalties") or {}
    return (
        bool(details.get("is_feasible", True))
        and float(penalties.get("hard_violations", 0.0)) <= 0.0
    )


def _tardiness_repair_score(details: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """Rank repair candidates by feasibility, tardiness, then schedule efficiency."""
    metrics = details.get("metrics") or {}
    penalties = details.get("penalties") or {}
    return (
        float(penalties.get("hard_violations", 0.0)),
        float(metrics.get("n_tardy_jobs", penalties.get("n_tardy_jobs", float("inf")))),
        float(metrics.get("weighted_tardiness", penalties.get("weighted_tardiness", float("inf")))),
        float(metrics.get("makespan", float("inf"))),
        float(metrics.get("total_energy", float("inf"))),
    )


def _feasible_tardiness_archive_score(details: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Rank hard-feasible immigrant candidates by tardiness, then efficiency."""
    metrics = details.get("metrics") or {}
    penalties = details.get("penalties") or {}
    return (
        float(metrics.get("n_tardy_jobs", penalties.get("n_tardy_jobs", float("inf")))),
        float(metrics.get("weighted_tardiness", penalties.get("weighted_tardiness", float("inf")))),
        float(metrics.get("makespan", float("inf"))),
        float(metrics.get("total_energy", float("inf"))),
    )


def _genome_signature(genome: Dict[str, np.ndarray]) -> Tuple[Any, ...]:
    """Build a deterministic, comparable signature for one genome."""
    return (
        tuple(int(value) for value in genome["sequence"]),
        tuple(int(value) for value in genome["machines"]),
        tuple(int(value) for value in genome["workers"]),
        tuple(int(value) for value in genome["modes"]),
        tuple(int(value) for value in genome["offsets"]),
        tuple((int(job_id), int(op_id)) for job_id, op_id in genome["op_list"]),
    )


def _details_from_individual(ind: "Individual") -> Dict[str, Any]:
    """Project one evaluated individual into the details-like payload used by archive helpers."""
    return {
        "metrics": dict(ind.metrics),
        "penalties": dict(ind.penalties),
        "constraint_violations": list(ind.constraint_violations),
        "is_feasible": bool(ind.is_feasible),
    }


@dataclass
class Individual:
    """
    Individual solution in the population
    """
    genome: Dict[str, np.ndarray]
    objectives: List[float] = field(default_factory=list)
    raw_objectives: List[float] = field(default_factory=list)
    penalized_objectives: List[float] = field(default_factory=list)
    rank: int = 0
    niche_count: int = 0
    reference_point: Optional[np.ndarray] = None
    penalties: Dict[str, float] = field(default_factory=dict)
    constraint_violations: List[str] = field(default_factory=list)
    constraint_key: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    is_feasible: bool = True
    metrics: Dict[str, float] = field(default_factory=dict)

    # For SFJSSP evaluation
    makespan: float = 0.0
    energy: float = 0.0
    ergonomic_risk: float = 0.0
    labor_cost: float = 0.0

    def __lt__(self, other):
        return self.rank < other.rank


@dataclass
class ImmigrantArchiveEntry:
    """Stored hard-feasible immigrant candidate tracked inside one NSGA run."""
    individual: Individual
    signature: Tuple[Any, ...]
    source_generation: int
    score: Tuple[float, float, float, float]


@dataclass
class Population:
    """Population of individuals"""
    individuals: List[Individual] = field(default_factory=list)
    size: int = 100

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]

    def add(self, ind: Individual):
        self.individuals.append(ind)

    def clear(self):
        self.individuals = []

    def get_objectives_matrix(self) -> np.ndarray:
        return np.array([ind.objectives for ind in self.individuals])

    def get_best(self, objective_idx: int = 0) -> Individual:
        if not self.individuals:
            return None
        best_idx = np.argmin([ind.objectives[objective_idx] for ind in self.individuals])
        return self.individuals[best_idx]


class NSGA3:
    """
    NSGA-III Multi-Objective Evolutionary Algorithm
    """

    def __init__(
        self,
        n_objectives: int = 4,
        population_size: int = 100,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        seed: int = 42,
        constraint_handling: str = NSGA3_DEFAULT_CONSTRAINT_HANDLING,
        parent_selection: str = "random_pairing",
        crossover_policy: str = NSGA3_DEFAULT_CROSSOVER_POLICY,
        local_improvement: str = NSGA3_DEFAULT_LOCAL_IMPROVEMENT,
        sequence_mutation: str = NSGA3_DEFAULT_SEQUENCE_MUTATION,
        immigrant_policy: str = NSGA3_DEFAULT_IMMIGRANT_POLICY,
        immigrant_count: int = 2,
        immigrant_period: int = 5,
        immigrant_archive_size: int = 8,
    ):
        if constraint_handling not in NSGA3_CONSTRAINT_HANDLING_POLICIES:
            raise ValueError(
                f"Unsupported constraint handling policy {constraint_handling!r}; "
                f"expected one of {list(NSGA3_CONSTRAINT_HANDLING_POLICIES)}"
            )
        if parent_selection not in NSGA3_PARENT_SELECTION_POLICIES:
            raise ValueError(
                f"Unsupported parent selection policy {parent_selection!r}; "
                f"expected one of {list(NSGA3_PARENT_SELECTION_POLICIES)}"
            )
        if crossover_policy not in NSGA3_CROSSOVER_POLICIES:
            raise ValueError(
                f"Unsupported crossover policy {crossover_policy!r}; "
                f"expected one of {list(NSGA3_CROSSOVER_POLICIES)}"
            )
        if local_improvement not in NSGA3_LOCAL_IMPROVEMENT_POLICIES:
            raise ValueError(
                f"Unsupported local improvement policy {local_improvement!r}; "
                f"expected one of {list(NSGA3_LOCAL_IMPROVEMENT_POLICIES)}"
            )
        if sequence_mutation not in NSGA3_SEQUENCE_MUTATION_POLICIES:
            raise ValueError(
                f"Unsupported sequence mutation policy {sequence_mutation!r}; "
                f"expected one of {list(NSGA3_SEQUENCE_MUTATION_POLICIES)}"
            )
        if immigrant_policy not in NSGA3_IMMIGRANT_POLICIES:
            raise ValueError(
                f"Unsupported immigrant policy {immigrant_policy!r}; "
                f"expected one of {list(NSGA3_IMMIGRANT_POLICIES)}"
            )
        if immigrant_count < 0:
            raise ValueError("immigrant_count must be >= 0")
        if immigrant_period < 1:
            raise ValueError("immigrant_period must be >= 1")
        if immigrant_archive_size < 1:
            raise ValueError("immigrant_archive_size must be >= 1")
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.constraint_handling = constraint_handling
        self.parent_selection = parent_selection
        self.crossover_policy = crossover_policy
        self.local_improvement = local_improvement
        self.sequence_mutation = sequence_mutation
        self.immigrant_policy = immigrant_policy
        self.immigrant_count = int(immigrant_count)
        self.immigrant_period = int(immigrant_period)
        self.immigrant_archive_size = int(immigrant_archive_size)
        self.rng = np.random.default_rng(seed)
        self.reference_points = self._generate_reference_points()
        self.evaluate_fn: Optional[Callable] = None
        self.evaluate_details_fn: Optional[Callable] = None
        self.create_individual_fn: Optional[Callable] = None
        self.seed_individuals_fn: Optional[Callable[[Any], List[Dict[str, np.ndarray]]]] = None
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []
        self.last_run_diagnostics: Dict[str, Any] = {
            "constraint_handling": self.constraint_handling,
            "parent_selection": self.parent_selection,
            "crossover_policy": self.crossover_policy,
            "local_improvement": self.local_improvement,
            "sequence_mutation": self.sequence_mutation,
            "immigrant_policy": self.immigrant_policy,
            "immigrant_count": self.immigrant_count,
            "immigrant_period": self.immigrant_period,
            "immigrant_archive_size": self.immigrant_archive_size,
            "immigrant_archive_final_size": 0,
            **_empty_crossover_stats(),
            **_empty_local_improvement_stats(),
            **_empty_sequence_mutation_stats(),
            **_empty_immigrant_stats(),
            **_empty_child_lineage_stats(),
        }

    def _select_ranking_objectives(
        self,
        raw_objectives: Sequence[float],
        penalized_objectives: Sequence[float],
    ) -> List[float]:
        if nsga3_policy_uses_penalized_objectives(self.constraint_handling):
            return list(penalized_objectives)
        return list(raw_objectives)

    def _generate_reference_points(self, n_divisions: int = 12) -> np.ndarray:
        points = []
        self._generate_reference_recursive(np.zeros(self.n_objectives), 0, n_divisions, points)
        return np.array(points) / n_divisions

    def _generate_reference_recursive(self, point: np.ndarray, obj_idx: int, remaining: int, points: List[np.ndarray]):
        if obj_idx == self.n_objectives - 1:
            point[obj_idx] = remaining
            points.append(point.copy())
        else:
            for i in range(remaining + 1):
                point[obj_idx] = i
                self._generate_reference_recursive(point, obj_idx + 1, remaining - i, points)

    def set_problem(
        self,
        evaluate_fn: Callable,
        create_individual_fn: Callable,
        seed_individuals_fn: Optional[Callable[[Any], List[Dict[str, np.ndarray]]]] = None,
        evaluate_details_fn: Optional[Callable] = None,
    ):
        self.evaluate_fn = evaluate_fn
        self.evaluate_details_fn = evaluate_details_fn
        self.create_individual_fn = create_individual_fn
        self.seed_individuals_fn = seed_individuals_fn

    def initialize_population(self, instance: Any) -> Population:
        pop = Population(size=self.population_size)
        seed_genomes: List[Dict[str, np.ndarray]] = []
        if self.seed_individuals_fn is not None:
            seed_genomes = list(self.seed_individuals_fn(instance) or [])

        for genome in seed_genomes[: self.population_size]:
            pop.add(Individual(genome=copy.deepcopy(genome)))

        while len(pop) < self.population_size:
            genome = self.create_individual_fn(instance, self.rng)
            pop.add(Individual(genome=genome))
        return pop

    def _assign_evaluation(self, ind: Individual, evaluation: Any):
        """Project one evaluation payload into the mutable individual fields."""
        if isinstance(evaluation, dict):
            raw_objectives = list(
                evaluation.get("raw_objectives")
                or evaluation.get("penalized_objectives")
                or []
            )
            penalized_objectives = list(
                evaluation.get("penalized_objectives")
                or raw_objectives
            )
            penalties = dict(evaluation.get("penalties") or {})
            ind.raw_objectives = raw_objectives
            ind.penalized_objectives = penalized_objectives
            ind.objectives = self._select_ranking_objectives(
                raw_objectives=raw_objectives,
                penalized_objectives=penalized_objectives,
            )
            ind.penalties = penalties
            ind.metrics = dict(evaluation.get("metrics") or {})
            ind.constraint_violations = list(
                evaluation.get("constraint_violations") or []
            )
            ind.constraint_key = self._build_constraint_key(penalties)
            ind.is_feasible = (
                bool(evaluation.get("is_feasible", True))
                and float(penalties.get("hard_violations", 0.0)) <= 0.0
            )
            objective_values = raw_objectives if raw_objectives else penalized_objectives
        else:
            if nsga3_policy_requires_detailed_evaluation(self.constraint_handling):
                raise ValueError(
                    f"Constraint handling policy {self.constraint_handling!r} "
                    "requires a detailed evaluation payload with raw and penalized objectives."
                )
            objectives = list(evaluation)
            ind.objectives = objectives
            ind.raw_objectives = list(objectives)
            ind.penalized_objectives = list(objectives)
            ind.penalties = {}
            ind.metrics = {}
            ind.constraint_violations = []
            ind.constraint_key = (0.0, 0.0, 0.0, 0.0)
            ind.is_feasible = True
            objective_values = objectives

        if len(objective_values) >= 4:
            ind.makespan = float(ind.metrics.get("makespan", objective_values[0]))
            ind.energy = float(ind.metrics.get("total_energy", objective_values[1]))
            ind.ergonomic_risk = float(ind.metrics.get("max_ergonomic_exposure", objective_values[2]))
            ind.labor_cost = float(ind.metrics.get("total_labor_cost", objective_values[3]))

    def evaluate_population(self, population: Population, instance: Any):
        for ind in population.individuals:
            evaluation = (
                self.evaluate_details_fn(instance, ind.genome)
                if self.evaluate_details_fn is not None
                else self.evaluate_fn(instance, ind.genome)
            )
            self._assign_evaluation(ind, evaluation)

    def _build_constraint_key(self, penalties: Optional[Dict[str, Any]]) -> Tuple[float, float, float, float]:
        if not penalties:
            return (0.0, 0.0, 0.0, 0.0)
        return (
            float(penalties.get("hard_violations", 0.0)),
            float(penalties.get("n_tardy_jobs", 0.0)),
            float(penalties.get("weighted_tardiness", 0.0)),
            float(penalties.get("ocra_penalty", 0.0)),
        )

    def _non_dominated_sort(self, population: Population) -> List[List[int]]:
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)

        idx = 0
        while fronts[idx]:
            next_front = []
            for i in fronts[idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = idx + 1
                        next_front.append(j)
            idx += 1
            if not next_front: break
            fronts.append(next_front)
        return [f for f in fronts if f]

    def _constraint_key_less(
        self,
        key1: Tuple[float, float, float, float],
        key2: Tuple[float, float, float, float],
    ) -> bool:
        if key1 == key2:
            return False
        for left, right in zip(key1, key2):
            if left < right - 1e-9:
                return True
            if left > right + 1e-9:
                return False
        return False

    def _objectives_dominate(self, obj1: List[float], obj2: List[float]) -> bool:
        better = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2 + 1e-7: return False
            if o1 < o2 - 1e-7: better = True
        return better

    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        if self.constraint_handling in {"legacy_scalar_penalized_objectives", "objective_only"}:
            return self._objectives_dominate(ind1.objectives, ind2.objectives)

        if self.constraint_handling == "hard_feasible_first_soft_penalties":
            left_hard = ind1.constraint_key[0]
            right_hard = ind2.constraint_key[0]
            if left_hard <= 0.0 < right_hard:
                return True
            if right_hard <= 0.0 < left_hard:
                return False
            if left_hard > 0.0 or right_hard > 0.0:
                if self._constraint_key_less(ind1.constraint_key, ind2.constraint_key):
                    return True
                if self._constraint_key_less(ind2.constraint_key, ind1.constraint_key):
                    return False
            return self._objectives_dominate(ind1.objectives, ind2.objectives)

        if self.constraint_handling == "feasibility_first_constrained_domination":
            left_hard = ind1.constraint_key[0]
            right_hard = ind2.constraint_key[0]
            if left_hard <= 0.0 < right_hard:
                return True
            if right_hard <= 0.0 < left_hard:
                return False
            if left_hard > 0.0 or right_hard > 0.0:
                if self._constraint_key_less(ind1.constraint_key, ind2.constraint_key):
                    return True
                if self._constraint_key_less(ind2.constraint_key, ind1.constraint_key):
                    return False
            return self._objectives_dominate(ind1.objectives, ind2.objectives)

        if self.constraint_handling == "feasibility_first_lexicographic":
            if self._constraint_key_less(ind1.constraint_key, ind2.constraint_key):
                return True
            if self._constraint_key_less(ind2.constraint_key, ind1.constraint_key):
                return False
            return self._objectives_dominate(ind1.objectives, ind2.objectives)

        raise ValueError(f"Unsupported constraint handling policy: {self.constraint_handling}")

    def _legacy_sequence_crossover(
        self,
        seq1: np.ndarray,
        seq2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the legacy POX sequence crossover results."""
        job_ids = list(set(seq1))
        if len(job_ids) <= 1:
            return seq1.copy(), seq2.copy()

        sub = set(self.rng.choice(job_ids, size=self.rng.integers(1, len(job_ids)), replace=False))

        def pox(s1, s2):
            c = np.full(len(s1), -1, dtype=int)
            for i in range(len(s1)):
                if s1[i] in sub:
                    c[i] = s1[i]
            ptr = 0
            for i in range(len(s1)):
                if c[i] == -1:
                    while ptr < len(s2) and s2[ptr] in sub:
                        ptr += 1
                    if ptr < len(s2):
                        c[i] = s2[ptr]
                        ptr += 1
            return c

        return pox(seq1, seq2), pox(seq2, seq1)

    def _parent_tardiness_key(self, ind: Individual) -> Tuple[float, float, float]:
        """Rank one parent by hard-feasible tardiness, then makespan."""
        return (
            float(ind.metrics.get("n_tardy_jobs", ind.penalties.get("n_tardy_jobs", 0.0))),
            float(ind.metrics.get("weighted_tardiness", ind.penalties.get("weighted_tardiness", 0.0))),
            float(ind.metrics.get("makespan", ind.makespan)),
        )

    def _collect_child_lineage_stats(
        self,
        children_with_parents: Sequence[Tuple[Individual, Individual, Individual]],
        generation: int,
    ) -> Dict[str, Any]:
        """Summarize whether evaluated children ever beat both parents."""
        diagnostics = _empty_child_lineage_stats()
        best_key: Optional[Tuple[float, float, float, int]] = None

        for child, left_parent, right_parent in children_with_parents:
            diagnostics["children_evaluated_count"] += 1
            if float(child.constraint_key[0]) > 0.0:
                continue

            diagnostics["children_hard_feasible_count"] += 1
            child_n_tardy = float(child.metrics.get("n_tardy_jobs", child.penalties.get("n_tardy_jobs", 0.0)))
            child_weighted_tardiness = float(
                child.metrics.get("weighted_tardiness", child.penalties.get("weighted_tardiness", 0.0))
            )
            child_makespan = float(child.metrics.get("makespan", child.makespan))
            if child_n_tardy <= 0.0:
                diagnostics["children_zero_tardy_count"] += 1

            parent_tardy_keys = [self._parent_tardiness_key(left_parent), self._parent_tardiness_key(right_parent)]
            child_tardiness_key = (child_n_tardy, child_weighted_tardiness, child_makespan)
            if all(child_tardiness_key < parent_key for parent_key in parent_tardy_keys):
                diagnostics["children_improve_both_parents_n_tardy_jobs_count"] += 1

            parent_weighted_tardiness = [key[1] for key in parent_tardy_keys]
            if all(child_weighted_tardiness + 1e-9 < value for value in parent_weighted_tardiness):
                diagnostics["children_improve_both_parents_weighted_tardiness_count"] += 1

            parent_makespans = [
                float(left_parent.metrics.get("makespan", left_parent.makespan)),
                float(right_parent.metrics.get("makespan", right_parent.makespan)),
            ]
            if all(child_makespan + 1e-9 < value for value in parent_makespans):
                diagnostics["children_improve_both_parents_makespan_count"] += 1

            candidate_key = (child_n_tardy, child_weighted_tardiness, child_makespan, generation)
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                diagnostics["best_child_weighted_tardiness"] = child_weighted_tardiness
                diagnostics["best_child_generation"] = generation
                diagnostics["best_child_n_tardy_jobs"] = child_n_tardy
                diagnostics["best_child_makespan"] = child_makespan

        return diagnostics

    def _crossover(
        self,
        p1: Individual,
        p2: Individual,
        instance: Any,
    ) -> Tuple[Individual, Individual, Dict[str, int]]:
        if self.rng.random() > self.crossover_rate:
            return (
                Individual(genome=copy.deepcopy(p1.genome)),
                Individual(genome=copy.deepcopy(p2.genome)),
                _empty_crossover_stats(),
            )

        crossover_stats = _empty_crossover_stats()
        g1, g2 = p1.genome, p2.genome
        seq1, seq2 = g1['sequence'].copy(), g2['sequence'].copy()
        legacy_c1_seq, legacy_c2_seq = self._legacy_sequence_crossover(seq1, seq2)
        if self.crossover_policy == "legacy_pox_uniform_assignments":
            c1_seq, c2_seq = legacy_c1_seq, legacy_c2_seq
        elif self.crossover_policy == "urgent_prefix_merge":
            crossover_stats["urgent_crossover_attempts"] = 1
            if self._parent_tardiness_key(p1) <= self._parent_tardiness_key(p2):
                urgent_parent, support_parent = p1, p2
            else:
                urgent_parent, support_parent = p2, p1
            urgent_seq, prefix_size = _build_urgent_prefix_merged_sequence(
                instance=instance,
                urgent_genome=urgent_parent.genome,
                support_genome=support_parent.genome,
            )
            if urgent_seq is None:
                c1_seq, c2_seq = legacy_c1_seq, legacy_c2_seq
                crossover_stats["urgent_crossover_fallback_legacy_count"] = 1
            else:
                c1_seq = urgent_seq
                c2_seq = legacy_c2_seq
                crossover_stats["urgent_crossover_applied"] = 1
                crossover_stats["urgent_crossover_prefix_total"] = int(prefix_size)
                crossover_stats["urgent_crossover_children_from_tardiness_better_parent_count"] = 1
        else:
            raise ValueError(f"Unsupported crossover policy: {self.crossover_policy}")

        mask = self.rng.random(len(g1['machines'])) < 0.5
        c1_m, c2_m = g1['machines'].copy(), g2['machines'].copy()
        c1_w, c2_w = g1['workers'].copy(), g2['workers'].copy()
        c1_mode, c2_mode = g1['modes'].copy(), g2['modes'].copy()
        c1_o, c2_o = g1['offsets'].copy(), g2['offsets'].copy()
        
        # [FIX] Machines/Workers are mapped to the operation at that index in op_list
        # We must swap by operation, not by sequence position
        for i in range(len(mask)):
            if mask[i]:
                c1_m[i], c2_m[i] = g2['machines'][i], g1['machines'][i]
                c1_w[i], c2_w[i] = g2['workers'][i], g1['workers'][i]
                c1_mode[i], c2_mode[i] = g2['modes'][i], g1['modes'][i]
                c1_o[i], c2_o[i] = g2['offsets'][i], g1['offsets'][i]

        return (
            Individual(
                genome={
                    'sequence': c1_seq,
                    'machines': c1_m,
                    'workers': c1_w,
                    'modes': c1_mode,
                    'offsets': c1_o,
                    'op_list': g1['op_list'],
                }
            ),
            Individual(
                genome={
                    'sequence': c2_seq,
                    'machines': c2_m,
                    'workers': c2_w,
                    'modes': c2_mode,
                    'offsets': c2_o,
                    'op_list': g1['op_list'],
                }
            ),
            crossover_stats,
        )

    def _apply_random_sequence_swap(self, sequence: np.ndarray) -> bool:
        """Apply the legacy in-place random swap used for sequence mutation."""
        if len(sequence) < 2:
            return False
        i1, i2 = self.rng.choice(len(sequence), 2, replace=False)
        sequence[i1], sequence[i2] = sequence[i2], sequence[i1]
        return True

    def _mutate(self, ind: Individual, instance: Any) -> Dict[str, int]:
        g = ind.genome
        ops = g['op_list']
        sequence_mutation_stats = _empty_sequence_mutation_stats()
        if self.rng.random() < self.mutation_rate:
            if self.sequence_mutation == "legacy_random_swap":
                self._apply_random_sequence_swap(g['sequence'])
            elif self.sequence_mutation == "urgent_pull_forward":
                sequence_mutation_stats["urgent_sequence_mutation_attempts"] = 1
                variant, pull_forward_distance = _apply_urgent_pull_forward_sequence_mutation(
                    instance=instance,
                    genome=g,
                )
                if variant is not None:
                    g['sequence'] = variant['sequence']
                    sequence_mutation_stats["urgent_sequence_mutation_applied"] = 1
                    sequence_mutation_stats["urgent_sequence_mutation_changed_position_distance_sum"] = (
                        pull_forward_distance
                    )
                elif self._apply_random_sequence_swap(g['sequence']):
                    sequence_mutation_stats["urgent_sequence_mutation_fallback_random_swap_count"] = 1
                else:
                    sequence_mutation_stats["urgent_sequence_mutation_noop_count"] = 1
            else:
                raise ValueError(f"Unsupported sequence mutation policy: {self.sequence_mutation}")
        for i in range(len(g['sequence'])):
            if self.rng.random() < self.mutation_rate:
                if i < len(ops):
                    job = instance.get_job(ops[i][0])
                    if job:
                        op = job.operations[ops[i][1]]
                        if op.eligible_machines:
                            g['machines'][i] = self.rng.choice(list(op.eligible_machines))
                        if op.eligible_workers:
                            g['workers'][i] = self.rng.choice(list(op.eligible_workers))
                        mode_choices = list(op.processing_times.get(int(g['machines'][i]), {}).keys())
                        if mode_choices:
                            g['modes'][i] = self.rng.choice(mode_choices)
                g['offsets'][i] = self.rng.integers(0, 5)
        return sequence_mutation_stats

    def _associate_to_reference(self, population: Population, fronts: List[List[int]]) -> Dict[int, int]:
        obj = population.get_objectives_matrix()
        ideal = np.min(obj, axis=0)
        denom = np.max(obj, axis=0) - ideal
        denom[denom < 1e-10] = 1.0
        norm = (obj - ideal) / denom
        assoc = {}
        for i, ind in enumerate(population.individuals):
            dist = np.linalg.norm(norm[i] - self.reference_points, axis=1)
            rp = np.argmin(dist)
            assoc[i] = rp
            ind.reference_point = self.reference_points[rp]
        return assoc

    def _select_niche(self, population: Population, assoc: Dict[int, int], counts: Dict[int, int], last_indices: List[int]) -> int:
        min_c = min(counts.values()) if counts else 0
        cand_rps = [rp for rp, count in counts.items() if count == min_c]
        if not cand_rps: return self.rng.choice(last_indices)
        rp = self.rng.choice(cand_rps)
        cands = [i for i in last_indices if assoc.get(i) == rp]
        return self.rng.choice(cands) if cands else self.rng.choice(last_indices)

    def _prefer_mating_parent(self, left: Individual, right: Individual) -> Individual:
        """Break mating ties toward hard-feasible and lower-tardiness parents."""
        if left.rank != right.rank:
            return left if left.rank < right.rank else right

        left_hard = float(left.constraint_key[0])
        right_hard = float(right.constraint_key[0])
        if left_hard <= 0.0 < right_hard:
            return left
        if right_hard <= 0.0 < left_hard:
            return right

        left_tardy_jobs = float(left.metrics.get("n_tardy_jobs", left.penalties.get("n_tardy_jobs", 0.0)))
        right_tardy_jobs = float(right.metrics.get("n_tardy_jobs", right.penalties.get("n_tardy_jobs", 0.0)))
        if abs(left_tardy_jobs - right_tardy_jobs) > 1e-9:
            return left if left_tardy_jobs < right_tardy_jobs else right

        left_tardiness = float(left.metrics.get("weighted_tardiness", left.penalties.get("weighted_tardiness", 0.0)))
        right_tardiness = float(right.metrics.get("weighted_tardiness", right.penalties.get("weighted_tardiness", 0.0)))
        if abs(left_tardiness - right_tardiness) > 1e-9:
            return left if left_tardiness < right_tardiness else right

        if abs(left.makespan - right.makespan) > 1e-9:
            return left if left.makespan < right.makespan else right

        return left if self.rng.random() < 0.5 else right

    def _select_parent_for_mating(self, population: Population) -> Individual:
        """Select one parent according to the configured mating policy."""
        if self.parent_selection == "random_pairing":
            return population[int(self.rng.integers(0, len(population)))]

        indices = self.rng.choice(len(population), size=2, replace=False)
        left = population[int(indices[0])]
        right = population[int(indices[1])]
        return self._prefer_mating_parent(left, right)

    def _individual_history_record(self, ind: Individual) -> Dict[str, Any]:
        """Serialize one evaluated individual for diagnostics and offline replay."""
        return {
            "objectives": list(ind.objectives),
            "raw_objectives": list(ind.raw_objectives),
            "penalized_objectives": list(ind.penalized_objectives),
            "penalties": dict(ind.penalties),
            "metrics": dict(ind.metrics),
            "constraint_violations": list(ind.constraint_violations),
            "constraint_key": list(ind.constraint_key),
            "is_feasible": bool(ind.is_feasible),
            "makespan": ind.makespan,
            "energy": ind.energy,
            "ergonomic_risk": ind.ergonomic_risk,
            "labor_cost": ind.labor_cost,
        }

    def _sorted_immigrant_archive_entries(
        self,
        archive: Dict[Tuple[Any, ...], ImmigrantArchiveEntry],
    ) -> List[ImmigrantArchiveEntry]:
        """Return archive entries in deterministic tardiness-first order."""
        return sorted(
            archive.values(),
            key=lambda entry: (entry.score, entry.source_generation, entry.signature),
        )

    def _archive_entry_from_individual(
        self,
        ind: Individual,
        source_generation: int,
    ) -> Optional[ImmigrantArchiveEntry]:
        """Convert one evaluated individual into a stored immigrant entry when eligible."""
        details = _details_from_individual(ind)
        if not _details_are_hard_feasible(details):
            return None
        return ImmigrantArchiveEntry(
            individual=copy.deepcopy(ind),
            signature=_genome_signature(ind.genome),
            source_generation=source_generation,
            score=_feasible_tardiness_archive_score(details),
        )

    def _update_immigrant_archive(
        self,
        archive: Dict[Tuple[Any, ...], ImmigrantArchiveEntry],
        candidates: Sequence[Individual],
        source_generation: int,
    ) -> Dict[str, int]:
        """Refresh the in-run tardiness archive from evaluated hard-feasible candidates."""
        diagnostics = _empty_immigrant_stats()
        if self.immigrant_policy == "none":
            return diagnostics
        if self.immigrant_policy != "feasible_tardiness_archive":
            raise ValueError(f"Unsupported immigrant policy: {self.immigrant_policy}")

        before_signatures = set(archive.keys())
        for candidate in candidates:
            entry = self._archive_entry_from_individual(candidate, source_generation)
            if entry is None:
                continue
            existing = archive.get(entry.signature)
            if existing is None or (entry.score, entry.source_generation, entry.signature) < (
                existing.score,
                existing.source_generation,
                existing.signature,
            ):
                archive[entry.signature] = entry

        if len(archive) > self.immigrant_archive_size:
            kept_entries = self._sorted_immigrant_archive_entries(archive)[: self.immigrant_archive_size]
            archive.clear()
            archive.update({entry.signature: entry for entry in kept_entries})

        after_signatures = set(archive.keys())
        diagnostics["immigrant_archive_admission_count"] = len(after_signatures - before_signatures)
        diagnostics["immigrant_archive_replaced_count"] = len(before_signatures - after_signatures)
        return diagnostics

    def _select_immigrant_injections(
        self,
        archive: Dict[Tuple[Any, ...], ImmigrantArchiveEntry],
        population: Population,
        offspring: Population,
    ) -> Tuple[List[Individual], Dict[str, int]]:
        """Select archive clones to append to the combined pool before survival."""
        diagnostics = _empty_immigrant_stats()
        if self.immigrant_policy == "none" or self.immigrant_count <= 0:
            return [], diagnostics
        if self.immigrant_policy != "feasible_tardiness_archive":
            raise ValueError(f"Unsupported immigrant policy: {self.immigrant_policy}")

        existing_signatures = {
            _genome_signature(ind.genome)
            for ind in population.individuals + offspring.individuals
        }
        selected_entries: List[ImmigrantArchiveEntry] = []
        selected_signatures = set()
        for entry in self._sorted_immigrant_archive_entries(archive):
            if len(selected_entries) >= self.immigrant_count:
                break
            if entry.signature in existing_signatures or entry.signature in selected_signatures:
                diagnostics["immigrant_skipped_duplicate_count"] += 1
                continue
            selected_entries.append(entry)
            selected_signatures.add(entry.signature)

        diagnostics["immigrant_injection_events"] = 1 if selected_entries else 0
        diagnostics["immigrant_injected_individuals"] = len(selected_entries)
        return [copy.deepcopy(entry.individual) for entry in selected_entries], diagnostics

    def _build_generation_snapshot(
        self,
        population: Population,
        generation: int,
        include_population_records: bool = False,
        crossover_stats: Optional[Dict[str, int]] = None,
        mutation_stats: Optional[Dict[str, int]] = None,
        repair_stats: Optional[Dict[str, int]] = None,
        immigrant_stats: Optional[Dict[str, int]] = None,
        child_lineage_stats: Optional[Dict[str, Any]] = None,
        immigrant_archive_size: int = 0,
    ) -> Dict[str, Any]:
        """Capture one generation summary, optionally with replayable population records."""
        hard_feasible = [
            ind for ind in population.individuals
            if float(ind.constraint_key[0]) <= 0.0
        ]
        hard_feasible_makespans = [
            float(ind.metrics.get("makespan", ind.makespan))
            for ind in hard_feasible
            if ind.metrics or ind.makespan
        ]
        hard_feasible_tardiness = [
            float(ind.metrics.get("weighted_tardiness", ind.penalties.get("weighted_tardiness", 0.0)))
            for ind in hard_feasible
        ]
        snapshot = {
            "generation": generation,
            "constraint_handling": self.constraint_handling,
            "crossover_policy": self.crossover_policy,
            "local_improvement": self.local_improvement,
            "sequence_mutation": self.sequence_mutation,
            "immigrant_policy": self.immigrant_policy,
            "population_size": len(population.individuals),
            "hard_feasible_count": len(hard_feasible),
            "min_hard_feasible_makespan": min(hard_feasible_makespans, default=None),
            "min_hard_feasible_weighted_tardiness": min(hard_feasible_tardiness, default=None),
            "zero_tardy_hard_feasible_count": sum(
                1
                for ind in hard_feasible
                if float(ind.metrics.get("n_tardy_jobs", ind.penalties.get("n_tardy_jobs", 0.0))) <= 0.0
            ),
            "immigrant_archive_size": int(immigrant_archive_size),
        }
        snapshot.update(_empty_sequence_mutation_stats())
        snapshot.update(_empty_local_improvement_stats())
        snapshot.update(_empty_immigrant_stats())
        snapshot.update(_empty_crossover_stats())
        snapshot.update(_empty_child_lineage_stats())
        if mutation_stats:
            snapshot.update({key: int(value) for key, value in mutation_stats.items()})
        if repair_stats:
            snapshot.update({key: int(value) for key, value in repair_stats.items()})
        if immigrant_stats:
            snapshot.update({key: int(value) for key, value in immigrant_stats.items()})
        if crossover_stats:
            snapshot.update({key: int(value) for key, value in crossover_stats.items()})
        if child_lineage_stats:
            snapshot.update(child_lineage_stats)
        if include_population_records:
            snapshot["population_records"] = [
                self._individual_history_record(ind)
                for ind in population.individuals
            ]
        return snapshot

    def _apply_local_improvement(self, ind: Individual, instance: Any) -> Dict[str, int]:
        """Optionally repair one evaluated child before survival selection."""
        diagnostics = _empty_local_improvement_stats()
        if self.local_improvement == "none":
            return diagnostics

        if self.local_improvement != "tardiness_sequence_repair":
            raise ValueError(f"Unsupported local improvement policy: {self.local_improvement}")

        if self.evaluate_details_fn is None:
            raise ValueError(
                f"Local improvement policy {self.local_improvement!r} requires a detailed evaluator."
            )

        if not ind.is_feasible or float(ind.constraint_key[0]) > 0.0:
            return diagnostics

        if float(ind.metrics.get("n_tardy_jobs", ind.penalties.get("n_tardy_jobs", 0.0))) <= 0.0:
            return diagnostics

        diagnostics["repair_attempted_children"] = 1
        base_details = self.evaluate_details_fn(instance, ind.genome)
        repaired_genome, repaired_details, repair_delta = _attempt_tardiness_sequence_repair(
            instance=instance,
            genome=ind.genome,
            base_details=base_details,
            evaluate_details_fn=self.evaluate_details_fn,
        )
        _merge_local_improvement_stats(diagnostics, repair_delta)
        if repaired_genome is not None and repaired_details is not None:
            ind.genome = repaired_genome
            self._assign_evaluation(ind, repaired_details)
        return diagnostics

    def evolve(
        self,
        instance: Any,
        verbose: bool = True,
        record_history: bool = False,
        include_population_records: bool = False,
    ) -> Population:
        self.history = []
        self.last_run_diagnostics = {
            "constraint_handling": self.constraint_handling,
            "parent_selection": self.parent_selection,
            "crossover_policy": self.crossover_policy,
            "local_improvement": self.local_improvement,
            "sequence_mutation": self.sequence_mutation,
            "immigrant_policy": self.immigrant_policy,
            "immigrant_count": self.immigrant_count,
            "immigrant_period": self.immigrant_period,
            "immigrant_archive_size": self.immigrant_archive_size,
            "immigrant_archive_final_size": 0,
            **_empty_crossover_stats(),
            **_empty_local_improvement_stats(),
            **_empty_sequence_mutation_stats(),
            **_empty_immigrant_stats(),
            **_empty_child_lineage_stats(),
        }
        if self.local_improvement != "none" and self.evaluate_details_fn is None:
            raise ValueError(
                f"Local improvement policy {self.local_improvement!r} requires a detailed evaluator."
            )
        pop = self.initialize_population(instance)
        self.evaluate_population(pop, instance)
        if self.parent_selection != "random_pairing":
            self._non_dominated_sort(pop)
        if record_history:
                self.history.append(
                    self._build_generation_snapshot(
                        pop,
                        generation=0,
                        include_population_records=include_population_records,
                        crossover_stats=_empty_crossover_stats(),
                        mutation_stats=_empty_sequence_mutation_stats(),
                        repair_stats=_empty_local_improvement_stats(),
                        immigrant_stats=_empty_immigrant_stats(),
                        child_lineage_stats=_empty_child_lineage_stats(),
                        immigrant_archive_size=0,
                    )
                )
        total_crossover_stats = _empty_crossover_stats()
        total_mutation_stats = _empty_sequence_mutation_stats()
        total_repair_stats = _empty_local_improvement_stats()
        total_immigrant_stats = _empty_immigrant_stats()
        total_child_lineage_stats = _empty_child_lineage_stats()
        immigrant_archive: Dict[Tuple[Any, ...], ImmigrantArchiveEntry] = {}
        for gen in range(self.n_generations):
            off = Population(size=self.population_size)
            generation_crossover_stats = _empty_crossover_stats()
            generation_mutation_stats = _empty_sequence_mutation_stats()
            child_parent_pairs: List[Tuple[Individual, Individual, Individual]] = []
            if self.parent_selection == "random_pairing":
                idx_list = self.rng.permutation(len(pop))
                parent_pairs = [
                    (pop[idx_list[i]], pop[idx_list[(i + 1) % len(pop)]])
                    for i in range(0, len(pop), 2)
                ]
            else:
                parent_pairs = [
                    (self._select_parent_for_mating(pop), self._select_parent_for_mating(pop))
                    for _ in range(0, len(pop), 2)
                ]
            for parent_one, parent_two in parent_pairs:
                c1, c2, crossover_stats = self._crossover(parent_one, parent_two, instance)
                _merge_crossover_stats(generation_crossover_stats, crossover_stats)
                _merge_sequence_mutation_stats(generation_mutation_stats, self._mutate(c1, instance))
                _merge_sequence_mutation_stats(generation_mutation_stats, self._mutate(c2, instance))
                off.add(c1)
                child_parent_pairs.append((c1, parent_one, parent_two))
                off.add(c2)
                child_parent_pairs.append((c2, parent_one, parent_two))
            _merge_crossover_stats(total_crossover_stats, generation_crossover_stats)
            _merge_sequence_mutation_stats(total_mutation_stats, generation_mutation_stats)
            self.evaluate_population(off, instance)
            generation_repair_stats = _empty_local_improvement_stats()
            if self.local_improvement != "none":
                for child in off.individuals:
                    _merge_local_improvement_stats(
                        generation_repair_stats,
                        self._apply_local_improvement(child, instance),
                    )
                _merge_local_improvement_stats(total_repair_stats, generation_repair_stats)
            generation_child_lineage_stats = self._collect_child_lineage_stats(
                child_parent_pairs,
                generation=gen + 1,
            )
            _merge_child_lineage_stats(total_child_lineage_stats, generation_child_lineage_stats)
            generation_immigrant_stats = _empty_immigrant_stats()
            if self.immigrant_policy != "none":
                _merge_immigrant_stats(
                    generation_immigrant_stats,
                    self._update_immigrant_archive(
                        immigrant_archive,
                        candidates=pop.individuals + off.individuals,
                        source_generation=gen + 1,
                    ),
                )
            injected_immigrants: List[Individual] = []
            if (
                self.immigrant_policy != "none"
                and (gen + 1) % self.immigrant_period == 0
            ):
                injected_immigrants, injection_stats = self._select_immigrant_injections(
                    immigrant_archive,
                    population=pop,
                    offspring=off,
                )
                _merge_immigrant_stats(generation_immigrant_stats, injection_stats)
            _merge_immigrant_stats(total_immigrant_stats, generation_immigrant_stats)

            comb = Population(size=2*self.population_size + len(injected_immigrants))
            comb.individuals = pop.individuals + off.individuals + injected_immigrants
            fronts = self._non_dominated_sort(comb)
            new_pop = Population(size=self.population_size)
            f_idx = 0
            while f_idx < len(fronts) and len(new_pop) + len(fronts[f_idx]) <= self.population_size:
                for i in fronts[f_idx]: new_pop.add(comb[i])
                f_idx += 1
            if len(new_pop) < self.population_size and f_idx < len(fronts):
                last = fronts[f_idx]; assoc = self._associate_to_reference(comb, fronts)
                counts = {}
                for i in range(len(new_pop)):
                    rp = np.argmin(np.linalg.norm(new_pop[i].reference_point - self.reference_points, axis=1))
                    counts[rp] = counts.get(rp, 0) + 1
                while len(new_pop) < self.population_size:
                    sel = self._select_niche(comb, assoc, counts, last)
                    new_pop.add(comb[sel])
                    rp = np.argmin(np.linalg.norm(comb[sel].reference_point - self.reference_points, axis=1))
                    counts[rp] = counts.get(rp, 0) + 1
            if injected_immigrants:
                injected_signatures = {_genome_signature(ind.genome) for ind in injected_immigrants}
                next_signatures = {_genome_signature(ind.genome) for ind in new_pop.individuals}
                generation_immigrant_stats["immigrant_survivors_in_next_population"] = len(
                    injected_signatures & next_signatures
                )
            pop = new_pop
            if record_history:
                self.history.append(
                    self._build_generation_snapshot(
                        pop,
                        generation=gen + 1,
                        include_population_records=include_population_records,
                        crossover_stats=generation_crossover_stats,
                        mutation_stats=generation_mutation_stats,
                        repair_stats=generation_repair_stats,
                        immigrant_stats=generation_immigrant_stats,
                        child_lineage_stats=generation_child_lineage_stats,
                        immigrant_archive_size=len(immigrant_archive),
                    )
                )
            if gen % 10 == 0 or gen == self.n_generations - 1:
                # [FIX] Select a balanced individual from the current population
                # Instead of strictly min(makespan), we normalize objectives and find
                # the one closest to the ideal point (minimum of all).
                
                all_m = np.array([ind.makespan for ind in pop.individuals])
                all_e = np.array([ind.objectives[1] for ind in pop.individuals if ind.objectives])
                
                if len(all_e) == len(all_m):
                    # Simple normalization
                    norm_m = (all_m - all_m.min()) / (all_m.max() - all_m.min() + 1e-9)
                    norm_e = (all_e - all_e.min()) / (all_e.max() - all_e.min() + 1e-9)
                    dist = np.sqrt(norm_m**2 + norm_e**2)
                    best_idx = np.argmin(dist)
                    best_ind = pop.individuals[best_idx]
                else:
                    best_ind = min(pop.individuals, key=lambda x: x.makespan)
                
                best_m = best_ind.makespan
                avg_m = np.mean([ind.makespan for ind in pop.individuals])
                if verbose: print(f"  Gen {gen:3d}: Balanced_M={best_m:10.2f}, Avg_M={avg_m:10.2f}")
        self.pareto_front = [pop.individuals[i] for i in self._non_dominated_sort(pop)[0]]
        self.last_run_diagnostics = {
            "constraint_handling": self.constraint_handling,
            "parent_selection": self.parent_selection,
            "crossover_policy": self.crossover_policy,
            "local_improvement": self.local_improvement,
            "sequence_mutation": self.sequence_mutation,
            "immigrant_policy": self.immigrant_policy,
            "immigrant_count": self.immigrant_count,
            "immigrant_period": self.immigrant_period,
            "immigrant_archive_size": self.immigrant_archive_size,
            "immigrant_archive_final_size": len(immigrant_archive),
            **total_crossover_stats,
            **total_mutation_stats,
            **total_repair_stats,
            **total_immigrant_stats,
            **total_child_lineage_stats,
        }
        return pop

    def get_pareto_solutions(self) -> List[Individual]:
        return self.pareto_front


def create_sfjssp_genome(instance: Any, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    n_ops = sum(len(j.operations) for j in instance.jobs)
    seq = []
    for j in instance.jobs: seq.extend([j.job_id] * len(j.operations))
    seq = np.array(seq); rng.shuffle(seq)
    ops = []
    for j in instance.jobs:
        for i in range(len(j.operations)): ops.append((j.job_id, i))
    mats = np.zeros(n_ops, dtype=int)
    wrks = np.zeros(n_ops, dtype=int)
    modes = np.zeros(n_ops, dtype=int)
    offs = np.zeros(n_ops, dtype=int)
    for i, (jid, oidx) in enumerate(ops):
        op = instance.get_job(jid).operations[oidx]
        em, ew = list(op.eligible_machines), list(op.eligible_workers)
        mats[i] = rng.choice(em) if em else 0
        wrks[i] = rng.choice(ew) if ew else 0
        mode_choices = list(op.processing_times.get(int(mats[i]), {}).keys())
        modes[i] = rng.choice(mode_choices) if mode_choices else 0
    return {
        'sequence': seq,
        'machines': mats,
        'workers': wrks,
        'modes': modes,
        'offsets': offs,
        'op_list': ops,
    }


def _clone_instance(instance: Any) -> Any:
    """Clone an instance through its serializer to isolate mutable solver state."""
    return instance.__class__.from_dict(instance.to_dict())


def schedule_to_sfjssp_genome(instance: Any, schedule: Any) -> Dict[str, np.ndarray]:
    """
    Convert a fully scheduled solution into an SFJSSP genome.

    The resource assignment arrays are aligned to `op_list`, while the sequence
    preserves greedy dispatch order from `schedule.add_operation(...)`.
    """
    op_list: List[Tuple[int, int]] = []
    for job in instance.jobs:
        for op_idx in range(len(job.operations)):
            op_list.append((job.job_id, op_idx))

    # Preserve insertion order: the decoder interprets `sequence` as dispatch
    # order, not realized chronological start order.
    sequence_ops = list(schedule.scheduled_ops.values())
    sequence = np.array([op.job_id for op in sequence_ops], dtype=int)

    machines = np.zeros(len(op_list), dtype=int)
    workers = np.zeros(len(op_list), dtype=int)
    modes = np.zeros(len(op_list), dtype=int)
    offsets = np.zeros(len(op_list), dtype=int)

    for idx, key in enumerate(op_list):
        scheduled_op = schedule.get_operation(*key)
        if scheduled_op is None:
            raise ValueError(f"Schedule is missing operation {key}")

        machines[idx] = int(scheduled_op.machine_id)
        workers[idx] = int(scheduled_op.worker_id)
        modes[idx] = int(scheduled_op.mode_id)
        offsets[idx] = 0

    return {
        'sequence': sequence,
        'machines': machines,
        'workers': workers,
        'modes': modes,
        'offsets': offsets,
        'op_list': op_list,
    }


def _clone_seed_genome(genome: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create a deep copy of a seed genome without mutating the original."""
    return {
        'sequence': np.array(genome['sequence'], copy=True),
        'machines': np.array(genome['machines'], copy=True),
        'workers': np.array(genome['workers'], copy=True),
        'modes': np.array(genome['modes'], copy=True),
        'offsets': np.array(genome['offsets'], copy=True),
        'op_list': list(genome['op_list']),
    }


def _seed_genome_signature(genome: Dict[str, np.ndarray]) -> Tuple[Tuple[int, ...], ...]:
    """Build a stable duplicate-detection signature for a seed genome."""
    return (
        tuple(int(value) for value in genome['sequence']),
        tuple(int(value) for value in genome['machines']),
        tuple(int(value) for value in genome['workers']),
        tuple(int(value) for value in genome['modes']),
        tuple(int(value) for value in genome['offsets']),
    )


def _build_seed_diagnostic(
    instance: Any,
    rule_name: str,
    status: str,
    rejection_reason: Optional[str] = None,
    hard_violations: Optional[Sequence[str]] = None,
    raw_objectives: Optional[Sequence[float]] = None,
    penalties: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return internal diagnostics for greedy-derived warm-start seeds."""
    return {
        'benchmark_id': getattr(instance, 'instance_id', '<unknown>'),
        'source_rule': rule_name,
        'status': status,
        'rejection_reason': rejection_reason,
        'hard_violations': list(hard_violations or []),
        'raw_objectives': list(raw_objectives) if raw_objectives is not None else None,
        'penalties': dict(penalties) if penalties is not None else None,
        'metrics': dict(metrics) if metrics is not None else None,
    }


def _operation_min_processing_time(operation: Any) -> float:
    """Return the minimum modeled processing time for one operation."""
    min_pt = float('inf')
    for machine_times in operation.processing_times.values():
        if machine_times:
            min_pt = min(min_pt, min(machine_times.values()))
    return min_pt


def _remaining_min_processing_time(instance: Any, job_id: int, op_idx: int) -> float:
    """Return the minimum remaining processing load from an operation onward."""
    job = instance.get_job(job_id)
    if job is None or op_idx >= len(job.operations):
        return float('inf')

    total = 0.0
    for remaining_idx in range(op_idx, len(job.operations)):
        total += _operation_min_processing_time(job.operations[remaining_idx])
    return total


def _sequence_operations(instance: Any, genome: Dict[str, np.ndarray]) -> List[Tuple[int, int]]:
    """Resolve each sequence slot into the operation decoded at that position."""
    next_op_idx = {job.job_id: 0 for job in instance.jobs}
    sequence_ops: List[Tuple[int, int]] = []

    for job_id in genome['sequence']:
        resolved_job_id = int(job_id)
        op_idx = next_op_idx.get(resolved_job_id, 0)
        sequence_ops.append((resolved_job_id, op_idx))
        next_op_idx[resolved_job_id] = op_idx + 1

    return sequence_ops


def _sequence_mutation_urgency_records(
    instance: Any,
    genome: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    """Score each sequence slot with a cheap due-date urgency proxy for mutation."""
    records: List[Dict[str, Any]] = []
    cumulative_min_processing = 0.0

    for position, (job_id, op_idx) in enumerate(_sequence_operations(instance, genome)):
        job = instance.get_job(job_id)
        operation = job.operations[op_idx]
        due_date = job.due_date if (job is not None and job.due_date is not None) else float('inf')
        remaining_work = _remaining_min_processing_time(instance, job_id, op_idx)
        slack = due_date - cumulative_min_processing - remaining_work
        critical_ratio = (due_date - cumulative_min_processing) / max(remaining_work, 1e-9)
        slot_processing = _operation_min_processing_time(operation)

        records.append(
            {
                'position': position,
                'job_id': job_id,
                'op_idx': op_idx,
                'occurrence_key': (job_id, op_idx),
                'due_date': due_date,
                'remaining_work': remaining_work,
                'slack': slack,
                'critical_ratio': critical_ratio,
                'slot_processing': slot_processing,
                'priority': (
                    slack,
                    critical_ratio,
                    due_date,
                    -remaining_work,
                    position,
                    job_id,
                    op_idx,
                ),
            }
        )
        cumulative_min_processing += slot_processing

    return records


def _build_urgent_prefix_merged_sequence(
    instance: Any,
    urgent_genome: Dict[str, np.ndarray],
    support_genome: Dict[str, np.ndarray],
    max_prefix_size: int = 6,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Build one deterministic urgent-prefix child sequence without extra evaluation.

    The urgent parent contributes a bounded prefix of the most due-date-critical
    operation occurrences. The support parent fills the remainder in its current
    dispatch order. Repeated job occurrences are tracked as `(job_id, op_idx)`
    identities so the output preserves the exact occurrence multiset.
    """
    urgent_records = _sequence_mutation_urgency_records(instance, urgent_genome)
    if len(urgent_records) < 2:
        return None, 0

    prefix_size = min(max(2, len(urgent_records) // 5), max_prefix_size, len(urgent_records) - 1)
    if prefix_size <= 0:
        return None, 0

    selected_occurrences = {
        tuple(record["occurrence_key"])
        for record in sorted(urgent_records, key=lambda record: record["priority"])[:prefix_size]
    }
    if not selected_occurrences:
        return None, 0

    urgent_occurrences = [
        tuple(record["occurrence_key"])
        for record in urgent_records
        if tuple(record["occurrence_key"]) in selected_occurrences
    ]
    support_occurrences = [tuple(operation) for operation in _sequence_operations(instance, support_genome)]
    child_occurrences = urgent_occurrences + [
        occurrence
        for occurrence in support_occurrences
        if occurrence not in selected_occurrences
    ]
    if len(child_occurrences) != len(urgent_records):
        return None, 0

    child_sequence = np.array([int(job_id) for job_id, _ in child_occurrences], dtype=int)
    urgent_sequence = np.array(urgent_genome["sequence"], dtype=int, copy=False)
    if np.array_equal(child_sequence, urgent_sequence):
        return None, 0

    return child_sequence, int(prefix_size)


def _apply_urgent_pull_forward_sequence_mutation(
    instance: Any,
    genome: Dict[str, np.ndarray],
    max_window: int = 3,
) -> Tuple[Optional[Dict[str, np.ndarray]], int]:
    """
    Build one deterministic urgent pull-forward variant for sequence mutation.

    The operator is intentionally cheap: it scores slots with a due-date urgency
    proxy, finds the most urgent movable slot within a bounded lookback window,
    and pulls that job occurrence forward once. Resource assignments remain
    untouched.
    """
    records = _sequence_mutation_urgency_records(instance, genome)

    for record in sorted(records, key=lambda candidate: candidate['priority']):
        position = record['position']
        if position <= 0:
            continue

        target_position: Optional[int] = None
        window_start = max(0, position - max_window)
        for earlier_pos in range(window_start, position):
            earlier = records[earlier_pos]
            if earlier['job_id'] == record['job_id']:
                continue
            if record['priority'] < earlier['priority']:
                target_position = earlier_pos
                break

        if target_position is None:
            continue

        variant = _clone_seed_genome(genome)
        moved_job = int(variant['sequence'][position])
        reduced_sequence = np.delete(variant['sequence'], position)
        variant['sequence'] = np.insert(reduced_sequence, target_position, moved_job).astype(int, copy=False)
        return variant, int(position - target_position)

    return None, 0


def _sequence_urgency_records(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
) -> List[Dict[str, Any]]:
    """Score each sequence slot by due-date urgency using the decoded schedule."""
    records: List[Dict[str, Any]] = []

    for position, (job_id, op_idx) in enumerate(_sequence_operations(instance, genome)):
        job = instance.get_job(job_id)
        scheduled_op = schedule.get_operation(job_id, op_idx) if schedule is not None else None
        start_time = scheduled_op.start_time if scheduled_op is not None else 0.0
        due_date = job.due_date if (job is not None and job.due_date is not None) else float('inf')
        remaining_work = _remaining_min_processing_time(instance, job_id, op_idx)
        slack = due_date - start_time - remaining_work
        critical_ratio = (due_date - start_time) / max(remaining_work, 1e-9)
        job_tardiness = schedule.get_job_tardiness(job_id, instance) if schedule is not None else 0.0

        records.append(
            {
                'position': position,
                'job_id': job_id,
                'op_idx': op_idx,
                'due_date': due_date,
                'start_time': start_time,
                'remaining_work': remaining_work,
                'slack': slack,
                'critical_ratio': critical_ratio,
                'job_tardiness': job_tardiness,
                'priority': (
                    0 if job_tardiness > 0 else 1,
                    slack,
                    critical_ratio,
                    due_date,
                    remaining_work,
                    position,
                    job_id,
                    op_idx,
                ),
            }
        )

    return records


def _iter_adjacent_urgent_swap_variants(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
    max_candidates: int = 3,
):
    """Yield adjacent swaps where the later dispatch slot is more urgent."""
    records = _sequence_urgency_records(instance, genome, schedule)
    inversions = []

    for idx in range(len(records) - 1):
        left = records[idx]
        right = records[idx + 1]
        if left['job_id'] == right['job_id']:
            continue
        if right['priority'] < left['priority']:
            inversions.append((right['priority'], left['priority'], idx))

    for _, _, idx in sorted(inversions)[:max_candidates]:
        variant = _clone_seed_genome(genome)
        variant['sequence'][idx], variant['sequence'][idx + 1] = (
            variant['sequence'][idx + 1],
            variant['sequence'][idx],
        )
        yield f"adjacent_urgent_swap_{idx}", variant


def _iter_pull_forward_variants(
    instance: Any,
    genome: Dict[str, np.ndarray],
    schedule: Any,
    max_candidates: int = 3,
    max_window: int = 3,
):
    """Yield local pull-forward variants for the most urgent late slots."""
    records = _sequence_urgency_records(instance, genome, schedule)
    candidate_records = sorted(records, key=lambda record: record['priority'])
    yielded = 0

    for record in candidate_records:
        position = record['position']
        if position <= 0:
            continue

        target_position: Optional[int] = None
        window_start = max(0, position - max_window)
        for earlier_pos in range(window_start, position):
            earlier = records[earlier_pos]
            if earlier['job_id'] == record['job_id']:
                continue
            if record['priority'] < earlier['priority']:
                target_position = earlier_pos
                break

        if target_position is None:
            continue

        variant = _clone_seed_genome(genome)
        moved_job = int(variant['sequence'][position])
        reduced = np.delete(variant['sequence'], position)
        variant['sequence'] = np.insert(reduced, target_position, moved_job).astype(int, copy=False)
        yield f"pull_forward_{position}_to_{target_position}", variant
        yielded += 1
        if yielded >= max_candidates:
            break


def _tardiness_seed_score(details: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """Rank candidate seed variants by tardiness first, then makespan and energy."""
    metrics = details.get('metrics', {})
    penalties = details.get('penalties', {})
    return (
        float(metrics.get('n_tardy_jobs', penalties.get('n_tardy_jobs', float('inf')))),
        float(metrics.get('weighted_tardiness', penalties.get('weighted_tardiness', float('inf')))),
        float(metrics.get('makespan', float('inf'))),
        float(metrics.get('total_energy', float('inf'))),
    )


def _collect_tardiness_seed_variants(
    instance: Any,
    rule_name: str,
    base_genome: Dict[str, np.ndarray],
    base_details: Dict[str, Any],
    max_variants_per_seed: int = 1,
) -> Tuple[List[Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]], List[Dict[str, Any]]]:
    """
    Build small, deterministic sequence variants around an accepted greedy seed.

    Variants keep the machine/worker/mode assignments fixed and only perturb the
    dispatch sequence. A candidate is accepted only if it stays hard-feasible and
    is not worse on the tardiness-first seed score.
    """
    diagnostics: List[Dict[str, Any]] = []
    accepted_variants: List[Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]] = []

    if max_variants_per_seed <= 0:
        return accepted_variants, diagnostics

    if not base_details.get('is_feasible'):
        return accepted_variants, diagnostics

    base_schedule = base_details.get('schedule')
    if base_schedule is None:
        return accepted_variants, diagnostics

    base_score = _tardiness_seed_score(base_details)
    base_metrics = base_details.get('metrics', {})
    if (
        float(base_metrics.get('n_tardy_jobs', 0)) <= 0
        and float(base_metrics.get('weighted_tardiness', 0.0)) <= 0.0
    ):
        return accepted_variants, diagnostics

    candidate_pool: List[Tuple[str, Dict[str, np.ndarray], Dict[str, Any]]] = []
    local_signatures = {_seed_genome_signature(base_genome)}

    for variant_name, candidate in list(_iter_adjacent_urgent_swap_variants(instance, base_genome, base_schedule)) + list(
        _iter_pull_forward_variants(instance, base_genome, base_schedule)
    ):
        signature = _seed_genome_signature(candidate)
        diagnostic_rule = f"{rule_name}::{variant_name}"
        if signature in local_signatures:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=diagnostic_rule,
                    status='rejected',
                    rejection_reason='duplicate_variant_candidate',
                )
            )
            continue

        local_signatures.add(signature)
        validation_instance = _clone_instance(instance)
        details = evaluate_sfjssp_genome_detailed(validation_instance, candidate)

        if not details['is_feasible']:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=diagnostic_rule,
                    status='rejected',
                    rejection_reason='decoded_hard_infeasible',
                    hard_violations=details['constraint_violations'],
                    raw_objectives=details['raw_objectives'],
                    penalties=details['penalties'],
                    metrics=details['metrics'],
                )
            )
            continue

        if _tardiness_seed_score(details) > base_score:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=diagnostic_rule,
                    status='rejected',
                    rejection_reason='not_tardiness_improving',
                    raw_objectives=details['raw_objectives'],
                    penalties=details['penalties'],
                    metrics=details['metrics'],
                )
            )
            continue

        candidate_pool.append((variant_name, candidate, details))

    candidate_pool.sort(
        key=lambda item: (
            _tardiness_seed_score(item[2]),
            item[0],
        )
    )

    for variant_name, candidate, details in candidate_pool[:max_variants_per_seed]:
        diagnostics.append(
            _build_seed_diagnostic(
                instance=instance,
                rule_name=f"{rule_name}::{variant_name}",
                status='accepted',
                raw_objectives=details['raw_objectives'],
                penalties=details['penalties'],
                metrics=details['metrics'],
            )
        )
        accepted_variants.append((variant_name, candidate, details))

    return accepted_variants, diagnostics


def _attempt_tardiness_sequence_repair(
    instance: Any,
    genome: Dict[str, np.ndarray],
    base_details: Dict[str, Any],
    evaluate_details_fn: Callable[[Any, Dict[str, np.ndarray]], Dict[str, Any]],
    max_rounds: int = 2,
    makespan_regression_cap: float = 0.03,
) -> Tuple[Optional[Dict[str, np.ndarray]], Optional[Dict[str, Any]], Dict[str, int]]:
    """
    Try a bounded, deterministic tardiness repair around one decoded genome.

    The repair only perturbs dispatch sequence slots. Machine, worker, mode, and
    offset assignments remain fixed. A candidate is accepted only if it stays
    hard-feasible, improves the tardiness-first repair score, and does not
    exceed the allowed makespan regression cap relative to the original child.
    """
    diagnostics = _empty_local_improvement_stats()

    if not _details_are_hard_feasible(base_details):
        return None, None, diagnostics

    base_metrics = base_details.get("metrics") or {}
    if float(base_metrics.get("n_tardy_jobs", 0.0)) <= 0.0:
        return None, None, diagnostics

    original_score = _tardiness_repair_score(base_details)
    original_makespan = float(base_metrics.get("makespan", float("inf")))
    current_genome = _clone_seed_genome(genome)
    current_details = base_details
    current_score = original_score
    seen_signatures = {_seed_genome_signature(current_genome)}

    for _ in range(max_rounds):
        current_schedule = current_details.get("schedule")
        if current_schedule is None:
            break

        best_variant_name: Optional[str] = None
        best_genome: Optional[Dict[str, np.ndarray]] = None
        best_details: Optional[Dict[str, Any]] = None
        best_score: Optional[Tuple[float, float, float, float, float]] = None

        variant_iterators = list(_iter_adjacent_urgent_swap_variants(instance, current_genome, current_schedule)) + list(
            _iter_pull_forward_variants(instance, current_genome, current_schedule)
        )

        for variant_name, candidate in variant_iterators:
            signature = _seed_genome_signature(candidate)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            candidate_details = evaluate_details_fn(instance, candidate)
            if not _details_are_hard_feasible(candidate_details):
                continue

            candidate_score = _tardiness_repair_score(candidate_details)
            if candidate_score >= current_score:
                continue

            candidate_makespan = float((candidate_details.get("metrics") or {}).get("makespan", float("inf")))
            allowed_makespan = original_makespan * (1.0 + makespan_regression_cap)
            if candidate_makespan > allowed_makespan + 1e-9:
                diagnostics["repair_rejected_due_to_makespan_cap_count"] += 1
                continue

            if (
                best_score is None
                or candidate_score < best_score
                or (candidate_score == best_score and variant_name < best_variant_name)
            ):
                best_variant_name = variant_name
                best_genome = candidate
                best_details = candidate_details
                best_score = candidate_score

        if best_genome is None or best_details is None or best_score is None:
            break

        current_genome = best_genome
        current_details = best_details
        current_score = best_score

    if current_score < original_score:
        diagnostics["repair_accepted_children"] = 1
        if current_score[1] < original_score[1] - 1e-9:
            diagnostics["repair_improved_n_tardy_jobs_count"] = 1
        if current_score[2] < original_score[2] - 1e-9:
            diagnostics["repair_improved_weighted_tardiness_count"] = 1
        return current_genome, current_details, diagnostics

    return None, None, diagnostics


def _collect_seed_genome_candidates(
    instance: Any,
    rules: Optional[Sequence[Callable[..., int]]] = None,
    include_tardiness_variants: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, Any]]]:
    """
    Build greedy-derived seed genomes plus internal diagnostics explaining
    which rules were accepted or rejected and why.
    """
    try:
        from ..baseline_solver.greedy_solvers import (
            GreedyScheduler,
            critical_ratio_rule,
            composite_rule,
            earliest_ready_rule,
            edt_rule,
            fifo_rule,
            least_slack_rule,
            min_energy_rule,
            min_ergonomic_rule,
            spt_rule,
            tardiness_composite_rule,
        )
    except ImportError:  # pragma: no cover - supports repo-root imports
        from baseline_solver.greedy_solvers import (
            GreedyScheduler,
            critical_ratio_rule,
            composite_rule,
            earliest_ready_rule,
            edt_rule,
            fifo_rule,
            least_slack_rule,
            min_energy_rule,
            min_ergonomic_rule,
            spt_rule,
            tardiness_composite_rule,
        )

    selected_rules = list(
        rules or (
            least_slack_rule,
            critical_ratio_rule,
            tardiness_composite_rule,
            spt_rule,
            fifo_rule,
            edt_rule,
            earliest_ready_rule,
            min_energy_rule,
            min_ergonomic_rule,
            composite_rule,
        )
    )

    expected_ops = sum(len(job.operations) for job in instance.jobs)
    accepted: List[Dict[str, np.ndarray]] = []
    diagnostics: List[Dict[str, Any]] = []
    seen_keys = set()

    for rule_fn in selected_rules:
        rule_name = getattr(rule_fn, '__name__', str(rule_fn))
        greedy_instance = _clone_instance(instance)
        schedule = GreedyScheduler(job_rule=rule_fn).schedule(greedy_instance, verbose=False)

        if not schedule.is_feasible or len(schedule.scheduled_ops) != expected_ops:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=rule_name,
                    status='rejected',
                    rejection_reason='greedy_schedule_infeasible_or_incomplete',
                    hard_violations=schedule.constraint_violations,
                )
            )
            continue

        try:
            genome = schedule_to_sfjssp_genome(greedy_instance, schedule)
        except ValueError as exc:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=rule_name,
                    status='rejected',
                    rejection_reason=f'schedule_to_genome_failed: {exc}',
                )
            )
            continue

        validation_instance = _clone_instance(instance)
        details = evaluate_sfjssp_genome_detailed(validation_instance, genome)
        if not details['is_feasible']:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=rule_name,
                    status='rejected',
                    rejection_reason='decoded_hard_infeasible',
                    hard_violations=details['constraint_violations'],
                    raw_objectives=details['raw_objectives'],
                    penalties=details['penalties'],
                    metrics=details['metrics'],
                )
            )
            continue

        signature = _seed_genome_signature(genome)
        if signature in seen_keys:
            diagnostics.append(
                _build_seed_diagnostic(
                    instance=instance,
                    rule_name=rule_name,
                    status='rejected',
                    rejection_reason='duplicate_seed',
                    raw_objectives=details['raw_objectives'],
                    penalties=details['penalties'],
                    metrics=details['metrics'],
                )
            )
            continue

        seen_keys.add(signature)
        accepted.append(genome)
        diagnostics.append(
            _build_seed_diagnostic(
                instance=instance,
                rule_name=rule_name,
                status='accepted',
                raw_objectives=details['raw_objectives'],
                penalties=details['penalties'],
                metrics=details['metrics'],
            )
        )

        if include_tardiness_variants:
            variant_candidates, variant_diagnostics = _collect_tardiness_seed_variants(
                instance=instance,
                rule_name=rule_name,
                base_genome=genome,
                base_details=details,
            )
            diagnostics.extend(variant_diagnostics)

            for variant_name, variant_genome, variant_details in variant_candidates:
                variant_signature = _seed_genome_signature(variant_genome)
                variant_rule_name = f"{rule_name}::{variant_name}"
                if variant_signature in seen_keys:
                    diagnostics.append(
                        _build_seed_diagnostic(
                            instance=instance,
                            rule_name=variant_rule_name,
                            status='rejected',
                            rejection_reason='duplicate_seed',
                            raw_objectives=variant_details['raw_objectives'],
                            penalties=variant_details['penalties'],
                            metrics=variant_details['metrics'],
                        )
                    )
                    continue

                seen_keys.add(variant_signature)
                accepted.append(variant_genome)

    return accepted, diagnostics


def create_sfjssp_seed_genomes(
    instance: Any,
    rules: Optional[Sequence[Callable[..., int]]] = None,
    include_tardiness_variants: bool = False,
) -> List[Dict[str, np.ndarray]]:
    """
    Build deterministic seed genomes from maintained greedy dispatching rules.

    Seeds are filtered aggressively:
    - the greedy schedule must be complete and hard-feasible
    - the reconstructed genome must re-evaluate without hard violations
    - tardiness-biased sequence perturbations are available only via opt-in
    """
    accepted, _ = _collect_seed_genome_candidates(
        instance,
        rules=rules,
        include_tardiness_variants=include_tardiness_variants,
    )
    return accepted


def _failed_genome_details(reason: str) -> Dict[str, Any]:
    """Return a consistent failure payload for invalid genomes."""
    return {
        'schedule': None,
        'metrics': {},
        'is_feasible': False,
        'constraint_violations': [reason],
        'penalties': {
            'hard_violations': 1,
            'n_tardy_jobs': 0,
            'weighted_tardiness': 0.0,
            'ocra_penalty': 0.0,
            'total_penalty': 1e9,
        },
        'raw_objectives': [1e6, 1e6, 1e6, 1e6],
        'penalized_objectives': [1e9, 1e9, 1e9, 1e9],
    }


def evaluate_sfjssp_genome_detailed(instance: Any, genome: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Decode a genome and return raw objectives plus legacy penalty diagnostics."""
    """
    Robust detailed evaluation function for SFJSSP genome.
    
    Args:
        instance: SFJSSPInstance
        genome: Dict containing 'sequence', 'machines', 'workers', 'modes', 'offsets', 'op_list'
        
    Returns:
        Dict containing:
        - schedule: decoded Schedule or None
        - metrics: raw schedule-level metrics from Schedule.evaluate(instance)
        - is_feasible: hard-feasibility result from Schedule.check_feasibility(instance)
        - penalties: penalty components used for NSGA selection
        - raw_objectives: [makespan, energy, ocra, labor] before penalties
        - penalized_objectives: legacy scalar-penalized objectives kept for diagnostics
    """
    try:
        from ..sfjssp_model.schedule import Schedule
    except ImportError:  # pragma: no cover - supports repo-root imports
        from sfjssp_model.schedule import Schedule
    
    # 1. Reset all resource states
    for m in instance.machines:
        m.reset()
    for w in instance.workers:
        w.reset()
        
    # 2. Extract genome components
    job_seq = genome['sequence']
    machine_indices = genome['machines']
    worker_indices = genome['workers']
    op_list = genome['op_list']  # List of (job_id, op_id)
    mode_indices = genome.get('modes', np.zeros(len(op_list), dtype=int))
    offsets = genome['offsets']
    
    # 3. Create operation-to-resource mapping
    # This ensures that even if sequence order changes, the specific 
    # resource assigned to (job_id, op_id) stays with it.
    op_resource_map = {}
    for i, (jid, oid) in enumerate(op_list):
        op_resource_map[(jid, oid)] = {
            'm_id': int(machine_indices[i]),
            'w_id': int(worker_indices[i]),
            'mode_id': int(mode_indices[i]) if i < len(mode_indices) else 0,
            'offset': int(offsets[i])
        }
        
    schedule = Schedule(instance_id=instance.instance_id)
    job_op_ptr = {j.job_id: 0 for j in instance.jobs}
    job_last_completion = {j.job_id: 0.0 for j in instance.jobs}
    clock = instance.period_clock
    
    # 4. Decode sequence
    for job_id in job_seq:
        job_id = int(job_id)
        job = instance.get_job(job_id)
        op_idx = job_op_ptr[job_id]
        if op_idx >= len(job.operations):
            continue
            
        op = job.operations[op_idx]
        
        # Retrieve resources assigned to THIS specific operation
        res = op_resource_map.get((job_id, op_idx))
        if res is None:
            # Fallback if op_list is inconsistent (should not happen with pox crossover)
            m_id = list(op.eligible_machines)[0]
            w_id = list(op.eligible_workers)[0]
            mode_id = list(op.processing_times.get(m_id, {0: 0.0}).keys())[0]
            offset = 0
        else:
            m_id = res['m_id']
            w_id = res['w_id']
            mode_id = res['mode_id']
            offset = res['offset']

        if m_id not in op.eligible_machines:
            return _failed_genome_details(f"Ineligible machine assignment for ({job_id}, {op_idx})")
        if w_id not in op.eligible_workers:
            return _failed_genome_details(f"Ineligible worker assignment for ({job_id}, {op_idx})")

        mode_times = op.processing_times.get(m_id, {})
        if not mode_times:
            return _failed_genome_details(f"No processing times for machine {m_id} on ({job_id}, {op_idx})")
        if mode_id not in mode_times:
            mode_id = next(iter(mode_times))
            
        machine = instance.get_machine(m_id)
        worker = instance.get_worker(w_id)
        if machine is None or worker is None:
            return _failed_genome_details(f"Missing machine/worker resource for ({job_id}, {op_idx})")
        
        # Determine earliest possible start time
        est = max(
            job.arrival_time,
            machine.available_time + machine.setup_time,
            worker.available_time,
            worker.mandatory_shift_lockout_until,
            job_last_completion[job_id]
        )
        
        # Apply shift-skipping offset
        if offset > 0:
            est = clock.period_start(clock.get_period(est) + offset)
            
        # Refine start time to satisfy hard constraints
        pt = op.get_processing_time(m_id, mode_id, worker.get_efficiency())
        risk_rate = instance.get_ergonomic_risk(job_id, op_idx)
        
        found = False
        curr_t = est
        for _ in range(50):
            # 1. Machine gap check (centralized)
            m_valid, m_next = machine.validate_gap(curr_t, machine.setup_time)
            if not m_valid:
                curr_t = max(curr_t, m_next)
                continue

            # 2. Worker assignment check (centralized Industry 5.0 engine)
            w_valid, w_next = worker.validate_assignment(curr_t, pt, risk_rate)
            if not w_valid:
                curr_t = max(curr_t, w_next)
                continue
                
            found = True
            break
            
        if not found:
            return _failed_genome_details(f"No feasible start time found for ({job_id}, {op_idx})")

        if curr_t > worker.available_time:
            worker.record_rest(curr_t - worker.available_time)

        # 5. Record operation
        schedule.add_operation(
            job_id, op_idx, m_id, w_id, mode_id,
            curr_t, curr_t + pt, pt, 
            machine.setup_time, getattr(op, 'transport_time', 0.0)
        )
        
        # 6. Update states
        machine.available_time = curr_t + pt
        worker.available_time = curr_t + pt
        # job_last_completion still needs local tracking or updating job objects
        job_last_completion[job_id] = curr_t + pt + getattr(op, 'transport_time', 0.0) + getattr(op, 'waiting_time', 0.0)
        worker.record_work(pt, risk_rate, curr_t, operation_type=op_idx)
        job_op_ptr[job_id] += 1
        
    # 7. Evaluate complete schedule
    metrics = schedule.evaluate(instance)
    is_feasible = schedule.check_feasibility(instance)
    
    # 8. Calculate penalties
    hard_violations = len(schedule.constraint_violations)
    tardiness_penalty = metrics.get('weighted_tardiness', 0.0)
    n_tardy = metrics.get('n_tardy_jobs', 0)
            
    # Ergonomic penalty if OCRA > threshold
    max_ocra = metrics.get('max_ergonomic_exposure', 0.0)
    ocra_threshold = getattr(instance, 'ocra_max_per_shift', 2.2)
    # [FIX] Scale penalty to be significant but not instantly front-collapsing (1e4 instead of 1e6)
    ocra_penalty = max(0.0, max_ocra - ocra_threshold) * 1e4
    
    # Balanced penalty structure
    total_penalty = (
        (hard_violations * 1e6)
        + (n_tardy * 1e3)
        + (tardiness_penalty * 10.0)
        + ocra_penalty
    )

    raw_objectives = [
        metrics.get('makespan', 1e6),
        metrics.get('total_energy', 1e6),
        max_ocra,
        metrics.get('total_labor_cost', 1e6),
    ]
    penalized_objectives = [
        raw_objectives[0] + total_penalty,
        raw_objectives[1] + total_penalty,
        raw_objectives[2] + (total_penalty / 1e6),  # Keep OCRA ranking scale readable
        raw_objectives[3] + total_penalty,
    ]

    return {
        'schedule': schedule,
        'metrics': metrics,
        'is_feasible': is_feasible,
        'constraint_violations': list(schedule.constraint_violations),
        'penalties': {
            'hard_violations': hard_violations,
            'n_tardy_jobs': n_tardy,
            'weighted_tardiness': tardiness_penalty,
            'ocra_penalty': ocra_penalty,
            'total_penalty': total_penalty,
        },
        'raw_objectives': raw_objectives,
        'penalized_objectives': penalized_objectives,
    }


def evaluate_sfjssp_genome(instance: Any, genome: Dict[str, np.ndarray]) -> List[float]:
    """Return the legacy scalar-penalized objective vector."""
    return evaluate_sfjssp_genome_detailed(instance, genome)['penalized_objectives']
