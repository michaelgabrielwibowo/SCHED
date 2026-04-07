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
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass, field
import copy


@dataclass
class Individual:
    """
    Individual solution in the population

    For SFJSSP, the genome encodes:
    - Operation sequence (permutation)
    - Machine assignment (for each operation)
    - Worker assignment (for each operation)
    """
    genome: Dict[str, np.ndarray]  # {'sequence': ..., 'machines': ..., 'workers': ...}
    objectives: List[float] = field(default_factory=list)
    rank: int = 0
    niche_count: int = 0
    reference_point: Optional[np.ndarray] = None

    # For SFJSSP evaluation
    makespan: float = 0.0
    energy: float = 0.0
    ergonomic_risk: float = 0.0
    labor_cost: float = 0.0

    def __lt__(self, other):
        """Compare by rank (lower is better)"""
        return self.rank < other.rank


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
        """Get objectives as matrix (n_individuals x n_objectives)"""
        return np.array([ind.objectives for ind in self.individuals])

    def get_best(self, objective_idx: int = 0) -> Individual:
        """Get individual with best value for given objective"""
        if not self.individuals:
            return None
        best_idx = np.argmin([ind.objectives[objective_idx] for ind in self.individuals])
        return self.individuals[best_idx]


class NSGA3:
    """
    NSGA-III Multi-Objective Evolutionary Algorithm

    Features:
    - Reference point-based selection for many objectives
    - Non-dominated sorting
    - Crowding distance for diversity

    Evidence: NSGA-III confirmed from Deb & Jain (2014)

    Args:
        n_objectives: Number of objectives (typically 4-6 for SFJSSP)
        population_size: Population size (should be multiple of 4 for reference points)
        n_generations: Number of generations to evolve
        mutation_rate: Probability of mutation
        crossover_rate: Probability of crossover
    """

    def __init__(
        self,
        n_objectives: int = 4,
        population_size: int = 100,
        n_generations: int = 100,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
        seed: int = 42,
    ):
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Generate reference points
        self.reference_points = self._generate_reference_points()

        # Problem-specific functions (set by user)
        self.evaluate_fn: Optional[Callable] = None
        self.create_individual_fn: Optional[Callable] = None

        # Tracking
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []

    def _generate_reference_points(self, n_divisions: int = 12) -> np.ndarray:
        """
        Generate well-distributed reference points using Das and Dennis method

        Evidence: Reference point generation from NSGA-III paper [CONFIRMED]
        """
        # For M objectives and p divisions, generate combinations
        # Number of reference points = C(M + p - 1, p)

        points = []
        self._generate_reference_recursive(
            np.zeros(self.n_objectives),
            0,
            n_divisions,
            points
        )

        return np.array(points) / n_divisions

    def _generate_reference_recursive(
        self,
        point: np.ndarray,
        obj_idx: int,
        remaining: int,
        points: List[np.ndarray]
    ):
        """Recursive helper for reference point generation"""
        if obj_idx == self.n_objectives - 1:
            point[obj_idx] = remaining
            points.append(point.copy())
        else:
            for i in range(remaining + 1):
                point[obj_idx] = i
                self._generate_reference_recursive(
                    point, obj_idx + 1, remaining - i, points
                )

    def set_problem(
        self,
        evaluate_fn: Callable,
        create_individual_fn: Callable
    ):
        """
        Set problem-specific functions

        Args:
            evaluate_fn: Function(instance, genome) -> list of objectives
            create_individual_fn: Function(instance, rng) -> genome dict
        """
        self.evaluate_fn = evaluate_fn
        self.create_individual_fn = create_individual_fn

    def initialize_population(self, instance: Any) -> Population:
        """Initialize population with random individuals"""
        population = Population(size=self.population_size)

        for _ in range(self.population_size):
            genome = self.create_individual_fn(instance, self.rng)
            ind = Individual(genome=genome)
            population.add(ind)

        return population

    def evaluate_population(self, population: Population, instance: Any):
        """Evaluate all individuals in population"""
        for ind in population.individuals:
            objectives = self.evaluate_fn(instance, ind.genome)
            ind.objectives = objectives

            # Store individual objective values (for SFJSSP)
            if len(objectives) >= 4:
                ind.makespan = objectives[0]
                ind.energy = objectives[1]
                ind.ergonomic_risk = objectives[2]
                ind.labor_cost = objectives[3]

    def _non_dominated_sort(self, population: Population) -> List[List[int]]:
        """
        Perform non-dominated sorting

        Returns:
            List of fronts, each front is a list of indices
        """
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        # Compare all pairs
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i].objectives, population[j].objectives):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j].objectives, population[i].objectives):
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # First front (non-dominated solutions)
        for i in range(n):
            if domination_count[i] == 0:
                population[i].rank = 0
                fronts[0].append(i)

        # Build subsequent fronts
        front_idx = 0
        while fronts[front_idx]:
            next_front = []
            for i in fronts[front_idx]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        population[j].rank = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            fronts.append(next_front)

        return fronts[:-1]  # Remove empty last front

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        """Check if obj1 dominates obj2 (all objectives minimized)"""
        better_in_one = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2:  # Worse in this objective
                return False
            if o1 < o2:  # Better in this objective
                better_in_one = True
        return better_in_one

    def _tournament_selection(
        self,
        population: Population,
        fronts: List[List[int]]
    ) -> Individual:
        """Select parent using binary tournament"""
        idx1, idx2 = self.rng.choice(len(population), size=2, replace=False)

        # Compare by rank
        rank1 = population[idx1].rank
        rank2 = population[idx2].rank

        if rank1 < rank2:
            return population[idx1]
        elif rank2 < rank1:
            return population[idx2]
        else:
            # Same rank: random choice
            return population[self.rng.choice([idx1, idx2])]

    def _crossover(
        self,
        parent1: Individual,
        parent2: Individual
    ) -> Tuple[Individual, Individual]:
        """
        Perform crossover between two parents

        Uses order crossover (OX) for sequence and uniform crossover for assignments
        """
        genome1 = parent1.genome
        genome2 = parent2.genome

        if self.rng.random() > self.crossover_rate:
            # No crossover: return copies
            return (
                Individual(genome=copy.deepcopy(genome1)),
                Individual(genome=copy.deepcopy(genome2))
            )

        # Order crossover for sequence
        seq1 = genome1['sequence'].copy()
        seq2 = genome2['sequence'].copy()

        n = len(seq1)
        start, end = sorted(self.rng.choice(n, size=2, replace=False))

        child1_seq = np.full(n, -1, dtype=int)
        child2_seq = np.full(n, -1, dtype=int)

        # Copy segment from parents
        child1_seq[start:end] = seq1[start:end]
        child2_seq[start:end] = seq2[start:end]

        # Fill remaining positions
        def fill_remaining(child_seq, parent_seq, start, end):
            result = child_seq.copy()
            fill_pos = end
            for i in range(n):
                pos = (end + i) % n
                if parent_seq[pos] not in child_seq:
                    while result[fill_pos % n] != -1:
                        fill_pos += 1
                    result[fill_pos % n] = parent_seq[pos]
            return result

        child1_seq = fill_remaining(child1_seq, seq2, start, end)
        child2_seq = fill_remaining(child2_seq, seq1, start, end)

        # Uniform crossover for machine/worker assignments
        mask = self.rng.random(len(genome1['machines'])) < 0.5

        child1_machines = np.where(mask, genome1['machines'], genome2['machines'])
        child1_workers = np.where(mask, genome1['workers'], genome2['workers'])

        child2_machines = np.where(mask, genome2['machines'], genome1['machines'])
        child2_workers = np.where(mask, genome2['workers'], genome1['workers'])

        # Copy op_list from parent (same for both children)
        op_list = genome1.get('op_list', genome2.get('op_list', []))

        child1 = Individual(genome={
            'sequence': child1_seq,
            'machines': child1_machines,
            'workers': child1_workers,
            'op_list': op_list,
        })

        child2 = Individual(genome={
            'sequence': child2_seq,
            'machines': child2_machines,
            'workers': child2_workers,
            'op_list': op_list,
        })

        return child1, child2

    def _mutate(self, individual: Individual, instance: Any):
        """
        Apply mutation operators

        - Swap mutation for sequence
        - Random reset for machine/worker assignments (eligibility-aware)
        """
        genome = individual.genome
        op_list = genome.get('op_list', [])

        # Sequence mutation (swap)
        if self.rng.random() < self.mutation_rate:
            idx1, idx2 = self.rng.choice(len(genome['sequence']), size=2, replace=False)
            genome['sequence'][[idx1, idx2]] = genome['sequence'][[idx2, idx1]]

        # Machine mutation — sample from eligible set
        for i in range(len(genome['machines'])):
            if self.rng.random() < self.mutation_rate:
                if i < len(op_list):
                    job_id, op_idx = op_list[i]
                    job = instance.get_job(job_id)
                    if job is not None and op_idx < len(job.operations):
                        eligible_m = list(job.operations[op_idx].eligible_machines)
                        if eligible_m:
                            genome['machines'][i] = self.rng.choice(eligible_m)
                            continue
                # Fallback if op_list is missing or op not found
                genome['machines'][i] = self.rng.integers(0, instance.n_machines)

        # Worker mutation — sample from eligible set
        for i in range(len(genome['workers'])):
            if self.rng.random() < self.mutation_rate:
                if i < len(op_list):
                    job_id, op_idx = op_list[i]
                    job = instance.get_job(job_id)
                    if job is not None and op_idx < len(job.operations):
                        eligible_w = list(job.operations[op_idx].eligible_workers)
                        if eligible_w:
                            genome['workers'][i] = self.rng.choice(eligible_w)
                            continue
                genome['workers'][i] = self.rng.integers(0, instance.n_workers)

    def _associate_to_reference(
        self,
        population: Population,
        fronts: List[List[int]]
    ) -> Dict[int, int]:
        """Associate each individual with nearest reference point"""
        # Normalize objectives
        obj_matrix = population.get_objectives_matrix()
        ideal_point = np.min(obj_matrix, axis=0)
        normalized = (obj_matrix - ideal_point) / (np.max(obj_matrix, axis=0) - ideal_point + 1e-10)

        # Find closest reference point for each individual
        associations = {}
        for i, ind in enumerate(population.individuals):
            distances = np.linalg.norm(normalized[i] - self.reference_points, axis=1)
            closest_rp = np.argmin(distances)
            associations[i] = closest_rp
            ind.reference_point = self.reference_points[closest_rp]

        return associations

    def _select_niche(
        self,
        population: Population,
        associations: Dict[int, int],
        niche_counts: Dict[int, int],
        last_front_indices: List[int]
    ) -> int:
        """Select individual from least crowded niche"""
        # Find minimum niche count
        min_count = min(niche_counts.values()) if niche_counts else 0

        # Get reference points with minimum count
        candidate_rps = [rp for rp, count in niche_counts.items() if count == min_count]

        if not candidate_rps:
            return self.rng.choice(last_front_indices)

        # Select random candidate RP
        selected_rp = self.rng.choice(candidate_rps)

        # Find individuals associated with this RP in the last front
        candidates = [
            i for i in last_front_indices
            if associations.get(i) == selected_rp
        ]

        if candidates:
            return self.rng.choice(candidates)

        return None

    def evolve(self, instance: Any, verbose: bool = True) -> Population:
        """
        Run NSGA-III optimization
        """
        if self.evaluate_fn is None or self.create_individual_fn is None:
            raise ValueError("Must call set_problem() before evolve()")

        # Initialize
        population = self.initialize_population(instance)
        self.evaluate_population(population, instance)

        for gen in range(self.n_generations):
            # Create offspring
            offspring = Population(size=self.population_size)

            fronts = self._non_dominated_sort(population)

            for _ in range(self.population_size // 2):
                parent1 = self._tournament_selection(population, fronts)
                parent2 = self._tournament_selection(population, fronts)

                child1, child2 = self._crossover(parent1, parent2)

                self._mutate(child1, instance)
                self._mutate(child2, instance)

                offspring.add(child1)
                offspring.add(child2)

            # Evaluate offspring
            self.evaluate_population(offspring, instance)

            # Combine parent and offspring
            combined = Population(size=2 * self.population_size)
            combined.individuals = population.individuals + offspring.individuals

            # Non-dominated sorting
            fronts = self._non_dominated_sort(combined)

            # Build next generation
            new_population = Population(size=self.population_size)
            front_idx = 0

            while len(new_population) + len(fronts[front_idx]) <= self.population_size:
                for i in fronts[front_idx]:
                    new_population.add(combined[i])
                front_idx += 1
                if front_idx >= len(fronts):
                    break

            # If we need more individuals from the last front
            if len(new_population) < self.population_size and front_idx < len(fronts):
                last_front = fronts[front_idx]

                # Associate with reference points
                associations = self._associate_to_reference(combined, fronts)

                # Count niche members
                niche_counts = {}
                for i in range(len(new_population)):
                    rp_idx = np.argmin(
                        np.linalg.norm(
                            new_population[i].reference_point - self.reference_points,
                            axis=1
                        )
                    )
                    niche_counts[rp_idx] = niche_counts.get(rp_idx, 0) + 1

                # Select remaining individuals
                while len(new_population) < self.population_size:
                    selected = self._select_niche(
                        combined, associations, niche_counts, last_front
                    )
                    if selected is None:
                        break
                    new_population.add(combined[selected])

                    # Update niche count
                    rp_idx = np.argmin(
                        np.linalg.norm(
                            combined[selected].reference_point - self.reference_points,
                            axis=1
                        )
                    )
                    niche_counts[rp_idx] = niche_counts.get(rp_idx, 0) + 1

            population = new_population

            # Track history
            if gen % 10 == 0 or gen == self.n_generations - 1:
                best_makespan = min(ind.makespan for ind in population.individuals)
                best_energy = min(ind.energy for ind in population.individuals)
                if verbose:
                    print(f"  Gen {gen}: makespan={best_makespan:.1f}, energy={best_energy:.1f}")

                self.history.append({
                    'generation': gen,
                    'best_makespan': best_makespan,
                    'best_energy': best_energy,
                })

        # Extract Pareto front
        fronts = self._non_dominated_sort(population)
        self.pareto_front = [population.individuals[i] for i in fronts[0]]

        if verbose:
            print(f"\nPareto front size: {len(self.pareto_front)}")

        return population

    def get_pareto_solutions(self) -> List[Individual]:
        """Get Pareto-optimal solutions"""
        return self.pareto_front


def create_sfjssp_genome(instance: Any, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """
    Create random genome for SFJSSP
    """
    # Count total operations
    n_ops = sum(len(job.operations) for job in instance.jobs)

    # Create operation list (flattened)
    op_list = []
    for job in instance.jobs:
        for op_idx in range(len(job.operations)):
            op_list.append((job.job_id, op_idx))

    # Random sequence (permutation of operations)
    sequence = rng.permutation(n_ops)

    # Eligibility-aware machine and worker assignment
    machines = np.zeros(n_ops, dtype=int)
    workers  = np.zeros(n_ops, dtype=int)

    for i, (job_id, op_idx) in enumerate(op_list):
        job = instance.get_job(job_id)
        if job is None or op_idx >= len(job.operations):
            continue
        op = job.operations[op_idx]

        eligible_m = list(op.eligible_machines)
        eligible_w = list(op.eligible_workers)

        machines[i] = rng.choice(eligible_m) if eligible_m else instance.machines[0].machine_id
        workers[i]  = rng.choice(eligible_w) if eligible_w else instance.workers[0].worker_id

    return {
        'sequence': sequence,
        'machines': machines,
        'workers': workers,
        'op_list': op_list,
    }


def evaluate_sfjssp_genome(instance: Any, genome: Dict[str, np.ndarray]) -> List[float]:
    """
    Evaluate genome for SFJSSP
    """
    from sfjssp_model.schedule import Schedule

    sequence = genome['sequence']
    machines = genome['machines']
    workers = genome['workers']
    op_list = genome.get('op_list', [])

    if not op_list:
        return [0.0, 0.0, 0.0, 0.0]

    assert len(op_list) == sum(len(job.operations) for job in instance.jobs), \
        "op_list misalignment"

    # Build schedule
    schedule = Schedule(instance_id=instance.instance_id)

    machine_available = {m.machine_id: 0.0 for m in instance.machines}
    worker_available = {w.worker_id: 0.0 for w in instance.workers}
    job_last_op_completion = {}

    for seq_pos in range(len(sequence)):
        op_list_idx = sequence[seq_pos]
        if op_list_idx >= len(op_list):
            continue

        job_id, local_op_idx = op_list[op_list_idx]
        job = instance.get_job(job_id)
        op = job.operations[local_op_idx]

        m_id = int(machines[seq_pos] % instance.n_machines)
        w_id = int(workers[seq_pos] % instance.n_workers)

        # Eligibility check
        if m_id not in op.eligible_machines:
            if not op.eligible_machines: continue
            eligible_m = list(op.eligible_machines)
            m_id = eligible_m[int(machines[seq_pos]) % len(eligible_m)]
            machines[seq_pos] = m_id

        if w_id not in op.eligible_workers:
            if not op.eligible_workers: continue
            eligible_w = list(op.eligible_workers)
            w_id = eligible_w[int(workers[seq_pos]) % len(eligible_w)]
            workers[seq_pos] = w_id

        machine = instance.get_machine(m_id)
        worker = instance.get_worker(w_id)

        # Calculate earliest start
        earliest_start = max(
            machine_available.get(m_id, 0),
            worker_available.get(w_id, 0),
            worker.mandatory_shift_lockout_until
        )

        if local_op_idx > 0:
            prev_completion = job_last_op_completion.get((job_id, local_op_idx - 1), 0)
            earliest_start = max(earliest_start, prev_completion)

        start_time = earliest_start

        # Record rest
        rest_duration = max(0.0, start_time - worker_available.get(w_id, 0))
        if rest_duration > 0:
            worker.record_rest(rest_duration)

        # Base pt
        mode_id = 0
        if m_id in op.processing_times and mode_id in op.processing_times[m_id]:
            base_pt = op.processing_times[m_id][mode_id]
        else:
            base_pt = 50.0
            
        pt = base_pt / max(0.1, worker.get_efficiency())
        completion_time = start_time + pt

        # [FIX] Enforce "A task cannot span two periods" and "no consecutive periods"
        clock = instance.period_clock
        max_tries = 10
        found = False
        for _ in range(max_tries):
            if clock.crosses_boundary(start_time, completion_time):
                start_time = clock.period_start(clock.get_period(start_time) + 1)
                pt = base_pt / max(0.1, worker.get_efficiency())
                completion_time = start_time + pt
                continue
            
            if not worker.can_work_in_period(start_time, completion_time):
                start_time = clock.period_start(clock.get_period(start_time) + 1)
                pt = base_pt / max(0.1, worker.get_efficiency())
                completion_time = start_time + pt
                continue
            
            found = True
            break
        
        # Check for mandatory 12.5% rest rule
        mandatory_rest = worker.requires_mandatory_rest(
            proposed_task_duration=pt, 
            current_time=start_time
        )
        if mandatory_rest > 0:
            start_time += mandatory_rest
            worker.record_rest(mandatory_rest)
            completion_time = start_time + pt

        # Final check if rest pushed us across a boundary (limit jumps to 2 for NSGA)
        jumps = 0
        while clock.crosses_boundary(start_time, completion_time) and jumps < 2:
            start_time = clock.period_start(clock.get_period(start_time) + 1)
            completion_time = start_time + pt
            jumps += 1

        op.start_time = start_time
        op.completion_time = completion_time
        try:
            op.assign_period_bounds(instance.period_clock)
        except ValueError:
            # If still crosses (very long task), we skip adding to schedule or just ignore for MOEA
            # MOEA can handle infeasible solutions via penalty
            pass

        schedule.add_operation(
            job_id=job_id,
            op_id=local_op_idx,
            machine_id=m_id,
            worker_id=w_id,
            mode_id=mode_id,
            start_time=start_time,
            completion_time=completion_time,
            processing_time=pt,
        )

        risk_rate = instance.get_ergonomic_risk(job_id, local_op_idx)
        worker.record_work(pt, risk_rate=risk_rate, current_time=start_time)

        if machine.total_processing_time == 0.0:
            machine.startup_count += 1
        machine.total_processing_time += pt

        machine_available[m_id] = completion_time
        worker_available[w_id] = completion_time
        job_last_op_completion[(job_id, local_op_idx)] = completion_time

    # Evaluate schedule
    if not schedule.scheduled_ops:
        return [1e9, 1e9, 1e9, 1e9]

    schedule.compute_makespan()
    schedule.compute_total_energy(instance)
    schedule.compute_ergonomic_metrics(instance)

    labor_cost = 0.0
    for w_id, w_sched in schedule.worker_schedules.items():
        worker = instance.get_worker(w_id)
        if worker:
            total_time = sum(op.processing_time for op in w_sched.operations)
            labor_cost += worker.labor_cost_per_hour * total_time

    max_ocra = schedule.ergonomic_metrics.get('max_exposure', 0.0)
    constraint_penalty = 0.0
    if max_ocra >= 2.2:
        constraint_penalty += 1e6 * (max_ocra - 2.2)
    
    if not schedule.check_feasibility(instance):
        constraint_penalty += 1e6 * len(schedule.constraint_violations)

    return [
        schedule.makespan + constraint_penalty,
        schedule.energy_breakdown.get('total', 1000.0) + constraint_penalty,
        max_ocra,  
        labor_cost + constraint_penalty,
    ]
