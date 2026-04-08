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
    """
    genome: Dict[str, np.ndarray]
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
    ):
        self.n_objectives = n_objectives
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.reference_points = self._generate_reference_points()
        self.evaluate_fn: Optional[Callable] = None
        self.create_individual_fn: Optional[Callable] = None
        self.pareto_front: List[Individual] = []
        self.history: List[Dict] = []

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

    def set_problem(self, evaluate_fn: Callable, create_individual_fn: Callable):
        self.evaluate_fn = evaluate_fn
        self.create_individual_fn = create_individual_fn

    def initialize_population(self, instance: Any) -> Population:
        pop = Population(size=self.population_size)
        for _ in range(self.population_size):
            genome = self.create_individual_fn(instance, self.rng)
            pop.add(Individual(genome=genome))
        return pop

    def evaluate_population(self, population: Population, instance: Any):
        for ind in population.individuals:
            objectives = self.evaluate_fn(instance, ind.genome)
            ind.objectives = objectives
            if len(objectives) >= 4:
                ind.makespan = objectives[0]
                ind.energy = objectives[1]
                ind.ergonomic_risk = objectives[2]
                ind.labor_cost = objectives[3]

    def _non_dominated_sort(self, population: Population) -> List[List[int]]:
        n = len(population)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i].objectives, population[j].objectives):
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j].objectives, population[i].objectives):
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

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        better = False
        for o1, o2 in zip(obj1, obj2):
            if o1 > o2 + 1e-7: return False
            if o1 < o2 - 1e-7: better = True
        return better

    def _crossover(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        if self.rng.random() > self.crossover_rate:
            return Individual(genome=copy.deepcopy(p1.genome)), Individual(genome=copy.deepcopy(p2.genome))
        
        g1, g2 = p1.genome, p2.genome
        seq1, seq2 = g1['sequence'].copy(), g2['sequence'].copy()
        job_ids = list(set(seq1))
        if len(job_ids) > 1:
            sub = set(self.rng.choice(job_ids, size=self.rng.integers(1, len(job_ids)), replace=False))
            def pox(s1, s2):
                c = np.full(len(s1), -1, dtype=int)
                for i in range(len(s1)):
                    if s1[i] in sub: c[i] = s1[i]
                ptr = 0
                for i in range(len(s1)):
                    if c[i] == -1:
                        while ptr < len(s2) and s2[ptr] in sub: ptr += 1
                        if ptr < len(s2):
                            c[i] = s2[ptr]
                            ptr += 1
                return c
            c1_seq, c2_seq = pox(seq1, seq2), pox(seq2, seq1)
        else:
            c1_seq, c2_seq = seq1.copy(), seq2.copy()

        mask = self.rng.random(len(g1['machines'])) < 0.5
        c1_m, c2_m = g1['machines'].copy(), g2['machines'].copy()
        c1_w, c2_w = g1['workers'].copy(), g2['workers'].copy()
        c1_o, c2_o = g1['offsets'].copy(), g2['offsets'].copy()
        
        # [FIX] Machines/Workers are mapped to the operation at that index in op_list
        # We must swap by operation, not by sequence position
        for i in range(len(mask)):
            if mask[i]:
                c1_m[i], c2_m[i] = g2['machines'][i], g1['machines'][i]
                c1_w[i], c2_w[i] = g2['workers'][i], g1['workers'][i]
                c1_o[i], c2_o[i] = g2['offsets'][i], g1['offsets'][i]

        return Individual(genome={'sequence': c1_seq, 'machines': c1_m, 'workers': c1_w, 'offsets': c1_o, 'op_list': g1['op_list']}), \
               Individual(genome={'sequence': c2_seq, 'machines': c2_m, 'workers': c2_w, 'offsets': c2_o, 'op_list': g1['op_list']})

    def _mutate(self, ind: Individual, instance: Any):
        g = ind.genome
        ops = g['op_list']
        if self.rng.random() < self.mutation_rate:
            i1, i2 = self.rng.choice(len(g['sequence']), 2, replace=False)
            g['sequence'][i1], g['sequence'][i2] = g['sequence'][i2], g['sequence'][i1]
            # [FIX] Resources must move with the job in the sequence if using sequence-index mapping
            g['machines'][i1], g['machines'][i2] = g['machines'][i2], g['machines'][i1]
            g['workers'][i1], g['workers'][i2] = g['workers'][i2], g['workers'][i1]
            g['offsets'][i1], g['offsets'][i2] = g['offsets'][i2], g['offsets'][i1]
        for i in range(len(g['sequence'])):
            if self.rng.random() < self.mutation_rate:
                if i < len(ops):
                    job = instance.get_job(ops[i][0])
                    if job:
                        op = job.operations[ops[i][1]]
                        if op.eligible_machines: g['machines'][i] = self.rng.choice(list(op.eligible_machines))
                        if op.eligible_workers: g['workers'][i] = self.rng.choice(list(op.eligible_workers))
                g['offsets'][i] = self.rng.integers(0, 5)

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

    def evolve(self, instance: Any, verbose: bool = True) -> Population:
        pop = self.initialize_population(instance)
        self.evaluate_population(pop, instance)
        for gen in range(self.n_generations):
            off = Population(size=self.population_size)
            idx_list = self.rng.permutation(len(pop))
            for i in range(0, len(pop), 2):
                c1, c2 = self._crossover(pop[idx_list[i]], pop[idx_list[(i+1)%len(pop)]])
                self._mutate(c1, instance); self._mutate(c2, instance)
                off.add(c1); off.add(c2)
            self.evaluate_population(off, instance)
            comb = Population(size=2*self.population_size)
            comb.individuals = pop.individuals + off.individuals
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
            pop = new_pop
            if gen % 10 == 0 or gen == self.n_generations - 1:
                # [FIX] Select individual with minimum makespan (which includes penalty)
                best_ind = min(pop.individuals, key=lambda x: x.makespan)
                best_m = best_ind.makespan
                avg_m = np.mean([ind.makespan for ind in pop.individuals])
                if verbose: print(f"  Gen {gen:3d}: Best={best_m:10.2f}, Avg={avg_m:10.2f}")
        self.pareto_front = [pop.individuals[i] for i in self._non_dominated_sort(pop)[0]]
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
    mats, wrks, offs = np.zeros(n_ops, dtype=int), np.zeros(n_ops, dtype=int), np.zeros(n_ops, dtype=int)
    for i, (jid, oidx) in enumerate(ops):
        op = instance.get_job(jid).operations[oidx]
        em, ew = list(op.eligible_machines), list(op.eligible_workers)
        mats[i] = rng.choice(em) if em else 0
        wrks[i] = rng.choice(ew) if ew else 0
    return {'sequence': seq, 'machines': mats, 'workers': wrks, 'offsets': offs, 'op_list': ops}


def evaluate_sfjssp_genome(instance: Any, genome: Dict[str, np.ndarray]) -> List[float]:
    """
    Robust evaluation function for SFJSSP genome.
    
    Args:
        instance: SFJSSPInstance
        genome: Dict containing 'sequence', 'machines', 'workers', 'offsets', 'op_list'
        
    Returns:
        List of 4 objectives: [Makespan, Energy, OCRA, Labor]
        Constraints are handled via heavy penalties.
    """
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
    offsets = genome['offsets']
    op_list = genome['op_list'] # List of (job_id, op_id)
    
    # 3. Create operation-to-resource mapping
    # This ensures that even if sequence order changes, the specific 
    # resource assigned to (job_id, op_id) stays with it.
    op_resource_map = {}
    for i, (jid, oid) in enumerate(op_list):
        op_resource_map[(jid, oid)] = {
            'm_id': int(machine_indices[i]),
            'w_id': int(worker_indices[i]),
            'offset': int(offsets[i])
        }
        
    schedule = Schedule(instance_id=instance.instance_id)
    job_op_ptr = {j.job_id: 0 for j in instance.jobs}
    machine_available = {m.machine_id: 0.0 for m in instance.machines}
    worker_available = {w.worker_id: 0.0 for w in instance.workers}
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
            offset = 0
        else:
            m_id, w_id, offset = res['m_id'], res['w_id'], res['offset']
            
        machine = instance.get_machine(m_id)
        worker = instance.get_worker(w_id)
        
        # Determine earliest possible start time
        est = max(
            machine_available[m_id] + machine.setup_time,
            worker_available[w_id],
            worker.mandatory_shift_lockout_until,
            job_last_completion[job_id] # [FIX] Already includes prev op's transport/waiting
        )
        
        # Apply shift-skipping offset
        if offset > 0:
            est = clock.period_start(clock.get_period(est) + offset)
            
        # Refine start time to satisfy hard constraints
        pt_base = op.processing_times.get(m_id, {0: 50.0}).get(0, 50.0)
        pt = pt_base / max(0.1, worker.get_efficiency())
        
        found = False
        curr_t = est
        for _ in range(50):
            # 1. Task cannot span two periods
            if clock.crosses_boundary(curr_t, curr_t + pt):
                curr_t = clock.period_start(clock.get_period(curr_t) + 1)
                continue
            # 2. No back-to-back 8h shifts
            if not worker.can_work_in_period(curr_t, curr_t + pt):
                curr_t = clock.period_start(clock.get_period(curr_t) + 1)
                continue
            # 3. Mandatory rest fraction (12.5%)
            m_rest = worker.requires_mandatory_rest(pt, curr_t)
            if m_rest > 0:
                curr_t += m_rest
                # [FIX] FORMALLY RECORD REST so fatigue recovers and lockout timer resets
                worker.record_rest(m_rest)
                continue
            found = True
            break
            
        if not found:
            # Individual is practically infeasible for this decoder
            return [1e9, 1e9, 1e9, 1e9]
            
        # 5. Record operation
        schedule.add_operation(
            job_id, op_idx, m_id, w_id, 0, # mode_id fixed to 0 for now
            curr_t, curr_t + pt, pt, 
            machine.setup_time, getattr(op, 'transport_time', 0.0)
        )
        
        # 6. Update states
        machine_available[m_id] = curr_t + pt
        worker_available[w_id] = curr_t + pt
        # [FIX] Add transport and waiting time to job availability for next op
        job_last_completion[job_id] = curr_t + pt + getattr(op, 'transport_time', 0.0) + getattr(op, 'waiting_time', 0.0)
        worker.record_work(pt, instance.get_ergonomic_risk(job_id, op_idx), curr_t)
        job_op_ptr[job_id] += 1
        
    # 7. Evaluate complete schedule
    metrics = schedule.evaluate(instance)
    is_feasible = schedule.check_feasibility(instance)
    
    # 8. Calculate penalties
    hard_violations = 0
    tardiness_penalty = 0.0
    n_tardy = 0
    
    for v in schedule.constraint_violations:
        if "Due date" in v:
            n_tardy += 1
            # Extract tardiness amount if possible
            try:
                # "Due date violation: Job X (C=100.00 > D=80.00)"
                parts = v.split("C=")[1].split(" > D=")
                c = float(parts[0])
                d = float(parts[1].rstrip(")"))
                tardiness_penalty += (c - d)
            except:
                tardiness_penalty += 1000.0
        else:
            hard_violations += 1
            
    # Ergonomic penalty if OCRA > threshold
    max_ocra = metrics.get('max_ergonomic_exposure', 0.0)
    ocra_threshold = getattr(instance, 'ocra_max_per_shift', 2.2)
    ocra_penalty = max(0.0, max_ocra - ocra_threshold) * 1e6
    
    total_penalty = (hard_violations * 1e7) + (n_tardy * 1e4) + (tardiness_penalty * 100.0) + ocra_penalty
    
    return [
        metrics.get('makespan', 1e6) + total_penalty,
        metrics.get('total_energy', 1e6) + total_penalty,
        max_ocra + (total_penalty / 1e6), # Keep OCRA in same scale for ranking but penalized
        metrics.get('total_labor_cost', 1e6) + total_penalty
    ]
