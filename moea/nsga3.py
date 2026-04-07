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
        def mix(a, b): return np.where(mask, a, b), np.where(mask, b, a)
        c1_m, c2_m = mix(g1['machines'], g2['machines'])
        c1_w, c2_w = mix(g1['workers'], g2['workers'])
        c1_o, c2_o = mix(g1['offsets'], g2['offsets'])

        return Individual(genome={'sequence': c1_seq, 'machines': c1_m, 'workers': c1_w, 'offsets': c1_o, 'op_list': g1['op_list']}), \
               Individual(genome={'sequence': c2_seq, 'machines': c2_m, 'workers': c2_w, 'offsets': c2_o, 'op_list': g1['op_list']})

    def _mutate(self, ind: Individual, instance: Any):
        g = ind.genome
        ops = g['op_list']
        if self.rng.random() < self.mutation_rate:
            i1, i2 = self.rng.choice(len(g['sequence']), 2, replace=False)
            g['sequence'][i1], g['sequence'][i2] = g['sequence'][i2], g['sequence'][i1]
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
    from sfjssp_model.schedule import Schedule
    for m in instance.machines: m.reset()
    for w in instance.workers: w.reset()
    
    seq = genome['sequence']
    mats = genome['machines']
    wrks = genome['workers']
    ops = genome.get('op_list', [])
    offs = genome.get('offsets', np.zeros(len(mats), dtype=int))
    
    if not ops: return [1e9, 1e9, 1e9, 1e9]
    sched = Schedule(instance_id=instance.instance_id)
    m_av, w_avail, j_last, j_cnt = {m.machine_id: 0.0 for m in instance.machines}, {w.worker_id: 0.0 for w in instance.workers}, {}, {j.job_id: 0 for j in instance.jobs}
    op_m = {p: i for i, p in enumerate(ops)}; clock = instance.period_clock
    for jid in seq:
        jid = int(jid); l_idx = j_cnt[jid]; j_cnt[jid] += 1
        job = instance.get_job(jid); op = job.operations[l_idx]; l_pos = op_m[(jid, l_idx)]
        mid, wid, ov = int(mats[l_pos]), int(wrks[l_pos]), int(offs[l_pos])
        if mid not in op.eligible_machines: mid = list(op.eligible_machines)[mid % len(op.eligible_machines)]
        if wid not in op.eligible_workers: wid = list(op.eligible_workers)[wid % len(op.eligible_workers)]
        m, w = instance.get_machine(mid), instance.get_worker(wid)
        start = max(m_av[mid] + m.setup_time, w_avail[wid], w.mandatory_shift_lockout_until)
        if l_idx > 0: start = max(start, j_last[(jid, l_idx - 1)] + getattr(op, 'transport_time', 0.0))
        if ov > 0: start = clock.period_start(clock.get_period(start) + ov)
        pt = op.processing_times.get(mid, {0: 50.0}).get(0, 50.0) / max(0.1, w.get_efficiency())
        for _ in range(50):
            if pt > clock.period_duration: pt = clock.period_duration * 0.95
            if clock.crosses_boundary(start, start+pt) or not w.can_work_in_period(start, start+pt):
                start = clock.period_start(clock.get_period(start) + 1)
                continue
            mr = w.requires_mandatory_rest(pt, start)
            if mr > 0: start += mr; w.record_rest(mr); continue
            break
        sched.add_operation(jid, l_idx, mid, wid, 0, start, start+pt, pt, m.setup_time, getattr(op, 'transport_time', 0.0))
        w.record_work(pt, instance.get_ergonomic_risk(jid, l_idx), start); m.total_processing_time += pt
        m_av[mid] = w_avail[wid] = j_last[(jid, l_idx)] = start + pt
    sched.compute_makespan(); sched.compute_total_energy(instance); sched.compute_ergonomic_metrics(instance)
    max_o = sched.ergonomic_metrics.get('max_exposure', 0.0); sched.check_feasibility(instance)
    hv, tp, nt = 0, 0.0, 0
    
    # [DEBUG] Print violations for one individual per 1000 evaluations to see what's failing
    if sched.constraint_violations and np.random.random() < 0.001:
        print(f"DEBUG: Sample Violations: {sched.constraint_violations[:3]}")

    for v in sched.constraint_violations:
        if "Due date" in v:
            nt += 1
            try:
                p = v.split("C=")[1].split(" > D=")
                tp += (float(p[0]) - float(p[1].rstrip(")")))
            except: tp += 1000.0
        else: hv += 1
    labor_cost = sum(instance.get_worker(w_id).labor_cost_per_hour * sum(o.processing_time for o in ws.operations) / 60.0 for w_id, ws in sched.worker_schedules.items())
    pen = (hv * 1e6) + (tp * 10.0) + (nt * 1000.0)
    if max_o > 2.2: pen += 1e6 * (max_o - 2.2)
    return [sched.makespan + pen, sched.energy_breakdown.get('total', 0.0) + pen, max_o + pen / 1e6, labor_cost + pen]
