# %%
import heapq
import random
import math
import numpy as np
from scipy.optimize import curve_fit
from collections import defaultdict
import matplotlib.pyplot as plt
from ortools.linear_solver import pywraplp
import csv
import warnings
import os

warnings.filterwarnings("ignore", category=np.RankWarning)

# Create necessary directories at the start
os.makedirs("./results/response_vs_overrun", exist_ok=True)
os.makedirs("./results/schedulability", exist_ok=True)
os.makedirs("./results/milp_analysis", exist_ok=True)


class Task:
    def __init__(self, id, period, wcet, net, utilization):
        self.id = id
        self.period = max(1, period)
        self.wcet = max(1, wcet)
        self.net = min(max(1, net), self.wcet)
        self.utilization = min(utilization, 1.0)
        self.missed_deadlines = 0
        self.response_time = 0
        self.total_jobs = 0
        self.core_assignment = -1
        self.total_execution = 0
        self.overrun = 0
        self.max_response = 0

    def __lt__(self, other):
        return self.period < other.period


class Job:
    def __init__(self, task, release_time):
        self.task = task
        self.release_time = release_time
        self.deadline = release_time + task.period
        self.remaining = task.wcet
        self.actual_execution = 0
        self.core = -1
        self.start_time = -1
        self.overrun = 0

    def __lt__(self, other):
        # Changed from deadline to release_time for FIFO
        return self.release_time < other.release_time


class Core:
    def __init__(self, id):
        self.id = id
        self.tasks = []
        self.current_job = None
        self.time = 0
        self.utilization = 0
        self.actual_utilization = 0
        self.schedule = []
        self.next_round_robin = 0  # For RR scheduling

    def add_task(self, task):
        if self.utilization + task.utilization <= 1.0:
            self.tasks.append(task)
            self.utilization += task.utilization
            task.core_assignment = self.id
            return True
        return False

    def reset(self):
        self.current_job = None
        self.time = 0
        self.actual_utilization = 0
        self.schedule = []
        self.next_round_robin = 0
        for task in self.tasks:
            task.missed_deadlines = 0
            task.response_time = 0
            task.total_jobs = 0
            task.total_execution = 0
            task.overrun = 0
            task.max_response = 0


class Scheduler:
    def __init__(self, num_cores, scheduling_policy, allocation_policy, is_preemptive):
        self.num_cores = num_cores
        self.cores = [Core(i) for i in range(num_cores)]
        self.scheduling_policy = scheduling_policy
        self.allocation_policy = allocation_policy
        self.is_preemptive = is_preemptive
        self.hyper_period = 800
        self.total_utilization = 0
        self.overrun_stats = defaultdict(list)

    def generate_tasks(self, num_tasks, target_utilization=None):
        tasks = []
        periods = [8, 10, 20, 25, 50, 100, 40, 80, 16, 32]

        if target_utilization is None:
            target_utilization = 0.9 * self.num_cores

        utilizations = []
        remaining_util = target_utilization
        for i in range(num_tasks):
            if i == num_tasks - 1:
                util = remaining_util
            else:
                util = remaining_util * (1 - random.random() ** (1 / (num_tasks - i)))
            util = max(0.05, min(0.95, util))
            utilizations.append(util)
            remaining_util -= util

        for i in range(num_tasks):
            period = random.choice(periods)
            wcet = max(1, int(period * utilizations[i]))
            net = random.randint(max(1, int(0.7 * wcet)), wcet)

            task = Task(i, period, wcet, net, utilizations[i])
            tasks.append(task)

        self.total_utilization = sum(t.utilization for t in tasks)
        return tasks

    def allocate_tasks(self, tasks):
        if "partitioned" in self.allocation_policy:
            for core in self.cores:
                core.tasks = []
                core.utilization = 0.0
            if "best_fit" in self.allocation_policy:
                self._allocate_best_fit(tasks)
            elif "worst_fit" in self.allocation_policy:
                self._allocate_worst_fit(tasks)
            elif "first_fit" in self.allocation_policy:
                self._allocate_first_fit(tasks)
            elif "round_robin" in self.allocation_policy:
                self._allocate_round_robin(tasks)
            else:
                # Default to best-fit decreasing
                self._allocate_best_fit_decreasing(tasks)
        else:  # global scheduling
            for core in self.cores:
                core.tasks = tasks.copy()
                core.utilization = self.total_utilization
                for task in tasks:
                    task.core_assignment = core.id

    def _allocate_best_fit_decreasing(self, tasks):
        """Best-fit decreasing heuristic for task allocation"""
        sorted_tasks = sorted(tasks, key=lambda x: -x.utilization)

        for task in sorted_tasks:
            best_core = None
            min_remaining = float("inf")

            for core in self.cores:
                remaining = 1.0 - (core.utilization + task.utilization)
                if remaining >= 0 and remaining < min_remaining:
                    best_core = core
                    min_remaining = remaining

            if best_core:
                best_core.add_task(task)
            else:
                # If no core has enough space, put it on the least utilized core
                min_core = min(self.cores, key=lambda x: x.utilization)
                min_core.add_task(task)

    def _allocate_best_fit(self, tasks):
        """Best-fit heuristic for task allocation"""
        for task in tasks:
            best_core = None
            min_remaining = float("inf")

            for core in self.cores:
                remaining = 1.0 - (core.utilization + task.utilization)
                if remaining >= 0 and remaining < min_remaining:
                    best_core = core
                    min_remaining = remaining

            if best_core:
                best_core.add_task(task)
            else:
                min_core = min(self.cores, key=lambda x: x.utilization)
                min_core.add_task(task)

    def _allocate_worst_fit(self, tasks):
        """Worst-fit heuristic for task allocation"""
        for task in tasks:
            worst_core = None
            max_remaining = -1

            for core in self.cores:
                remaining = 1.0 - (core.utilization + task.utilization)
                if remaining >= 0 and remaining > max_remaining:
                    worst_core = core
                    max_remaining = remaining

            if worst_core:
                worst_core.add_task(task)
            else:
                min_core = min(self.cores, key=lambda x: x.utilization)
                min_core.add_task(task)

    def _allocate_first_fit(self, tasks):
        """First-fit heuristic for task allocation"""
        for task in tasks:
            allocated = False
            for core in self.cores:
                if core.add_task(task):
                    allocated = True
                    break
            if not allocated:
                min_core = min(self.cores, key=lambda x: x.utilization)
                min_core.add_task(task)

    def _allocate_round_robin(self, tasks):
        """Round-robin task allocation"""
        core_idx = 0
        for task in tasks:
            allocated = False
            start_idx = core_idx
            while True:
                if self.cores[core_idx].add_task(task):
                    allocated = True
                    core_idx = (core_idx + 1) % self.num_cores
                    break
                core_idx = (core_idx + 1) % self.num_cores
                if core_idx == start_idx:
                    break
            if not allocated:
                min_core = min(self.cores, key=lambda x: x.utilization)
                min_core.add_task(task)

    def simulate(self, tasks, max_time=800):
        if "partitioned" in self.allocation_policy:
            self._simulate_partitioned(tasks, max_time)
        else:
            self._simulate_global(tasks, max_time)

    def _simulate_partitioned(self, tasks, max_time):
        for core in self.cores:
            if core.tasks:
                if self.is_preemptive:
                    self._simulate_core_pfifo(core, max_time)
                else:
                    self._simulate_core_npfifo(core, max_time)

    def _simulate_core_pfifo(self, core, max_time):
        jobs = []
        for task in core.tasks:
            releases = range(0, max_time, task.period)
            task.total_jobs = len(releases)
            for release in releases:
                jobs.append(Job(task, release))

        jobs.sort(key=lambda x: x.release_time)
        ready_queue = []
        current_job = None
        time = 0

        while time < max_time and (jobs or ready_queue or current_job):
            # Add newly released jobs
            while jobs and jobs[0].release_time <= time:
                job = jobs.pop(0)
                heapq.heappush(ready_queue, job)

            # Handle preemption if enabled
            if self.is_preemptive and current_job and ready_queue:
                if ready_queue[0].release_time < current_job.release_time:
                    heapq.heappush(ready_queue, current_job)
                    current_job = None

            # Get next job to execute
            if not current_job and ready_queue:
                current_job = heapq.heappop(ready_queue)
                current_job.start_time = time

            if current_job:
                execution = self._get_execution_time(current_job)

                # Determine how long to execute
                next_release = jobs[0].release_time if jobs else float("inf")
                execute_until = min(time + execution, next_release)
                execute_time = execute_until - time

                if execute_time > 0:
                    current_job.remaining -= execute_time
                    current_job.actual_execution += execute_time
                    core.actual_utilization += execute_time
                    core.schedule.append(
                        (
                            time,
                            execute_until,
                            current_job.task.id,
                            current_job.actual_execution,
                        )
                    )
                    time = execute_until

                # Check if job completed or missed deadline
                if current_job.remaining <= 0:
                    response = time - current_job.release_time
                    current_job.task.response_time = max(
                        current_job.task.response_time, response
                    )
                    current_job.task.max_response = max(
                        current_job.task.max_response, response
                    )
                    current_job.task.total_execution += current_job.actual_execution
                    current_job = None
                elif time >= current_job.deadline:
                    current_job.task.missed_deadlines += 1
                    current_job.task.response_time = max(
                        current_job.task.response_time,
                        current_job.deadline - current_job.release_time,
                    )
                    current_job.task.max_response = max(
                        current_job.task.max_response,
                        current_job.deadline - current_job.release_time,
                    )
                    current_job = None
            else:
                # No jobs to execute, advance to next event
                next_event = min(
                    jobs[0].release_time if jobs else float("inf"),
                    ready_queue[0].release_time if ready_queue else float("inf"),
                )
                time = min(next_event, max_time)

        # Calculate actual utilization
        if max_time > 0:
            core.actual_utilization /= max_time

    def _simulate_core_npfifo(self, core, max_time):
        jobs = []
        for task in core.tasks:
            releases = range(0, max_time, task.period)
            task.total_jobs = len(releases)
            for release in releases:
                jobs.append(Job(task, release))

        jobs.sort(key=lambda x: x.release_time)
        ready_queue = []
        current_job = None
        time = 0

        while time < max_time and (jobs or ready_queue or current_job):
            # Add newly released jobs
            while jobs and jobs[0].release_time <= time:
                job = jobs.pop(0)
                heapq.heappush(ready_queue, job)

            # Get next job to execute (non-preemptive)
            if not current_job and ready_queue:
                current_job = heapq.heappop(ready_queue)
                current_job.start_time = time

            if current_job:
                execution = self._get_execution_time(current_job)
                execute_time = min(execution, current_job.remaining)

                if execute_time > 0:
                    current_job.remaining -= execute_time
                    current_job.actual_execution += execute_time
                    core.actual_utilization += execute_time
                    core.schedule.append(
                        (
                            time,
                            time + execute_time,
                            current_job.task.id,
                            current_job.actual_execution,
                        )
                    )
                    time += execute_time

                # Check if job completed or missed deadline
                if current_job.remaining <= 0:
                    response = time - current_job.release_time
                    current_job.task.response_time = max(
                        current_job.task.response_time, response
                    )
                    current_job.task.max_response = max(
                        current_job.task.max_response, response
                    )
                    current_job.task.total_execution += current_job.actual_execution
                    current_job = None
                elif time >= current_job.deadline:
                    current_job.task.missed_deadlines += 1
                    current_job.task.response_time = max(
                        current_job.task.response_time,
                        current_job.deadline - current_job.release_time,
                    )
                    current_job.task.max_response = max(
                        current_job.task.max_response,
                        current_job.deadline - current_job.release_time,
                    )
                    current_job = None
            else:
                # No jobs to execute, advance to next event
                next_event = min(
                    jobs[0].release_time if jobs else float("inf"),
                    ready_queue[0].release_time if ready_queue else float("inf"),
                )
                time = min(next_event, max_time)

        # Calculate actual utilization
        if max_time > 0:
            core.actual_utilization /= max_time

    def _simulate_global(self, tasks, max_time):
        jobs = []
        for task in tasks:
            releases = range(0, max_time, task.period)
            task.total_jobs = len(releases)
            for release in releases:
                jobs.append(Job(task, release))

        jobs.sort(key=lambda x: x.release_time)
        ready_queue = []
        current_jobs = {core.id: None for core in self.cores}
        time = 0

        while time < max_time and (jobs or ready_queue or any(current_jobs.values())):
            # Add newly released jobs
            while jobs and jobs[0].release_time <= time:
                job = jobs.pop(0)
                heapq.heappush(ready_queue, job)

            # Handle preemption if enabled
            if self.is_preemptive:
                for core_id, current_job in current_jobs.items():
                    if current_job and ready_queue:
                        if ready_queue[0].release_time < current_job.release_time:
                            heapq.heappush(ready_queue, current_job)
                            current_jobs[core_id] = None

            # Assign jobs to idle cores
            available_cores = [cid for cid, job in current_jobs.items() if job is None]
            while available_cores and ready_queue:
                core_id = available_cores.pop(0)
                job = heapq.heappop(ready_queue)
                current_jobs[core_id] = job
                job.core = core_id
                job.start_time = time

            # Determine next event time
            next_events = []
            if jobs:
                next_events.append(jobs[0].release_time)

            for core_id, job in current_jobs.items():
                if job:
                    execution = self._get_execution_time(job)
                    execute_until = time + min(execution, job.remaining)
                    next_events.append(execute_until)

            if not next_events:
                break

            next_time = min(next_events)
            time_step = min(next_time, max_time) - time

            if time_step <= 0:
                time = min(next_time, max_time)
                continue

            # Execute jobs on all cores
            for core_id, job in current_jobs.items():
                if job:
                    execute_time = min(time_step, job.remaining)
                    if execute_time > 0:
                        job.remaining -= execute_time
                        job.actual_execution += execute_time
                        self.cores[core_id].actual_utilization += execute_time
                        self.cores[core_id].schedule.append(
                            (
                                time,
                                time + time_step,
                                job.task.id,
                                job.actual_execution,
                            )
                        )
                        if job.remaining <= 0:
                            # Job completed
                            response = (time + execute_time) - job.release_time
                            job.task.response_time = max(
                                job.task.response_time, response
                            )
                            job.task.max_response = max(job.task.max_response, response)
                            job.task.total_execution += job.actual_execution
                            current_jobs[core_id] = None
                        elif (time + execute_time) >= job.deadline:
                            # Deadline missed
                            job.task.missed_deadlines += 1
                            job.task.response_time = max(
                                job.task.response_time, job.deadline - job.release_time
                            )
                            job.task.max_response = max(
                                job.task.max_response, job.deadline - job.release_time
                            )
                            current_jobs[core_id] = None

            time += time_step

        # Calculate actual utilizations
        for core in self.cores:
            if max_time > 0:
                core.actual_utilization /= max_time

    def _get_execution_time(self, job):
        execution = job.task.net
        if random.random() < 0.1:  # 10% chance of overrun
            overrun = random.randint(0, job.task.wcet - job.task.net)
            execution += overrun
            job.overrun = overrun
            job.task.overrun += overrun
            self.overrun_stats[job.task.id].append(overrun)
        return execution

    def analyze_results(self, tasks, max_time, scenario_name):
        task_metrics = []
        critical_overruns = []
        nonlinear_points = []

        for task in tasks:
            # Get tasks on the same core for response time calculation
            core_tasks = [t for t in tasks if t.core_assignment == task.core_assignment]

            # Calculate theoretical response time based on scheduling policy
            if self.is_preemptive:
                theoretical_response = calculate_response_time_pfifo(task, core_tasks)
            else:
                theoretical_response = calculate_response_time_npfifo(task, core_tasks)

            expected_completions = max(1, task.total_jobs)
            actual_completions = expected_completions - task.missed_deadlines

            # Calculate miss rate (task is missed if it has more than one missed deadline job)
            miss_ratio = 1.0 if task.missed_deadlines > 1 else 0.0

            avg_response = 0
            if actual_completions > 0:
                avg_response = task.response_time / actual_completions

            task_metrics.append(
                {
                    "id": task.id,
                    "period": task.period,
                    "wcet": task.wcet,
                    "net": task.net,
                    "utilization": task.utilization,
                    "missed_deadlines": task.missed_deadlines,
                    "total_jobs": task.total_jobs,
                    "miss_ratio": miss_ratio,
                    "theoretical_response": theoretical_response,
                    "actual_response": task.response_time,
                    "max_response": task.max_response,
                    "avg_response": avg_response,
                    "response_ratio": (
                        avg_response / task.period if task.period > 0 else 0
                    ),
                    "core": task.core_assignment,
                    "total_execution": task.total_execution,
                    "overrun": task.overrun,
                    "overrun_ratio": (
                        task.overrun / (task.wcet - task.net)
                        if task.wcet > task.net
                        else 0
                    ),
                }
            )

            # Identify critical overruns and nonlinear points
            if task.overrun > 0:
                if task.max_response > 1.2 * task.period:
                    critical_overruns.append(
                        {
                            "task_id": task.id,
                            "period": task.period,
                            "wcet": task.wcet,
                            "net": task.net,
                            "overrun": task.overrun,
                            "max_response": task.max_response,
                            "miss_ratio": miss_ratio,
                            "core": task.core_assignment,
                            "utilization": task.utilization,
                        }
                    )

                # Detect nonlinear response increases
                if len(self.overrun_stats[task.id]) > 3:
                    overruns = np.array(self.overrun_stats[task.id])
                    responses = []
                    for o in overruns:
                        responses.append(
                            task.net + o + task.period * 0.5
                        )  # Simplified model

                    # Check for nonlinearity
                    if (
                        np.polyfit(overruns, responses, 2)[0] > 0.1
                    ):  # Quadratic coefficient
                        nonlinear_points.append(
                            {
                                "task_id": task.id,
                                "threshold": np.mean(overruns),
                                "slope": np.polyfit(overruns, responses, 1)[0],
                            }
                        )

        # Core statistics
        core_stats = []
        for core in self.cores:
            core_tasks = [t for t in task_metrics if t["core"] == core.id]
            core_miss_ratio = (
                np.mean([t["miss_ratio"] for t in core_tasks]) if core_tasks else 0
            )
            core_stats.append(
                {
                    "id": core.id,
                    "utilization": core.utilization,
                    "actual_utilization": core.actual_utilization,
                    "miss_ratio": core_miss_ratio,
                    "task_count": len(core.tasks),
                    "overrun_tasks": sum(1 for t in core_tasks if t["overrun"] > 0),
                }
            )

        # Overall statistics
        avg_miss_ratio = (
            np.mean([t["miss_ratio"] for t in task_metrics]) if task_metrics else 0
        )
        max_miss_ratio = (
            max([t["miss_ratio"] for t in task_metrics]) if task_metrics else 0
        )
        avg_response_ratio = (
            np.mean([t["response_ratio"] for t in task_metrics if t["period"] > 0])
            if task_metrics
            else 0
        )
        # Enhanced nonlinear analysis

        critical_paths = self._analyze_critical_paths(tasks)
        if critical_paths:
            self.plot_milp_analysis(critical_paths, tasks, scenario_name)
        return {
            "task_metrics": task_metrics,
            "core_stats": core_stats,
            "critical_paths": critical_paths,
            "critical_overruns": critical_overruns,
            "nonlinear_points": nonlinear_points,
            "avg_miss_ratio": avg_miss_ratio,
            "max_miss_ratio": max_miss_ratio,
            "avg_response_ratio": avg_response_ratio,
            "total_utilization": self.total_utilization,
            "overrun_stats": dict(self.overrun_stats),
        }

    def plot_response_vs_overrun(self, results, scenario_name):
        plt.figure(figsize=(12, 8))

        for task in results["task_metrics"]:
            if task["overrun"] > 0:
                overruns = results["overrun_stats"].get(task["id"], [])
                if overruns:
                    responses = []
                    for o in overruns:
                        responses.append(
                            task["net"] + o + task["period"] * 0.3
                        )  # Simplified response model

                    plt.scatter(
                        overruns, responses, label=f"Task {task['id']}", alpha=0.6
                    )

        plt.xlabel("Overrun Amount (e)")
        plt.ylabel("Response Time")
        plt.title(f"Response Time vs NET Overrun\nScenario: {scenario_name}")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            f"./results/response_vs_overrun/response_vs_overrun_{scenario_name}.png"
        )
        plt.close()

    def plot_schedulability(self, results, scenario_name):
        plt.figure(figsize=(12, 8))

        util_bins = np.linspace(0, 1, 11)
        miss_ratios = []

        for u in util_bins:
            tasks_in_bin = [
                t for t in results["task_metrics"] if (u - 0.1) <= t["utilization"] < u
            ]
            if tasks_in_bin:
                count_gt_0 = sum(1 for t in tasks_in_bin if t["miss_ratio"] > 0)
                miss_ratios.append(count_gt_0)
            else:
                miss_ratios.append(0)

        plt.bar(util_bins, miss_ratios, width=0.1)
        plt.xlabel("Task Utilization")
        plt.ylabel("Miss count")
        plt.title(f"Schedulability Analysis\nScenario: {scenario_name}")
        plt.grid(True)
        plt.savefig(f"./results/schedulability/schedulability_{scenario_name}.png")
        plt.close()

    def _setup_milp_solver(self):
        """Initialize MILP solver for critical path analysis"""
        self.milp_solver = pywraplp.Solver.CreateSolver("SCIP")
        self.milp_vars = {}

    def _add_milp_constraints(self, tasks):
        """Add constraints for critical path analysis"""
        # Create variables for each task
        for task in tasks:
            self.milp_vars[task.id] = self.milp_solver.IntVar(
                0, task.period, f"t{task.id}"
            )

        # Add constraints for dependencies and deadlines
        for task in tasks:
            # Constraint: Execution time <= Period
            self.milp_solver.Add(self.milp_vars[task.id] <= task.period - task.wcet)

    def _analyze_critical_paths(self, tasks):
        """Use MILP to find critical paths and worst-case scenarios"""
        try:
            self._setup_milp_solver()
            self._add_milp_constraints(tasks)

            # Objective: Maximize response time considering overruns
            objective = self.milp_solver.Objective()
            for task in tasks:
                # Weight by both utilization and potential overrun impact
                weight = task.utilization * (task.wcet - task.net) / task.wcet
                objective.SetCoefficient(self.milp_vars[task.id], weight)
            objective.SetMaximization()

            status = self.milp_solver.Solve()

            if status == pywraplp.Solver.OPTIMAL:
                critical_times = {}
                for task in tasks:
                    crit_time = self.milp_vars[task.id].solution_value()
                    # Calculate criticality score (0-1)
                    criticality = min(1.0, crit_time / (task.period - task.net))
                    critical_times[task.id] = {
                        "critical_time": crit_time,
                        "criticality": criticality,
                        "period": task.period,
                        "wcet": task.wcet,
                        "net": task.net,
                    }
                return critical_times
        except Exception as e:
            print(f"MILP analysis failed: {str(e)}")
        return None

    def plot_milp_analysis(self, critical_paths, tasks, scenario_name):
        """Visualize MILP critical path analysis results"""
        if not critical_paths:
            return

        # Prepare data
        task_ids = []
        criticalities = []
        periods = []
        wcets = []
        nets = []

        for task_id, data in critical_paths.items():
            task_ids.append(task_id)
            criticalities.append(data["criticality"])
            periods.append(data["period"])
            wcets.append(data["wcet"])
            nets.append(data["net"])

        # Create figure
        plt.figure(figsize=(15, 10))

        # Criticality heatmap - FIXED COLORBAR ISSUE
        plt.subplot(2, 2, 1)
        sorted_idx = np.argsort(criticalities)[::-1]
        sorted_crit = np.array(criticalities)[sorted_idx]
        bars = plt.bar(
            range(len(criticalities)), sorted_crit, color=plt.cm.viridis(sorted_crit)
        )
        plt.xticks(
            range(len(criticalities)), np.array(task_ids)[sorted_idx], rotation=45
        )
        plt.xlabel("Task ID")
        plt.ylabel("Criticality Score (0-1)")
        plt.title("Task Criticality Ranking")

        # Create scalar mappable for colorbar
        sm = plt.cm.ScalarMappable(
            cmap="viridis",
            norm=plt.Normalize(vmin=min(criticalities), vmax=max(criticalities)),
        )
        sm.set_array([])
        plt.colorbar(sm, ax=plt.gca(), label="Criticality")

        # Criticality vs Period
        plt.subplot(2, 2, 2)
        scatter = plt.scatter(periods, criticalities, c=wcets, cmap="plasma", s=100)
        plt.xlabel("Task Period")
        plt.ylabel("Criticality Score")
        plt.title("Criticality vs Period (Size=WCET)")
        plt.colorbar(scatter, label="WCET")

        # Criticality vs NET/WCET ratio
        plt.subplot(2, 2, 3)
        ratios = [n / w for n, w in zip(nets, wcets)]
        scatter = plt.scatter(ratios, criticalities, c=periods, cmap="cool", s=100)
        plt.xlabel("NET/WCET Ratio")
        plt.ylabel("Criticality Score")
        plt.title("Criticality vs NET/WCET (Color=Period)")
        plt.colorbar(scatter, label="Period")

        # Criticality distribution
        plt.subplot(2, 2, 4)
        plt.hist(criticalities, bins=20, edgecolor="black")
        plt.xlabel("Criticality Score")
        plt.ylabel("Number of Tasks")
        plt.title("Criticality Distribution")

        plt.tight_layout()
        plt.suptitle(f"MILP Critical Path Analysis\nScenario: {scenario_name}", y=1.02)
        plt.savefig(
            f"./results/milp_analysis/milp_analysis_{scenario_name}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

    def generate_milp_report(self, critical_paths):
        """Generate report of critical paths identified by MILP"""
        report = "Critical Path Analysis (MILP):\n"
        report += "Task ID | Critical Time | Period | WCET | NET\n"
        report += "--------|---------------|--------|------|----\n"

        for task_id, crit_time in critical_paths.items():
            task = next(t for t in self.tasks if t.id == task_id)
            report += (
                f"{task_id:7} | {crit_time:13.2f} | {task.period:6} | "
                f"{task.wcet:4} | {task.net:3}\n"
            )

        return report


def calculate_response_time_pfifo(task, core_tasks):
    """Calculate response time for preemptive FIFO with NET overrun"""
    # Sort tasks by release time (FIFO order)
    sorted_tasks = sorted(
        core_tasks, key=lambda x: x.period
    )  # Using period as proxy for release order

    # Find all tasks that come before our task in the FIFO queue
    preceding_tasks = [
        t for t in sorted_tasks if t.period <= task.period and t.id != task.id
    ]

    # Calculate response time using FIFO formula: R_i = C_i + sum(C_j + Δ_j) for all preceding tasks
    R_i = task.wcet
    for t in preceding_tasks:
        R_i += t.wcet + (t.wcet - t.net)  # C_j + Δ_j

    return min(R_i, task.period)


def calculate_response_time_npfifo(task, core_tasks):
    """Calculate response time for non-preemptive FIFO with NET overrun"""
    # Sort tasks by release time (FIFO order)
    sorted_tasks = sorted(
        core_tasks, key=lambda x: x.period
    )  # Using period as proxy for release order

    # Find all tasks that come before our task in the FIFO queue
    preceding_tasks = [
        t for t in sorted_tasks if t.period <= task.period and t.id != task.id
    ]

    # Calculate response time using FIFO formula: R_i = sum(C_j + Δ_j) for all preceding tasks + C_i
    R_i = task.wcet
    for t in preceding_tasks:
        R_i += t.wcet + (t.wcet - t.net)  # C_j + Δ_j

    return min(R_i, task.period)


def run_comprehensive_scenarios():
    scenarios = []

    # Vary number of tasks
    for n in [10, 50, 100, 200, 400, 500]:
        for alloc_policy in [
            "partitioned_best_fit",
            "partitioned_worst_fit",
            "partitioned_first_fit",
            "partitioned_round_robin",
            "global",
        ]:
            scenarios.append(
                {
                    "name": f"{alloc_policy}_FIFO_{n}tasks_4cores",
                    "num_cores": 4,
                    "num_tasks": n,
                    "scheduling_policy": "FIFO",
                    "allocation_policy": alloc_policy,
                    "is_preemptive": True,
                    "max_time": 800,
                    "target_util": 0.8 * 4,  # 80% system utilization
                }
            )

    # Vary number of cores
    for c in [4, 8, 16, 32]:
        for alloc_policy in [
            "partitioned_best_fit",
            "partitioned_worst_fit",
            "partitioned_first_fit",
            "partitioned_round_robin",
        ]:
            scenarios.append(
                {
                    "name": f"{alloc_policy}_FIFO_100tasks_{c}cores",
                    "num_cores": c,
                    "num_tasks": 100,
                    "scheduling_policy": "FIFO",
                    "allocation_policy": alloc_policy,
                    "is_preemptive": True,
                    "max_time": 800,
                    "target_util": 0.8 * c,
                }
            )

        # Vary number of cores
    for c in [4, 8, 16, 32]:
        for alloc_policy in ["global"]:
            scenarios.append(
                {
                    "name": f"{alloc_policy}_FIFO_100tasks_{c}cores",
                    "num_cores": c,
                    "num_tasks": 100,
                    "scheduling_policy": "FIFO",
                    "allocation_policy": alloc_policy,
                    "is_preemptive": True,
                    "max_time": 800,
                    "target_util": 0.5 * c,
                }
            )

    # Different scheduling policies
    for policy in ["partitioned_best_fit", "global"]:
        scenarios.append(
            {
                "name": f"{policy}_FIFO_50tasks_8cores",
                "num_cores": 8,
                "num_tasks": 50,
                "scheduling_policy": "FIFO",
                "allocation_policy": policy,
                "is_preemptive": True,
                "max_time": 800,
                "target_util": 0.8 * 8,
            }
        )

        # Different scheduling policies
    for policy in ["partitioned_best_fit", "global"]:
        scenarios.append(
            {
                "name": f"{policy}_non_preemptive_FIFO_50tasks_8cores",
                "num_cores": 4,
                "num_tasks": 100,
                "scheduling_policy": "FIFO",
                "allocation_policy": policy,
                "is_preemptive": False,
                "max_time": 800,
                "target_util": 0.8 * 4,
            }
        )

    # Different utilization levels
    for util in [0.25, 0.5, 0.75]:
        for alloc_policy in ["partitioned_best_fit", "partitioned_worst_fit"]:
            scenarios.append(
                {
                    "name": f"{alloc_policy}_FIFO_100tasks_4cores_util{util}",
                    "num_cores": 4,
                    "num_tasks": 100,
                    "scheduling_policy": "FIFO",
                    "allocation_policy": alloc_policy,
                    "is_preemptive": True,
                    "max_time": 800,
                    "target_util": util * 4,
                }
            )

    all_results = {}

    for scenario in scenarios:
        print(f"Running scenario: {scenario['name']}")

        scheduler = Scheduler(
            num_cores=scenario["num_cores"],
            scheduling_policy=scenario["scheduling_policy"],
            allocation_policy=scenario["allocation_policy"],
            is_preemptive=scenario["is_preemptive"],
        )

        tasks = scheduler.generate_tasks(
            scenario["num_tasks"], scenario.get("target_util")
        )
        scheduler.allocate_tasks(tasks)
        scheduler.simulate(tasks, scenario["max_time"])
        results = scheduler.analyze_results(
            tasks, scenario["max_time"], scenario["name"]
        )
        all_results[scenario["name"]] = results

        # Generate specific plots
        scheduler.plot_response_vs_overrun(results, scenario["name"])
        scheduler.plot_schedulability(results, scenario["name"])

    generate_detailed_report(all_results)


def generate_detailed_report(all_results):
    with open("./results/comprehensive_fifo_report.txt", "w") as f:
        f.write("Comprehensive FIFO Scheduling with NET Overruns Report\n")
        f.write("===================================================\n\n")

        # 1. Upper bounds for response time with NET overruns
        f.write("1. Upper Bounds for Response Time with NET Overruns:\n")
        for scenario, results in all_results.items():
            f.write(f"{scenario}:\n")
            f.write(f"- Avg Response/Period: {results['avg_response_ratio']:.2f}\n")
            f.write(f"- Miss Rate: {results['avg_miss_ratio']:.2%}\n")
            f.write(
                f"- Max Response/Period: {max(t['response_ratio'] for t in results['task_metrics']):.2f}\n"
            )
            f.write(
                f"- Worst-case Response: {max(t['max_response'] for t in results['task_metrics'] if t['period'] > 0)} units\n\n"
            )

        # 2. Critical overrun values (e)
        f.write("\n2. Critical Overrun Values Causing Nonlinear Response:\n")
        for scenario, results in all_results.items():
            if results["critical_overruns"]:
                f.write(f"{scenario}:\n")
                for task in sorted(
                    results["critical_overruns"], key=lambda x: -x["overrun"]
                )[:5]:
                    f.write(f"Task {task['task_id']}: Overrun={task['overrun']} ")
                    f.write(f"(NET={task['net']}, WCET={task['wcet']}) ")
                    f.write(
                        f"Response={task['max_response']} (Period={task['period']})\n"
                    )
                f.write("\n")

        # 3. System sensitive points
        f.write("\n3. System Sensitive Points (Nonlinear Response):\n")
        for scenario, results in all_results.items():
            if results["nonlinear_points"]:
                f.write(f"{scenario}:\n")
                for point in results["nonlinear_points"]:
                    f.write(
                        f"Task {point['task_id']}: Threshold={point['threshold']:.1f} "
                    )
                    f.write(f"Slope={point['slope']:.2f}\n")
                f.write("\n")

        # 4. Results for different task counts
        f.write("\n4. Results for Different Task Counts:\n")
        f.write(
            "Alloc Policy | Tasks | Avg Miss Ratio | Max Miss Ratio | Avg Response/Period\n"
        )
        f.write(
            "-------------|-------|----------------|----------------|--------------------\n"
        )
        for scenario, results in all_results.items():
            if "tasks" in scenario:
                alloc_policy = scenario.split("_")[0] + "_" + scenario.split("_")[1]
                num_tasks = scenario.split("_")[3]
                f.write(
                    f"{alloc_policy:13} | {num_tasks:5} | {results['avg_miss_ratio']:.2%} | "
                )
                f.write(
                    f"{results['max_miss_ratio']:.2%} | {results['avg_response_ratio']:.2f}\n"
                )

        # 5. Results for different core counts
        f.write("\n5. Results for Different Core Counts:\n")
        f.write(
            "Alloc Policy | Cores | Avg Miss Ratio | Max Miss Ratio | Avg Response/Period\n"
        )
        f.write(
            "-------------|-------|----------------|----------------|--------------------\n"
        )
        for scenario, results in all_results.items():
            if "cores" in scenario and "tasks" in scenario:
                parts = scenario.split("_")
                alloc_policy = parts[0] + "_" + parts[1]
                cores = parts[-2]
                f.write(
                    f"{alloc_policy:13} | {cores:5} | {results['avg_miss_ratio']:.2%} | "
                )
                f.write(
                    f"{results['max_miss_ratio']:.2%} | {results['avg_response_ratio']:.2f}\n"
                )

        # 6. Results for different utilization levels
        f.write("\n6. Results for Different Utilization Levels:\n")
        f.write(
            "Alloc Policy | Util | Avg Miss Ratio | Max Miss Ratio | Avg Response/Period\n"
        )
        f.write(
            "-------------|------|----------------|----------------|--------------------\n"
        )
        for scenario, results in all_results.items():
            if "util" in scenario:
                parts = scenario.split("_")
                alloc_policy = parts[0] + "_" + parts[1]
                util = parts[-1].replace("util", "")
                f.write(
                    f"{alloc_policy:13} | {util:4} | {results['avg_miss_ratio']:.2%} | "
                )
                f.write(
                    f"{results['max_miss_ratio']:.2%} | {results['avg_response_ratio']:.2f}\n"
                )

        # 7. Comparison of allocation policies
        f.write("\n7. Comparison of Allocation Policies:\n")
        f.write(
            "Scenario | Alloc Policy | Avg Miss Ratio | Max Miss Ratio | Avg Response/Period\n"
        )
        f.write(
            "---------|--------------|----------------|----------------|--------------------\n"
        )
        for scenario, results in all_results.items():
            parts = scenario.split("_")
            if len(parts) >= 2:
                alloc_policy = parts[0] + "_" + parts[1]
                f.write(
                    f"{scenario:8} | {alloc_policy:12} | {results['avg_miss_ratio']:.2%} | "
                )
                f.write(
                    f"{results['max_miss_ratio']:.2%} | {results['avg_response_ratio']:.2f}\n"
                )
        # 8. MILP Critical Path Analysis
        f.write("\n8. MILP Critical Path Analysis:\n")
        for scenario, results in all_results.items():
            if results.get("critical_paths"):  # Use .get() for safer access
                f.write(f"\n{scenario}:\n")
                f.write("Task ID | Critical Time | Criticality | Period | WCET | NET\n")
                f.write("--------|---------------|-------------|--------|------|----\n")
                for task_id, crit_data in results["critical_paths"].items():
                    task = next(
                        (t for t in results["task_metrics"] if t["id"] == task_id), None
                    )
                    if task:  # Only proceed if we found the task
                        f.write(
                            f"{task_id:7} | {crit_data['critical_time']:13.2f} | "
                            f"{crit_data['criticality']:11.2f} | "
                            f"{task['period']:6} | {task['wcet']:4} | {task['net']:3}\n"
                        )

    # Save detailed data to CSV for further analysis
    with open("./results/detailed_results.csv", "w", newline="") as csvfile:
        fieldnames = [
            "scenario",
            "task_id",
            "period",
            "wcet",
            "net",
            "utilization",
            "miss_ratio",
            "max_response",
            "overrun",
            "core",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for scenario, results in all_results.items():
            for task in results["task_metrics"]:
                writer.writerow(
                    {
                        "scenario": scenario,
                        "task_id": task["id"],
                        "period": task["period"],
                        "wcet": task["wcet"],
                        "net": task["net"],
                        "utilization": task["utilization"],
                        "miss_ratio": task["miss_ratio"],
                        "max_response": task["max_response"],
                        "overrun": task["overrun"],
                        "core": task["core"],
                    }
                )


if __name__ == "__main__":
    run_comprehensive_scenarios()
