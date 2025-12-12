"""
This module contains integration tests that verify the _calculate_gpu_assignment() method specifically for
ParallelFoldFittingStrategy works correctly by submitting actual Ray remote tasks and verifying that tasks see the
correct GPU assignments via CUDA_VISIBLE_DEVICES and torch.cuda.
"""

import pytest
import time
import numpy as np
import pandas as pd
from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models import AbstractModel



@pytest.fixture(scope="session", autouse=True)
def ray_session_teardown():
    """Ensure Ray is fully shut down at the start and end of test session."""
    import ray
    ray.shutdown()
    time.sleep(0.2)
    yield
    ray.shutdown()
    time.sleep(0.2)


@pytest.fixture(autouse=True)
def ray_safe_test():
    """Ensure every test starts and ends with Ray fully shut down."""
    import ray
    ray.shutdown()
    time.sleep(0.2)
    yield
    ray.shutdown()
    time.sleep(0.2)


class DummyBaseModel(AbstractModel):
    def __init__(self, minimum_resources=None, default_resources=None, **kwargs):
        self._minimum_resources = minimum_resources
        self._default_resources = default_resources
        super().__init__(**kwargs)

    def get_minimum_resources(self, **kwargs):
        return self._minimum_resources

    def _get_default_resources(self):
        num_cpus = self._default_resources.get("num_cpus")
        num_gpus = self._default_resources.get("num_gpus")
        return num_cpus, num_gpus


class DummyModel(DummyBaseModel):
    pass


class DummyBaggedModel(BaggedEnsembleModel):
    pass


def _prepare_data():
    data = [[1, 10], [2, 20], [3, 30]]
    df = pd.DataFrame(data, columns=["Number", "Age"])
    label = "Age"
    X = df.drop(columns=[label])
    y = df[label]
    return X, y


def _construct_parallel_fold_strategy(
    num_cpus=8,
    num_gpus=4,
    num_jobs=8,
    num_folds_parallel=None,
    model_base_minimum_resources=None,
    model_base_default_resources=None,
    X_pseudo=None,
    y_pseudo=None,
    sample_weight=None,
    time_start=None,
    time_limit=None,
    save_folds=True,
):
    """
    Construct a ParallelLocalFoldFittingStrategy instance for testing.

    Parameters
    ----------
    num_cpus : int, default=8
        Total CPUs available
    num_gpus : int or float, default=4
        Total GPUs available
    num_jobs : int, default=8
        Total number of fold jobs
    num_folds_parallel : int, optional
        Number of folds to train in parallel. Defaults to num_jobs if not specified.
    model_base_minimum_resources : dict, optional
        Minimum resources required by the model. Defaults to {"num_cpus": 1, "num_gpus": 0}
    model_base_default_resources : dict, optional
        Default resources for the model. Defaults to {"num_cpus": 4, "num_gpus": 0}
    X_pseudo : DataFrame, optional
        Pseudo-labeled data features. Defaults to None
    y_pseudo : Series, optional
        Pseudo-labeled data labels. Defaults to None
    sample_weight : array-like, optional
        Sample weights for training. Defaults to None
    time_start : float, optional
        Time when training started. Defaults to current time.time()
    time_limit : float, optional
        Time limit for training in seconds. Defaults to None
    save_folds : bool, default=True
        Whether to save the fold models to disk

    Returns
    -------
    ParallelLocalFoldFittingStrategy
        An instance of the fold fitting strategy for testing
    """
    if num_folds_parallel is None:
        num_folds_parallel = num_jobs
    if model_base_minimum_resources is None:
        model_base_minimum_resources = {"num_cpus": 1, "num_gpus": 0}
    if model_base_default_resources is None:
        model_base_default_resources = {"num_cpus": 4, "num_gpus": 0}
    if time_start is None:
        time_start = time.time()

    dummy_model_base = DummyModel(
        minimum_resources=model_base_minimum_resources,
        default_resources=model_base_default_resources,
        hyperparameters={}
    )
    dummy_bagged_ensemble_model = DummyBaggedModel(dummy_model_base, hyperparameters={})
    train_data, test_data = _prepare_data()

    args = dict(
        num_jobs=num_jobs,
        num_folds_parallel=num_folds_parallel,
        model_base=dummy_model_base,
        model_base_kwargs=dict(),
        bagged_ensemble_model=dummy_bagged_ensemble_model,
        X=train_data,
        y=test_data,
        X_pseudo=X_pseudo,
        y_pseudo=y_pseudo,
        sample_weight=sample_weight,
        time_start=time_start,
        time_limit=time_limit,
        models=[],
        oof_pred_proba=np.array([]),
        oof_pred_model_repeats=np.array([]),
        save_folds=save_folds,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
    )
    return ParallelLocalFoldFittingStrategy(**args)


def _create_ray_gpu_reporter():
    """
    Create a Ray remote function that reports GPU information.

    This function is created outside the test class so it can be serialized
    and sent to Ray remote workers.
    """
    import ray

    @ray.remote
    def ray_task_report_gpu_info(task_id, assigned_gpu_ids):
        """
        Ray remote task that reports GPU information.

        This simulates what happens in _ray_fit() in fold_fitting_strategy.py
        where GPU assignments are used to set CUDA_VISIBLE_DEVICES and torch reports GPU info.

        Parameters
        ----------
        task_id : int
            The task ID for identification
        assigned_gpu_ids : list
            The list of GPU IDs assigned to this task

        Returns
        -------
        dict
            Dictionary containing GPU info reported by the task
        """
        import os

        # Simulate the assignment logic in _ray_fit()
        if assigned_gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, assigned_gpu_ids))

        report = {
            'task_id': task_id,
            'assigned_gpu_ids': assigned_gpu_ids,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
        }

        # Try to get torch GPU info if available
        try:
            import torch
            report['torch_gpu_count'] = torch.cuda.device_count()
            if torch.cuda.is_available():
                report['torch_current_device'] = torch.cuda.current_device()
            else:
                report['torch_current_device'] = None
        except ImportError:
            report['torch_gpu_count'] = 'torch not available'
            report['torch_current_device'] = 'torch not available'

        return report

    return ray_task_report_gpu_info


class TestRayGpuAssignmentIntegration:
    """Integration tests that verify GPU assignments work correctly with Ray."""

    @staticmethod
    def _get_system_gpu_count():
        """Get actual GPU count on the system."""
        try:
            import torch
            return torch.cuda.device_count()
        except Exception:
            return 0

    def _check_ray_init(self, num_cpus, num_gpus):
        """Ensure Ray is initialized with the specified resources."""
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=num_cpus, num_gpus=num_gpus)

    def _calculate_assignments(self, strategy, num_tasks, gpus_per_task, total_gpus):
        """Calculate GPU assignments for multiple tasks."""
        gpu_assignments = {}
        for task_id in range(num_tasks):
            gpu_assignments[task_id] = strategy._calculate_gpu_assignment(
                task_id=task_id,
                gpus_per_task=gpus_per_task,
                total_gpus=total_gpus
            )
        return gpu_assignments

    def _submit_and_collect_tasks(self, ray_task_report_gpu_info, gpu_assignments):
        """Submit Ray tasks and collect their results."""
        import ray
        task_refs = []
        for task_id, assigned_gpus in gpu_assignments.items():
            ref = ray_task_report_gpu_info.remote(task_id, assigned_gpus)
            task_refs.append(ref)
        results = ray.get(task_refs)
        return results

    def _verify_assignment_structure(self, gpu_assignments, num_tasks, expected_gpus_per_task):
        """Verify the structure of GPU assignments."""
        assert len(gpu_assignments) == num_tasks, \
            f"Expected {num_tasks} task assignments, got {len(gpu_assignments)}"
        for task_id in range(num_tasks):
            assert len(gpu_assignments[task_id]) == expected_gpus_per_task, \
                f"Expected {expected_gpus_per_task} GPUs for task {task_id}, got {len(gpu_assignments[task_id])}"

    def _verify_cuda_visible_devices(self, results, expected_cuda_visible):
        """Verify CUDA_VISIBLE_DEVICES for each task result."""
        for result in results:
            task_id = result['task_id']
            expected = expected_cuda_visible[task_id]
            assert result['CUDA_VISIBLE_DEVICES'] == expected, \
                f"Task {task_id} expected CUDA_VISIBLE_DEVICES='{expected}', got '{result['CUDA_VISIBLE_DEVICES']}'"

    def _verify_torch_gpu_count(self, results, expected_gpu_counts):
        """Verify torch reports correct GPU count for each task."""
        for result in results:
            task_id = result['task_id']
            if result['torch_gpu_count'] != 'torch not available':
                expected_count = expected_gpu_counts.get(task_id, expected_gpu_counts.get('default'))
                assert result['torch_gpu_count'] == expected_count, \
                    f"Task {task_id} expected {expected_count} GPU(s), saw {result['torch_gpu_count']}"

    def test_ray_gpu_assignment_no_gpu_integration(self):
        """
        Integration Test: Verify GPU assignment with no GPUs

        Scenario: 0 GPUs, 1 task
        Expected: Task runs on CPU, CUDA_VISIBLE_DEVICES = 'not set'
        """
        import ray  # Ray is required for these tests

        self._check_ray_init(num_cpus=4, num_gpus=0)

        strategy = _construct_parallel_fold_strategy(num_cpus=4, num_gpus=0, num_jobs=1)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=1, gpus_per_task=0, total_gpus=0)

        assert gpu_assignments[0] == [], f"Expected empty GPU list for task 0, got {gpu_assignments[0]}"

        ray.shutdown()

    def test_ray_gpu_assignment_single_gpu_task_execution(self):
        """
        Integration Test: Verify Ray tasks see correct GPU assignment with single GPU

        Scenario: 1 GPU available, 2 tasks, each task requests 0.5 GPU
        num_gpus_per_task = 0.5 (fractional, round-robin)
        Expected: Both tasks assigned to GPU 0, both see GPU 0 in CUDA_VISIBLE_DEVICES
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus == 0:
            pytest.skip("No GPUs available on system")

        self._check_ray_init(num_cpus=4, num_gpus=1)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=4, num_gpus=1, num_jobs=2)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=2, gpus_per_task=0.5, total_gpus=1)

        self._verify_assignment_structure(gpu_assignments, num_tasks=2, expected_gpus_per_task=1)
        assert gpu_assignments[0] == [0], f"Expected [0] for task 0, got {gpu_assignments[0]}"
        assert gpu_assignments[1] == [0], f"Expected [0] for task 1, got {gpu_assignments[1]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0', 1: '0'})
        self._verify_torch_gpu_count(results, {'default': 1})

        ray.shutdown()

    def test_ray_gpu_assignment_insufficient_gpus_multiple_tasks(self):
        """
        Integration Test: Multiple tasks with insufficient GPUs (oversubscription)

        Scenario: 1 GPU available, 2 tasks, each task requests 1 GPU
        num_gpus_per_task = 1 (whole number)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0'
            - Task 1 sees CUDA_VISIBLE_DEVICES='0' (wraps around due to oversubscription)
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus == 0:
            pytest.skip("No GPUs available on system")

        self._check_ray_init(num_cpus=4, num_gpus=1)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=4, num_gpus=1, num_jobs=2)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=2, gpus_per_task=1, total_gpus=1)

        self._verify_assignment_structure(gpu_assignments, num_tasks=2, expected_gpus_per_task=1)
        assert gpu_assignments[0] == [0], f"Expected [0] for task 0, got {gpu_assignments[0]}"
        assert gpu_assignments[1] == [0], f"Expected [0] for task 1, got {gpu_assignments[1]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0', 1: '0'})
        self._verify_torch_gpu_count(results, {'default': 1})

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_multiple_gpus_consecutive_task_execution(self):
        """
        Integration Test: Ray tasks verify consecutive GPU assignment

        Scenario: 4 GPUs, 2 tasks, each task requests 2 GPUs
        num_gpus_per_task = 2 (whole number, consecutive assignment)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0,1'
            - Task 1 sees CUDA_VISIBLE_DEVICES='2,3'
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 4:
            pytest.skip(f"Need at least 4 GPUs, only {system_gpus} available")

        self._check_ray_init(num_cpus=8, num_gpus=4)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=4, num_jobs=2)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=2, gpus_per_task=2, total_gpus=4)

        self._verify_assignment_structure(gpu_assignments, num_tasks=2, expected_gpus_per_task=2)
        assert gpu_assignments[0] == [0, 1], f"Expected [0, 1] for task 0, got {gpu_assignments[0]}"
        assert gpu_assignments[1] == [2, 3], f"Expected [2, 3] for task 1, got {gpu_assignments[1]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0,1', 1: '2,3'})
        self._verify_torch_gpu_count(results, {0: 2, 1: 2})

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_multiple_gpus_round_robin_task_execution(self):
        """
        Integration Test: Ray tasks verify round-robin GPU assignment

        Scenario: 2 GPUs, 4 tasks, each task requests 0.5 GPUs
        num_gpus_per_task = 0.5 (fractional, round-robin)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0'
            - Task 1 sees CUDA_VISIBLE_DEVICES='1'
            - Task 2 sees CUDA_VISIBLE_DEVICES='0'
            - Task 3 sees CUDA_VISIBLE_DEVICES='1'
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 2:
            pytest.skip(f"Need at least 2 GPUs, only {system_gpus} available")

        self._check_ray_init(num_cpus=8, num_gpus=2)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=2, num_jobs=4)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=4, gpus_per_task=0.5, total_gpus=2)

        self._verify_assignment_structure(gpu_assignments, num_tasks=4, expected_gpus_per_task=1)
        expected_assignments = [0, 1, 0, 1]
        for task_id, expected_gpu in enumerate(expected_assignments):
            assert gpu_assignments[task_id] == [expected_gpu], \
                f"Expected [{expected_gpu}] for task {task_id}, got {gpu_assignments[task_id]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0', 1: '1', 2: '0', 3: '1'})
        self._verify_torch_gpu_count(results, {'default': 1})

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_concurrent_task_execution(self):
        """
        Integration Test: Multiple Ray tasks execute concurrently with correct GPU assignments

        Scenario: 4 GPUs, 4 tasks (one-to-one mapping), each task requests 1 GPU
        num_gpus_per_task = 1 (whole number)
        Expected: Each task sees its assigned GPU (0, 1, 2, or 3)
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 4:
            pytest.skip(f"Need at least 4 GPUs, only {system_gpus} available")

        self._check_ray_init(num_cpus=16, num_gpus=4)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=16, num_gpus=4, num_jobs=4)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=4, gpus_per_task=1, total_gpus=4)

        self._verify_assignment_structure(gpu_assignments, num_tasks=4, expected_gpus_per_task=1)
        for task_id in range(4):
            assert gpu_assignments[task_id] == [task_id], \
                f"Expected [{task_id}] for task {task_id}, got {gpu_assignments[task_id]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        expected_cuda = {i: str(i) for i in range(4)}
        self._verify_cuda_visible_devices(results, expected_cuda)
        self._verify_torch_gpu_count(results, {'default': 1})

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_edge_case_fractional_round_robin(self):
        """
        Edge Case Test: Fractional GPU allocation with round-robin

        Scenario: 2 GPUs, 3 tasks, each task requests 2/3 GPUs
        num_gpus_per_task = 0.666... (fractional, round-robin)
        Expected: Tasks assigned in round-robin: Task 0->GPU 0, Task 1->GPU 1, Task 2->GPU 0
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus == 0:
            pytest.skip("No GPUs available on system")

        self._check_ray_init(num_cpus=8, num_gpus=2)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=2, num_jobs=3)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=3, gpus_per_task=2/3, total_gpus=2)

        self._verify_assignment_structure(gpu_assignments, num_tasks=3, expected_gpus_per_task=1)
        assert gpu_assignments[0] == [0], f"Task 0 should get GPU [0], got {gpu_assignments[0]}"
        assert gpu_assignments[1] == [1], f"Task 1 should get GPU [1], got {gpu_assignments[1]}"
        assert gpu_assignments[2] == [0], f"Task 2 should get GPU [0], got {gpu_assignments[2]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0', 1: '1', 2: '0'})
        self._verify_torch_gpu_count(results, {'default': 1})

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_edge_case_float_gpu_per_task_raises(self):
        """
        Edge Case Test: Float GPU allocation when >= 1 should raise AssertionError

        Scenario: 3 GPUs, 2 tasks, each task requests 1.5 GPUs
        num_gpus_per_task = 1.5 (float, >= 1)
        Expected: Should raise AssertionError because range() requires an integer
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 3:
            pytest.skip(f"Need at least 3 GPUs, only {system_gpus} available")

        self._check_ray_init(num_cpus=8, num_gpus=3)

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=3, num_jobs=2)

        # Should raise AssertionError because gpus_per_task must be int when >= 1
        with pytest.raises(AssertionError, match="When gpus_per_task >= 1, it must be an int"):
            self._calculate_assignments(strategy, num_tasks=2, gpus_per_task=1.5, total_gpus=3)

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_gpus_per_task_greater_than_total_task(self):
        """
        Integration Test: Ray tasks where each task requires all available GPUs

        Scenario: 4 GPUs available, 2 tasks, each task requires 4 GPUs
        num_gpus_per_task = 4 (whole number, requires oversubscription)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0,1,2,3'
            - Task 1 sees CUDA_VISIBLE_DEVICES='0,1,2,3'
        """
        import ray

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 4:
            pytest.skip(f"Need at least 4 GPUs, only {system_gpus} available")

        self._check_ray_init(num_cpus=8, num_gpus=4)
        ray_task_report_gpu_info = _create_ray_gpu_reporter()

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=4, num_jobs=2)
        gpu_assignments = self._calculate_assignments(strategy, num_tasks=2, gpus_per_task=4, total_gpus=4)

        self._verify_assignment_structure(gpu_assignments, num_tasks=2, expected_gpus_per_task=4)
        assert gpu_assignments[0] == [0, 1, 2, 3], f"Expected [0, 1, 2, 3] for task 0, got {gpu_assignments[0]}"
        assert gpu_assignments[1] == [0, 1, 2, 3], f"Expected [0, 1, 2, 3] for task 1, got {gpu_assignments[1]}"

        results = self._submit_and_collect_tasks(ray_task_report_gpu_info, gpu_assignments)
        self._verify_cuda_visible_devices(results, {0: '0,1,2,3', 1: '0,1,2,3'})
        self._verify_torch_gpu_count(results, {0: 4, 1: 4})

        ray.shutdown()
