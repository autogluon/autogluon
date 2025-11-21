"""
This module contains integration tests that verify the _calculate_gpu_assignment() method specifically for
ParallelFoldFittingStrategy works correctly by submitting actual Ray remote tasks and verifying that tasks see the
correct GPU assignments via CUDA_VISIBLE_DEVICES and torch.cuda.

The tests simulate what happens in _ray_fit() at fold_fitting_strategy.py:428 where
GPU assignments are used to set CUDA_VISIBLE_DEVICES for each task.

Test Scenarios Covered:
- No GPUs available (CPU-only execution)
- Single GPU with round-robin assignment
- Multiple GPUs with consecutive assignment
- Multiple GPUs with round-robin assignment
- Concurrent execution of multiple tasks with one-to-one GPU assignment
"""

import pytest
import time
import numpy as np
import pandas as pd
from autogluon.core.models.ensemble.fold_fitting_strategy import ParallelLocalFoldFittingStrategy
from autogluon.core.models.ensemble.bagged_ensemble_model import BaggedEnsembleModel
from autogluon.tabular.models import AbstractModel


class DummyBaseModel(AbstractModel):
    """Dummy model for testing fold fitting strategy without actual training."""
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
    """Prepare test data for fold fitting strategy initialization."""
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
):
    """
    Construct a ParallelLocalFoldFittingStrategy instance for testing.

    Parameters
    ----------
    num_cpus : int
        Total CPUs available
    num_gpus : int or float
        Total GPUs available
    num_jobs : int
        Total number of fold jobs
    num_folds_parallel : int, optional
        Number of folds to train in parallel. Defaults to num_jobs if not specified.
    model_base_minimum_resources : dict, optional
        Minimum resources required by the model
    model_base_default_resources : dict, optional
        Default resources for the model

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
        X_pseudo=None,
        y_pseudo=None,
        sample_weight=None,
        time_start=time.time(),
        time_limit=None,
        models=[],
        oof_pred_proba=np.array([]),
        oof_pred_model_repeats=np.array([]),
        save_folds=True,
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
    try:
        import ray
    except ImportError:
        return None

    @ray.remote
    def ray_task_report_gpu_info(task_id, assigned_gpu_ids):
        """
        Ray remote task that reports GPU information.

        This simulates what happens in _ray_fit() at fold_fitting_strategy.py:428
        where GPU assignments are used to set CUDA_VISIBLE_DEVICES.

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
        try:
            import torch
        except ImportError:
            torch = None

        # Simulate the assignment logic in _ray_fit()
        if assigned_gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, assigned_gpu_ids))

        # Report what the task actually sees
        report = {
            'task_id': task_id,
            'assigned_gpu_ids': assigned_gpu_ids,
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', 'not set'),
        }

        if torch is not None:
            report['torch_gpu_count'] = torch.cuda.device_count()
            if torch.cuda.is_available():
                report['torch_current_device'] = torch.cuda.current_device()
            else:
                report['torch_current_device'] = None
        else:
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

    def test_ray_gpu_assignment_no_gpu_integration(self):
        """
        Integration Test: Verify GPU assignment with no GPUs
        This test should run on continuous_integration.yaml (single GPU CI)

        Scenario: 0 GPUs, 1 task
        Expected: Task runs on CPU, CUDA_VISIBLE_DEVICES = 'not set'
        """
        try:
            import ray
        except ImportError:
            pytest.skip("Ray not available")

        if not ray.is_initialized():
            ray.init(num_cpus=4, num_gpus=0, ignore_reinit_error=True)

        strategy = _construct_parallel_fold_strategy(num_cpus=4, num_gpus=0, num_jobs=1)
        gpu_assignments = {}

        gpu_assignments = strategy._calculate_gpu_assignment(
            gpu_assignments=gpu_assignments,
            task_id=0,
            gpus_per_task=0,
            total_gpus=0
        )

        assert gpu_assignments[0] == []

        ray.shutdown()

    def test_ray_gpu_assignment_single_gpu_task_execution(self):
        """
        Integration Test: Verify Ray tasks see correct GPU assignment with single GPU
        This test should run on continuous_integration.yaml (single GPU CI)

        Scenario: 1 GPU available, 2 tasks
        num_gpus_per_task = 1 / 2 = 0.5 (fractional, round-robin)
        Expected: Both tasks assigned to GPU 0, both see GPU 0 in CUDA_VISIBLE_DEVICES
        """
        try:
            import ray
        except ImportError:
            pytest.skip("Ray not available")

        system_gpus = self._get_system_gpu_count()
        if system_gpus == 0:
            pytest.skip("No GPUs available on system")

        if not ray.is_initialized():
            ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)

        ray_task_report_gpu_info = _create_ray_gpu_reporter()
        if ray_task_report_gpu_info is None:
            pytest.skip("Could not create Ray remote function")

        strategy = _construct_parallel_fold_strategy(num_cpus=4, num_gpus=1, num_jobs=2)
        gpu_assignments = {}

        # Calculate assignments for 2 tasks with 1 GPU (round-robin)
        for task_id in range(2):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0.5,
                total_gpus=1
            )

        # Both tasks should be assigned to GPU 0
        assert gpu_assignments[0] == [0]
        assert gpu_assignments[1] == [0]

        # Submit tasks and collect reports
        task_refs = []
        for task_id in range(2):
            ref = ray_task_report_gpu_info.remote(task_id, gpu_assignments[task_id])
            task_refs.append(ref)

        # Collect results
        results = ray.get(task_refs)

        # Verify each task sees the correct GPUs
        for result in results:
            task_id = result['task_id']
            assert result['assigned_gpu_ids'] == [0]
            # Task should see GPU 0 in CUDA_VISIBLE_DEVICES
            assert result['CUDA_VISIBLE_DEVICES'] == '0'

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_multiple_gpus_consecutive_task_execution(self):
        """
        Integration Test: Ray tasks verify consecutive GPU assignment
        This test should run on continuous_integration_multigpu.yaml

        Scenario: 4 GPUs, 2 tasks
        num_gpus_per_task = 4 / 2 = 2 (whole number, consecutive assignment)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0,1'
            - Task 1 sees CUDA_VISIBLE_DEVICES='2,3'
        """
        try:
            import ray
        except ImportError:
            pytest.skip("Ray not available")

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 4:
            pytest.skip(f"Need at least 4 GPUs, only {system_gpus} available")

        if not ray.is_initialized():
            ray.init(num_cpus=8, num_gpus=4, ignore_reinit_error=True)

        ray_task_report_gpu_info = _create_ray_gpu_reporter()
        if ray_task_report_gpu_info is None:
            pytest.skip("Could not create Ray remote function")

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=4, num_jobs=2)
        gpu_assignments = {}

        # Calculate assignments for 2 tasks with 4 GPUs (2 per task)
        for task_id in range(2):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=2,
                total_gpus=4
            )

        assert gpu_assignments[0] == [0, 1]
        assert gpu_assignments[1] == [2, 3]

        # Submit tasks and collect reports
        task_refs = []
        for task_id in range(2):
            ref = ray_task_report_gpu_info.remote(task_id, gpu_assignments[task_id])
            task_refs.append(ref)

        # Collect results
        results = ray.get(task_refs)

        # Verify each task sees the correct consecutive GPUs
        expected_cuda_visible = ['0,1', '2,3']
        for result in results:
            task_id = result['task_id']
            expected = expected_cuda_visible[task_id]
            assert result['CUDA_VISIBLE_DEVICES'] == expected
            # Verify torch sees correct number of GPUs
            if result['torch_gpu_count'] != 'torch not available':
                assert result['torch_gpu_count'] == 2, \
                    f"Task {task_id} expected 2 GPUs, saw {result['torch_gpu_count']}"

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_multiple_gpus_round_robin_task_execution(self):
        """
        Integration Test: Ray tasks verify round-robin GPU assignment
        This test should run on continuous_integration_multigpu.yaml

        Scenario: 2 GPUs, 4 tasks
        num_gpus_per_task = 2 / 4 = 0.5 (fractional, round-robin)
        Expected:
            - Task 0 sees CUDA_VISIBLE_DEVICES='0'
            - Task 1 sees CUDA_VISIBLE_DEVICES='1'
            - Task 2 sees CUDA_VISIBLE_DEVICES='0'
            - Task 3 sees CUDA_VISIBLE_DEVICES='1'
        """
        try:
            import ray
        except ImportError:
            pytest.skip("Ray not available")

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 2:
            pytest.skip(f"Need at least 2 GPUs, only {system_gpus} available")

        if not ray.is_initialized():
            ray.init(num_cpus=8, num_gpus=2, ignore_reinit_error=True)

        ray_task_report_gpu_info = _create_ray_gpu_reporter()
        if ray_task_report_gpu_info is None:
            pytest.skip("Could not create Ray remote function")

        strategy = _construct_parallel_fold_strategy(num_cpus=8, num_gpus=2, num_jobs=4)
        gpu_assignments = {}

        # Calculate assignments for 4 tasks with 2 GPUs (round-robin)
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=0.5,
                total_gpus=2
            )

        expected = [0, 1, 0, 1]
        for task_id, expected_gpu in enumerate(expected):
            assert gpu_assignments[task_id] == [expected_gpu]

        # Submit tasks and collect reports
        task_refs = []
        for task_id in range(4):
            ref = ray_task_report_gpu_info.remote(task_id, gpu_assignments[task_id])
            task_refs.append(ref)

        # Collect results
        results = ray.get(task_refs)

        # Verify each task sees the correct round-robin GPU assignment
        expected_cuda_visible = ['0', '1', '0', '1']
        for result in results:
            task_id = result['task_id']
            expected = expected_cuda_visible[task_id]
            assert result['CUDA_VISIBLE_DEVICES'] == expected
            # Each task should see only 1 GPU
            if result['torch_gpu_count'] != 'torch not available':
                assert result['torch_gpu_count'] == 1, \
                    f"Task {task_id} expected 1 GPU, saw {result['torch_gpu_count']}"

        ray.shutdown()

    @pytest.mark.multi_gpu
    def test_ray_gpu_assignment_concurrent_task_execution(self):
        """
        Integration Test: Multiple Ray tasks execute concurrently with correct GPU assignments
        This test should run on continuous_integration_multigpu.yaml

        Scenario: 4 GPUs, 4 tasks (one-to-one mapping)
        num_gpus_per_task = 4 / 4 = 1 (whole number)
        Expected: Each task sees its assigned GPU (0, 1, 2, or 3)

        This is the most realistic scenario from scratch/test.py with 4 GPUs, 4 folds
        """
        try:
            import ray
        except ImportError:
            pytest.skip("Ray not available")

        system_gpus = self._get_system_gpu_count()
        if system_gpus < 4:
            pytest.skip(f"Need at least 4 GPUs, only {system_gpus} available")

        if not ray.is_initialized():
            ray.init(num_cpus=16, num_gpus=4, ignore_reinit_error=True)

        ray_task_report_gpu_info = _create_ray_gpu_reporter()
        if ray_task_report_gpu_info is None:
            pytest.skip("Could not create Ray remote function")

        strategy = _construct_parallel_fold_strategy(num_cpus=16, num_gpus=4, num_jobs=4)
        gpu_assignments = {}

        # Calculate assignments for 4 tasks with 4 GPUs (one-to-one)
        for task_id in range(4):
            gpu_assignments = strategy._calculate_gpu_assignment(
                gpu_assignments=gpu_assignments,
                task_id=task_id,
                gpus_per_task=1,
                total_gpus=4
            )

        # Verify assignments are one-to-one
        for task_id in range(4):
            assert gpu_assignments[task_id] == [task_id]

        # Submit all tasks concurrently
        task_refs = []
        for task_id in range(4):
            ref = ray_task_report_gpu_info.remote(task_id, gpu_assignments[task_id])
            task_refs.append(ref)

        # Collect all results
        results = ray.get(task_refs)

        # Verify each task sees its assigned GPU
        for result in results:
            task_id = result['task_id']
            expected_gpu = task_id
            assert result['CUDA_VISIBLE_DEVICES'] == str(expected_gpu)
            # Each task should see exactly 1 GPU
            if result['torch_gpu_count'] != 'torch not available':
                assert result['torch_gpu_count'] == 1, \
                    f"Task {task_id} expected 1 GPU, saw {result['torch_gpu_count']}"

        ray.shutdown()
