from ray.tune.schedulers import AsyncHyperBandScheduler, FIFOScheduler


class SchedulerFactory:
    scheduler_presets = {
        "FIFO": FIFOScheduler,
        "ASHA": AsyncHyperBandScheduler,
        # Not support PBT/PB2 for now: https://github.com/ray-project/ray_lightning/issues/145
        # 'PBT' : PopulationBasedTraining,
        # 'PB2': PB2,
    }

    # These needs to be provided explicitly as scheduler init args
    # These should be experiment specified required args
    scheduler_required_args = {
        "FIFO": [],
        "ASHA": [],
    }

    # These are the default values if user not specified
    # These should be non-experiment specific args, which we just pick a default value for the users
    # Can be overridden by the users
    scheduler_default_args = {
        "FIFO": {},
        "ASHA": {"max_t": 9999},
    }

    @staticmethod
    def get_scheduler(scheduler_name: str, user_init_args, **kwargs):
        assert scheduler_name in SchedulerFactory.scheduler_presets, (
            f"{scheduler_name} is not a valid option. Options are {SchedulerFactory.scheduler_presets.keys()}"
        )
        init_args = {arg: kwargs[arg] for arg in SchedulerFactory.scheduler_required_args[scheduler_name]}
        init_args.update(SchedulerFactory.scheduler_default_args[scheduler_name])
        init_args.update(user_init_args)

        return SchedulerFactory.scheduler_presets[scheduler_name](**init_args)
