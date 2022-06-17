from autogluon.core.utils.try_import import try_import_ray

class RayManager:
    """
    A singleton maintaining status of ray cluster resources
    """
    
    ray = None
    current_config = None
    
    @staticmethod
    def init_ray(**config):
        """Init ray only if the configuration changes"""
        if RayManager.ray is None:
            RayManager.ray = try_import_ray()
        if RayManager.current_config != config:
            if RayManager.ray.is_initialized():
                RayManager.ray.shutdown()
            RayManager.ray.init(**config)
            RayManager.current_config = config
