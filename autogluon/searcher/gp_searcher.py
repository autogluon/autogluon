import ConfigSpace as CS

from ampercore.autogluon.gp_multifidelity_searcher import \
    GPMultiFidelitySearcher as _GPMultiFidelitySearcher
from ampercore.autogluon.gp_fifo_searcher import \
    GPFIFOSearcher as _GPFIFOSearcher
from .searcher import BaseSearcher

__all__ = ['GPMultiFidelitySearcher', 'GPFIFOSearcher']


class GPMultiFidelitySearcher(BaseSearcher):
    def __init__(self, gp_searcher: _GPMultiFidelitySearcher):
        super(GPMultiFidelitySearcher, self).__init__(
            gp_searcher.hp_ranges.config_space)
        self.gp_searcher = gp_searcher

    def get_config(self, **kwargs):
        with self.LOCK:
            config_cs = self.gp_searcher.get_config(**kwargs)
            return config_cs.get_dictionary()

    def update(self, config, reward, **kwargs):
        super(GPMultiFidelitySearcher, self).update(config, reward, **kwargs)
        with self.LOCK:
            # TODO: This is pretty extraneous. It's easier for the underlying
            # gp_searcher to work directly with config dictionaries.
            # Just doing this for consistency so that gp_searcher works solely
            # with CS.Configuration input/outputs and this wrapper acts as the
            # interface responsible for casting to and from dicts.
            config_cs = self._to_config_cs(config)
            self.gp_searcher.update(config_cs, reward, **kwargs)
            # If evaluation task has terminated, cleanup pending evaluations
            # which may have been overlooked
            if kwargs.get('done', False) or kwargs.get('terminated', False):
                self.gp_searcher.cleanup_pending(config_cs)

    def register_pending(self, config, milestone=None):
        assert milestone is not None, \
            "This searcher works with a multi-fidelity scheduler only"
        with self.LOCK:
            # TODO: See above comment.
            config_cs = self._to_config_cs(config)
            self.gp_searcher.register_pending(config_cs, milestone)

    def _to_config_cs(self, config):
        return CS.Configuration(self.gp_searcher.hp_ranges.config_space,
                                values=config)


class GPFIFOSearcher(BaseSearcher):
    def __init__(self, gp_searcher: _GPFIFOSearcher):
        super(GPFIFOSearcher, self).__init__(
            gp_searcher.hp_ranges.config_space)
        self.gp_searcher = gp_searcher

    def get_config(self, **kwargs):
        with self.LOCK:
            config_cs = self.gp_searcher.get_config(**kwargs)
            return config_cs.get_dictionary()

    def update(self, config, reward, **kwargs):
        super(GPFIFOSearcher, self).update(config, reward, **kwargs)
        with self.LOCK:
            # TODO: This is pretty extraneous. It's easier for the underlying
            # gp_searcher to work directly with config dictionaries.
            # Just doing this for consistency so that gp_searcher works solely
            # with CS.Configuration input/outputs and this wrapper acts as the
            # interface responsible for casting to and from dicts.
            config_cs = self._to_config_cs(config)
            self.gp_searcher.update(config_cs, reward, **kwargs)

    def register_pending(self, config, milestone=None):
        assert milestone is None, \
            "This searcher does not work with multi-fidelity schedulers"
        with self.LOCK:
            # TODO: See above comment.
            config_cs = self._to_config_cs(config)
            self.gp_searcher.register_pending(config_cs)

    def _to_config_cs(self, config):
        return CS.Configuration(self.gp_searcher.hp_ranges.config_space,
                                values=config)
