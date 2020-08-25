from typing import Optional, Union
import logging
import numpy as np
import ConfigSpace as CS
import json
from collections import OrderedDict

from .config_ext import ExtendedConfiguration
from ..datatypes.tuning_job_state import TuningJobState

logger = logging.getLogger(__name__)


def _to_key(
        config: Union[CS.Configuration, dict],
        configspace_ext: Optional[ExtendedConfiguration]) -> \
        (str, dict, Optional[int]):
    if isinstance(config, CS.Configuration):
        config_dict = config.get_dictionary()
    else:
        assert isinstance(config, dict)
        config_dict = config
    if configspace_ext is None:
        resource = None
    else:
        resource = config_dict.get(configspace_ext.resource_attr_name)
        if resource is not None:
            config_dict = configspace_ext.remove_resource(config, as_dict=True)
    return json.dumps(OrderedDict(config_dict)), config_dict, resource


class DebugLogPrinter(object):
    """
    Supports a concise debug log, currently specialized to GPFIFOSearcher (it
    could be extended to GPMultiFidelitySearcher as well).

    The log is made concise and readable by a few properties:
    - configs are mapped to config IDs 0, 1, 2, ... as they get returned by
        get_config. For multi-fidelity schedulers, extended config IDs are
        of the form "<k>:<r>", k the ID of the config, <r> the resource
        parameter. Note that even in this case, configs coming out of
        get_config are not extended
    - Information about get_config is displayed in a single block. For that,
      different parts are first collected until the end of get_config

    """
    def __init__(
            self, configspace_ext: Optional[ExtendedConfiguration] = None):
        self.config_counter = 0
        self._config_id = dict()
        self.block_info = dict()
        self.get_config_type = None
        self.configspace_ext = configspace_ext

    def config_id(self, config: Union[CS.Configuration, dict]) -> str:
        config_key, _, resource = _to_key(config, self.configspace_ext)
        _id = str(self._config_id[config_key])
        if resource is None:
            return _id
        else:
            return _id + ':{}'.format(resource)

    def start_get_config(self, gc_type):
        assert gc_type in {'random', 'BO'}
        assert self.get_config_type is None, \
            "Block for get_config of type '{}' is currently open".format(
                self.get_config_type)
        self.get_config_type = gc_type
        logger.info("Starting get_config[{}] for config_id {}".format(
            gc_type, self.config_counter))

    def set_final_config(self, config: Union[CS.Configuration, dict]):
        assert self.get_config_type is not None, "No block open right now"
        config_key, config, resource = _to_key(config, self.configspace_ext)
        assert resource is None, \
            "set_final_config: config must not be extended"
        _id = self._config_id.get(config_key)
        assert _id is None, \
            "Config {} already has been assigned a config ID = {}".format(
                config, _id)
        # Counter is advanced in write_block
        self._config_id[config_key] = self.config_counter
        entries = ['{}: {}'.format(k, v) for k, v in config.items()]
        msg = '\n'.join(entries)
        self.block_info['final_config'] = msg

    def set_state(self, state: TuningJobState):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        labeled_configs = [
            x.candidate for x in state.candidate_evaluations]
        labeled_str = ', '.join(
            [self.config_id(x) for x in labeled_configs])
        pending_str = ', '.join(
            [self.config_id(x) for x in state.pending_candidates])
        msg = 'Labeled: ' + labeled_str + '. Pending: ' + pending_str
        self.block_info['state'] = msg

    def set_targets(self, targets: np.ndarray):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'Targets: ' + str(targets.reshape((-1,)))
        self.block_info['targets'] = msg

    def set_gp_params(self, params: dict):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'GP params:' + str(params)
        self.block_info['params'] = msg

    def set_fantasies(self, fantasies: np.ndarray):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        msg = 'Fantasized targets:\n' + str(fantasies)
        self.block_info['fantasies'] = msg

    def set_init_config(self, config: Union[CS.Configuration, dict],
                        top_scores: np.ndarray = None):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        _, config, _ = _to_key(config, self.configspace_ext)
        entries = ['{}: {}'.format(k, v) for k, v in config.items()]
        msg = "Started BO from (top scorer):\n" + '\n'.join(entries)
        if top_scores is not None:
            msg += ("\nTop score values: " + str(top_scores.reshape((-1,))))
        self.block_info['start_config'] = msg

    def set_num_evaluations(self, num_evals: int):
        assert self.get_config_type == 'BO', "Need to be in 'BO' block"
        self.block_info['num_evals'] = num_evals

    def write_block(self):
        assert self.get_config_type is not None, "No block open right now"
        info = self.block_info
        if 'num_evals' in info:
            parts = ['[{}: {}] ({} evaluations)'.format(
                self.config_counter, self.get_config_type, info['num_evals'])]
        else:
            parts = ['[{}: {}]'.format(
                self.config_counter, self.get_config_type)]
        parts.append(info['final_config'])
        if self.get_config_type == 'BO':
            parts.extend(
                [info['start_config'], info['state'], info['targets'],
                 info['params']])
            if 'fantasies' in info:
                parts.append(info['fantasies'])
        msg = '\n'.join(parts)
        logger.info(msg)
        # Advance counter
        self.config_counter += 1
        self.get_config_type = None
        self.block_info = dict()

    def get_mutable_state(self) -> dict:
        return {
            'config_counter': self.config_counter,
            'config_id': self._config_id}

    def set_mutable_state(self, state: dict):
        assert self.get_config_type is None, \
            "Block for get_config of type '{}' is currently open".format(
                self.get_config_type)
        self.config_counter = state['config_counter']
        self._config_id = state['config_id']
        self.block_info = dict()
