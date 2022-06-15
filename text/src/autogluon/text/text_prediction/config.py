import yacs.config

class CfgNode(yacs.config.CfgNode):
    def clone_merge(self, cfg_filename_or_other_cfg):
        """Create a new cfg by cloning and mering with the given cfg

        Parameters
        ----------
        cfg_filename_or_other_cfg

        Returns
        -------

        """
        ret = self.clone()
        if isinstance(cfg_filename_or_other_cfg, str):
            ret.merge_from_file(cfg_filename_or_other_cfg)
            return ret
        elif isinstance(cfg_filename_or_other_cfg, CfgNode):
            ret.merge_from_other_cfg(cfg_filename_or_other_cfg)
            return ret
        elif cfg_filename_or_other_cfg is None:
            return ret
        else:
            raise TypeError('Type of config path is not supported!')

    def to_flat_dict(self):
        """Dump the config to a dictionary that is not nested.

        For example:

        Input:
            {'A': {'aaa': 1, 'bbb': 2}, 'B' : {'ccc': 3}
        Output:
            {'A.aaa': 1, 'A.bbb': 2, 'B.ccc': 3}
        """

        def convert_to_dict(cfg_node, key_list):
            from yacs.config import _assert_with_logging, _valid_type, _VALID_TYPES

            if not isinstance(cfg_node, CfgNode):
                _assert_with_logging(
                    _valid_type(cfg_node),
                    "Key {} with value {} is not a valid type; valid types: {}".format(
                        ".".join(key_list), type(cfg_node), _VALID_TYPES
                    ),
                )
                return cfg_node
            else:
                new_dict = dict()
                for k in cfg_node:
                    v = getattr(cfg_node, k)
                    sub_dict = convert_to_dict(v, key_list + [k])
                    if isinstance(sub_dict, dict):
                        for ck, cv in sub_dict.items():
                            new_dict[f'{k}.{ck}'] = cv
                    else:
                        new_dict[k] = sub_dict
                return new_dict
        return convert_to_dict(self, [])
