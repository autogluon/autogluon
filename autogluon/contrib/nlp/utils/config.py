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
