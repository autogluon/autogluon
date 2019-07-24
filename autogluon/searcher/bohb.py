from .searcher BaseSearcher
class BOHB(BaseSearcher):
    """BOHB Searcher
    """
    def get_config(self, budget=None):
        """Function to sample a new configuration
        This function is called inside Hyperband to query a new configuration
        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled
        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        return self.configspace.sample_configuration().get_dictionary()

    def update(self, *args, **kwargs):
        """Update the searcher with the newest metric report
        """
        super(RandomSampling, self).update(*args, **kwargs)

