from ray.tune.suggest.basic_variant import BasicVariantGenerator
from ray.tune.suggest.hyperopt import HyperOptSearch

class SearcherFactory:
    
    searcher_presets = {
        'random': BasicVariantGenerator,
        'bayes': HyperOptSearch,
    }
    
    # These needs to be provided explicitly as searcher init args
    # These should be experiment specified required args
    searcher_required_args = {
        'random': [],
        'bayes': ['metric', 'mode'],
    }
    
    # These are the default values if user not specified
    # These should be non-experiment specific args, which we just pick a default value for the users
    # Can be overridden by the users
    searcher_default_args = {
        'random': {},
        'bayes': {},
    }
    
    @staticmethod
    def get_searcher(searcher_name: str, user_init_args, **kwargs):
        assert searcher_name in SearcherFactory.searcher_presets, f'{searcher_name} is not a valid option. Options are {SearcherFactory.searcher_presets.keys()}'
        init_args = {arg: kwargs[arg] for arg in SearcherFactory.searcher_required_args[searcher_name]}
        init_args.update(SearcherFactory.searcher_default_args[searcher_name])
        init_args.update(user_init_args)
        
        return SearcherFactory.searcher_presets[searcher_name](**init_args)
