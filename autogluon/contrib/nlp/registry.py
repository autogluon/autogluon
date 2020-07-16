from .utils.registry import Registry

BACKBONE_REGISTRY = Registry('Backbone Models')
TOKENIZER_REGISTRY = Registry('Tokenizer')
DATA_PARSER_REGISTRY = Registry('NLP Data CLI Parser')
DATA_MAIN_REGISTRY = Registry('NLP Data CLI Main Function')
NLP_PREPROCESS_REGISTRY = Registry('NLP Preprocess CLI')
