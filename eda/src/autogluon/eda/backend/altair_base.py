class AltairMixin:
    QUANTITATIVE = 'quantitative'
    ORDINAL = 'ordinal'
    NOMINAL = 'nominal'
    TEMPORAL = 'temporal'
    GEOJSON = 'geojson'

    ENCODING_DATA_TYPES = {
        QUANTITATIVE: 'Q',
        ORDINAL: 'O',
        NOMINAL: 'N',
        TEMPORAL: 'T',
        GEOJSON: 'G',
    }
