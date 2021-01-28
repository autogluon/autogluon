
# Raw types: Raw data type information grouped into families.
# For example: uint8, int8, int16, int32, and int64 features all map to 'int'
R_INT = 'int'
R_FLOAT = 'float'
R_OBJECT = 'object'
R_CATEGORY = 'category'
R_DATETIME = 'datetime'
R_BOOL = 'bool'  # TODO: R_BOOL/R_BOOLEAN?
# TODO: R_FLOAT_SPARSE/R_INT_SPARSE/R_CATEGORY_SPARSE?

# Special types: Meta information about the special meaning of a feature that is not present in the raw data.
# feature has been binned into discrete integer values from its original representation
S_BINNED = 'binned'

# feature was originally a datetime type that was converted to numeric
S_DATETIME_AS_INT = 'datetime_as_int'

# feature is a datetime in object form (string dates), which can be converted to datetime via pd.to_datetime
S_DATETIME_AS_OBJECT = 'datetime_as_object'

# feature is an object type that contains text information that can be utilized in natural language processing
S_TEXT = 'text'

# feature is a categorical that was originally text information. It may or may not still contain the raw text in its data.
S_TEXT_AS_CATEGORY = 'text_as_category'

# feature is a generated feature based off of a text feature but is not an ngram. Examples include character count, word count, symbol count, etc.
S_TEXT_SPECIAL = 'text_special'

# feature is a generated feature based off of a text feature that is an ngram.
S_TEXT_NGRAM = 'text_ngram'

# feature is a generated feature based off of a ML model's prediction probabilities of the label column for the row.
# Any model which takes a stack feature as input is a stack ensemble.
S_STACK = 'stack'
