import re


def perform_replacements(df, col, replacements):
    for k, v in replacements.items():
        df[col] = df[col].str.replace(k, v, flags=re.IGNORECASE)
        print(f'\t- {k} -> {v} finished')
    return df


def perform_extraction(df, col, extract_feature):
    for feature, text in extract_feature.items():
        df[feature] = df[col].str.extract(text, flags=re.IGNORECASE)
        print(f'\t- {feature} finished')
    return df
