import re

from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor


class Stage2FeaturesProcessor(AbstractPreprocessor):

    def __init__(self, is_training):
        self.name = "Stage2FeaturesProcessor"
        self.is_training = is_training

    def run(self, context, df, ):
        df['isriver'] = df['isriver'].fillna(0) > 0

        if self.is_training:
            df['root_cause'] = df['root_cause'].fillna('')

        df['short_description_has_link'] = df['short_description'].str.contains('amazon.com', flags=re.IGNORECASE)
        print('\t- short_description_has_link finished')

        contains_features = {  # tags presence of string
            'ticketed_by_infra_scripts': 'Submitted from Host: infra-scripts',
            'ticketed_by_itm_core': 'Ticketed by: itm-cor',
            'ticketed_by_itm_consoles': 'Ticketed by: consoles',
            'ticketed_by_itm_fcnapmon': 'Ticketed by: fcnapmon',
            'ticket_created_by_user': 'This (ticket|issue) was created by user',
            'ticket_created_by_flx_river': 'created by flx-river on behalf of',
            'ticket_created_by_flx_cstech': 'created by flx-cstech on behalf of',
            'ticket_created_by_infra_scripts': 'Submitted from Host: infra-scripts',
            'contains_defect_details_str': 'DEFECT DETAILS BELOW',
            'slim_ticket': 'Visit SLIM and generate Trouble Tickets for failing stations',
            'mdf_room_status_check': 'Perform the daily status check of the MDF room',
        }

        for feature, text in contains_features.items():
            df[feature] = df['details'].str.contains(text, flags=re.IGNORECASE)
            print(f'\t- {feature} finished')

        extract_feature = {  # extracts regex group as a feature
            'policy_number': 'http[s]?://policy.amazon.com/[\\w]+/([0-9]+)',
            'policy_section': 'http[s]?://policy.amazon.com/([\\w]+)/[0-9]+',
        }

        for feature, text in extract_feature.items():
            df[feature] = df['details'].str.extract(text, flags=re.IGNORECASE)
            print(f'\t- {feature} finished')

        return df
