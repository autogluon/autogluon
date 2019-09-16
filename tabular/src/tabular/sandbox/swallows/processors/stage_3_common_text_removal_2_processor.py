import re

from tabular.sandbox.swallows.processors.constants import OTHER_AMZ_LINK, OTHER_EXT_LINK, WFSS_WORKFLOW, UUID
from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements


class Stage3CommonTextRemovalProcessor2(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage3CommonTextRemovalProcessor2"

    def run(self, context, df):
        replacements = [
            'NARF Audit picked up discrepancies for the device xxnarflink',
            'Perform the daily status check of the MDF room. Follow SOP: xxpolicylink',
        ]
        r = '|'.join(replacements)
        df = perform_replacements(df, 'details', {f'({r})': ''})

        df = perform_replacements(df, 'details', {
            'http[s]?://[a-z0-9\\-.]*amazon\\.[\\w/.#?=&\\-%{}]*': OTHER_AMZ_LINK,
            'http[s]?://[a-z0-9\\-.]*[\\w/.#?=&\\-%]*': OTHER_EXT_LINK,
            'WFSS Workflow name: (.*) Hello': WFSS_WORKFLOW,
            '[a-f\\d]{8}-[a-f\\d]{4}-[a-f\\d]{4}-[a-f\\d]{4}-[a-f\\d]{12}': UUID,

        })

        # Add features
        df['amz_domain_other_link'] = df['details'].str.contains(OTHER_AMZ_LINK.strip(), flags=re.IGNORECASE)
        df['amz_external_link'] = df['details'].str.contains(OTHER_EXT_LINK.strip(), flags=re.IGNORECASE)
        df['wfss_workflow_present'] = df['details'].str.contains(WFSS_WORKFLOW.strip(), flags=re.IGNORECASE)
        df['has_uuid'] = df['details'].str.contains(UUID.strip(), flags=re.IGNORECASE)

        return df
