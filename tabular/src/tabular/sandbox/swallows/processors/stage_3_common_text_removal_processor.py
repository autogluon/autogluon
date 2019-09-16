from tabular.sandbox.swallows.processors.preprocessor import AbstractPreprocessor
from tabular.sandbox.swallows.processors.utils import perform_replacements


class Stage3CommonTextRemovalProcessor(AbstractPreprocessor):

    def __init__(self):
        self.name = "Stage3CommonTextRemovalProcessor"

    def run(self, context, df):
        replacements = [
            'Visit SLIM and generate Trouble Tickets for failing stations.*',
            'Availability: Note: . Live chat support is now available 24 hours a day, 7 days a week.*',
            'This ticket was created by user \\w+',
            'This issue was created by user \\w+',
            'created by flx-river on behalf of \\w+',
            'created by flx-cstech on behalf of cstech.*',
            'Submitted from Host: infra-scripts.*amazon.com',
            'Ticketed by: itm-cor.*This is an automatically generated ticket.*',
            'Ticketed by: consoles-.*amazon.com.*',
            'Ticketed by: fcnapmon-.*amazon.com.*',
            '[Tt]hank(s| yo[u]?|)',
            'DEFECT DETAILS BELOW',
            'This ticket is cut by River.',
            'These discrepancies prevent NARF AutoPush from pushing to this device Remember to update NARF first! Please read the problems below.*ConfigAudits for more information.: '
        ]

        r = '|'.join(replacements)
        return perform_replacements(df, 'details', {f'({r})': ' '})
