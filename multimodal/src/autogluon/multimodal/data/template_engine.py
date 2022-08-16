import numpy as np

from autogluon.multimodal.data.templates import Template, DatasetTemplates, TemplateCollection
from omegaconf import OmegaConf



class TemplateEngine():
    """
    Class to manage the selection and use of templates.
    """

    def __init__(self, template_config: dict):
        """
        Initialize the TemplateEngine using preset templates from existing datasets or custom templates specified in config config.data.templates, if specified.

        Parameters
        ---------------
        template_config
            The templates configuration specified in config.data.templates.
        """
        self.templates = []
        self.template_config = template_config
        collection = TemplateCollection()
        self.all_datasets = collection.keys
        self.preset_templates = OmegaConf.select(self.template_config, "preset_templates")
        self.custom_templates = OmegaConf.select(self.template_config, "custom_templates")

        if self.preset_templates:
            assert len(self.preset_templates) == 2, f"Preset templates has the wrong format. Needs to be [DATASET, SUBSET]."
            dataset_templates = DatasetTemplates(self.preset_templates[0], self.preset_templates[1])
            self.templates += dataset_templates.templates.values()

        if self.custom_templates:
            for key, value in self.custom_templates.items():
                template = Template(key, value.template, 'custom', answer_choices=value.answer_choices)
                self.templates.append(template)


    def has_templates(self):
        return len(self.templates)  > 0

    def get_templates(self):
        return self.templates

    def sample_and_apply_template(self, example: dict):
        """
        Randomly sample a template from the collection of available templates and apply it to the sample.
        If collection of templates is empty return original sample.

        Parameters
        ---------------
        example
            A data sample, i.e. a dictionary of text columns.
        
        Returns
        ------------------
        A tuple consisting of the selected tuple and the sample after the template has been applied to it.
        """
        if not self.templates:
            return [None, example]
        template = np.random.choice(self.templates)
        return [template, template.apply(example)]
