from pathlib import Path
from typing import Self

import yaml


class ConfigSaveLoadMixin(yaml.YAMLObject):
    
    def save(self, path: Path) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self, f, default_flow_style=False)


    @classmethod
    def load(cls, path: Path) -> Self:

        with open(path, 'r') as f:
            # It's unsafe, but not unsafer than the pickle module
            config = yaml.unsafe_load(f)

        return config
