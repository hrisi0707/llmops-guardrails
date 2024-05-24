import json
import re
from copy import deepcopy
from logging import getLogger
from typing import Optional

from langkit import LangKitConfig, lang_config


diagnostic_logger = getLogger(__name__)


class PatternLoader:
    def __init__(self, config: Optional[LangKitConfig] = None):
        self.config: LangKitConfig = config or deepcopy(lang_config)
        self.regex_groups = self.load_patterns()

    def load_patterns(self):
        json_path = self.config.pattern_file_path
        try:
            skip = False
            with open(json_path, "r") as myfile:
                _REGEX_GROUPS = json.load(myfile)
            regex_groups = []
            for group in _REGEX_GROUPS:
                compiled_expressions = []
                for expression in group["expressions"]:
                    compiled_expressions.append(re.compile(expression))

                regex_groups.append(
                    {"name": group["name"], "expressions": compiled_expressions}
                )
                diagnostic_logger.info(f"Loaded regex pattern for {group['name']}")
        except FileNotFoundError:
            skip = True
            diagnostic_logger.warning(f"Could not find {json_path}")
        except json.decoder.JSONDecodeError as json_error:
            skip = True
            diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
        if not skip:
            return regex_groups
        return None

    def set_config(self, config: LangKitConfig):
        self.config = config

    def update_patterns(self):
        self.regex_groups = self.load_patterns()

    def get_regex_groups(self):
        return self.regex_groups


class PresidioEntityLoader:
    def __init__(self, config: Optional[LangKitConfig] = None):
        self.config: LangKitConfig = config or deepcopy(lang_config)
        self.entities = self.load_entities()

    def load_entities(self):
        json_path = self.config.pii_entities_file_path
        try:
            skip = False
            with open(json_path, "r") as myfile:
                _ENTITIES = json.load(myfile)
            entities = []
            for entity in _ENTITIES["entities"]:
                entities.append(entity)
                diagnostic_logger.info(f"Loaded entity pattern for {entity}")
        except FileNotFoundError:
            skip = True
            diagnostic_logger.warning(f"Could not find {json_path}")
        except json.decoder.JSONDecodeError as json_error:
            skip = True
            diagnostic_logger.warning(f"Could not parse {json_path}: {json_error}")
        if not skip:
            return entities
        return None

    def set_config(self, config: LangKitConfig):
        self.config = config

    def update_entities(self):
        self.entities = self.load_entities()

    def get_entities(self):
        return self.entities
