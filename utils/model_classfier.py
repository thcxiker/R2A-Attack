import yaml
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class ModelClassifier:
    """Classify models into strong/weak categories"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        strong_models: Optional[List[str]] = None,
        weak_models: Optional[List[str]] = None,
        default_classification: str = "weak"
    ):
        """
        Args:
            config_path: Path to YAML config file
            strong_models: List of strong models (overrides config)
            weak_models: List of weak models
            default_classification: Default classification ("strong" or "weak")
        """
        self.default_classification = default_classification
        
        # Load config from file
        if config_path is not None:
            config = self._load_config(config_path)
            self.strong_models = set(config.get('strong_models', []))
            self.weak_models = set(config.get('weak_models', []))
            rules = config.get('classification_rules', {})
            self.strong_patterns = [
                re.compile(p) for p in rules.get('strong_patterns', [])
            ]
            self.weak_patterns = [
                re.compile(p) for p in rules.get('weak_patterns', [])
            ]
        else:
            self.strong_models = set()
            self.weak_models = set()
            self.strong_patterns = []
            self.weak_patterns = []
        
        # Override with provided parameters
        if strong_models is not None:
            self.strong_models.update(strong_models)
        if weak_models is not None:
            self.weak_models.update(weak_models)
    
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def classify(self, model_name: str) -> str:
        """
        Classify a single model.
        
        Returns:
            "strong" or "weak"
        """
        # Check exact match
        if model_name in self.strong_models:
            return "strong"
        if model_name in self.weak_models:
            return "weak"
        
        # Check regex patterns
        for pattern in self.strong_patterns:
            if pattern.match(model_name):
                return "strong"
        
        for pattern in self.weak_patterns:
            if pattern.match(model_name):
                return "weak"
        
        # Return default
        return self.default_classification
    
    def classify_list(
        self,
        model_list: List[str]
    ) -> Tuple[List[str], List[str]]:
        """
        Classify a list of models.
        
        Returns:
            (strong_models, weak_models)
        """
        strong = []
        weak = []
        
        for model in model_list:
            if self.classify(model) == "strong":
                strong.append(model)
            else:
                weak.append(model)
        
        return strong, weak
    
    def get_classification_dict(
        self,
        model_list: List[str]
    ) -> Dict[str, str]:
        """
        Return a dictionary mapping models to classifications.
        
        Returns:
            {model_name: "strong"/"weak"}
        """
        return {
            model: self.classify(model)
            for model in model_list
        }


def classify_model_lists(
    model_lists: Dict[str, List[str]],
    classifier: ModelClassifier
) -> Dict[str, Dict[str, List[str]]]:
    """
    Batch classify model lists for multiple routers.
    
    Args:
        model_lists: {router_name: [model1, model2, ...]}
        classifier: ModelClassifier instance
    
    Returns:
        {
            router_name: {
                "strong": [model1, model2, ...],
                "weak": [model3, model4, ...]
            }
        }
    """
    results = {}
    
    for router_name, models in model_lists.items():
        strong, weak = classifier.classify_list(models)
        
        results[router_name] = {
            "strong": strong,
            "weak": weak
        }
    
    return results
