import yaml
import os

# Construire le chemin absolu vers param.yaml de manière robuste
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
yaml_path = os.path.join(BASE_DIR, 'param.yaml')

# Charger le YAML une seule fois
with open(yaml_path, 'r', encoding='utf-8') as file:
    params = yaml.safe_load(file)

# Tu peux ajouter des fonctions si nécessaire
def get_param(key):
    return params.get(key)