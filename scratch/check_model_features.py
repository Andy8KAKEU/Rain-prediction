import joblib
import os
import pandas as pd
from config import params

model_name = "sgd"
model_path = os.path.join(params['model_output_dir'], model_name, f"{model_name}.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
    print(f"Modèle : {model_name}")
    
    # On regarde ce que le modèle scikit-learn a mémorisé
    if hasattr(model, "feature_names_in_"):
        print("Features attendues (via feature_names_in_) :")
        print(model.feature_names_in_)
        print(f"Nombre total : {len(model.feature_names_in_)}")
    elif hasattr(model, "feature_names_"):
        print("Features attendues (via feature_names_) :")
        print(model.feature_names_)
        print(f"Nombre total : {len(model.feature_names_)}")
    else:
        print("Aucun attribut de nom de feature trouvé sur le modèle.")
else:
    print(f"Fichier non trouvé : {model_path}")
