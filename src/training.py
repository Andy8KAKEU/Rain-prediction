from models.model_registry import MODEL_REGISTRY
import pandas as pd
import joblib
from config import params
from utils.Functions import split_dataset

def train_model(model_name):
    """
    Load processed data, train the model, and save it.
    """

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in MODEL_REGISTRY")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class()

    print(f"Loading data from {params['data_dir']}{params['data_name']}...")
    df = pd.read_csv(params['data_dir'] + params['data_name'])

    print("Standardizing dataset...")
    dataset = model.preparation(df)
    
    print("Splitting dataset...")
    cv_strategy = get_cv_strategy()

    
    print("Training model...")
    result = cross_validate(model, dataset[params['colonnes_numeriques']], dataset[params['colonne_cible']], 
                            cv=cv_strategy, scoring=['roc_auc', 'accuracy', 'precision', 'recall', 'f1'], 
                            return_train_score=True)
                            
    print("Training completed.")
    print(result)

    model.train(dataset[params['colonnes_numeriques']], dataset[params['colonne_cible']])
    print(f"Saving model to {params['model_output_dir']}{model_name}...")
    joblib.dump(model, params['model_output_dir'] + model_name + ".pkl")

    return model

if __name__ == "__main__":
    # Example usage
    #train_model("RandomForestClassifier")
    pass
