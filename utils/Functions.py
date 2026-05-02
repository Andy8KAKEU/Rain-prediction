import os
from datetime import datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from config import params


def split_dataset(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Divise X et y en ensembles d'entraînement et de test.
    Le preprocessing (standardisation) a déjà été fait par model.prepare().
    """
    # Pour les séries temporelles, on ne peut pas stratifier sans mélanger.
    return train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=False)


def get_cv_strategy(n_splits: int = 5, random_state: int = 42):
    """
    Retourne un TimeSeriesSplit pour la validation croisée.
    C'est la méthode recommandée pour les séries temporelles (évite la fuite de données).
    """
    return TimeSeriesSplit(n_splits=n_splits)

def save_experiment_report(model_name: str, cv_results: dict, metrics: dict, output_dir: str):
    """
    Sauvegarde les résultats complets de l'expérience (CV, métriques, rapport)
    dans un fichier texte au sein du dossier du modèle.
    """
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(model_dir, f"report_{timestamp}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"=== RAPPORT D'EXPERIMENTATION : {model_name.upper()} ===\n")
        f.write(f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("--- Validation Croisée (TimeSeriesSplit) ---\n")
        if cv_results:
            for metric, scores in cv_results.items():
                f.write(f"  {metric:12s} -> val: {scores['val_mean']:.4f} +/- {scores['val_std']:.4f} | train: {scores['train_mean']:.4f} +/- {scores['train_std']:.4f}\n")
        else:
            f.write("  Aucune donnée de validation croisée.\n")
            
        f.write("\n--- Evaluation Finale (Test Set) ---\n")
        for key, value in metrics.items():
            if key != "classification_report":
                f.write(f"  {key}: {value}\n")
                
        if "classification_report" in metrics:
            f.write("\n--- Rapport de Classification ---\n")
            f.write(metrics["classification_report"])
            
    print(f"Rapport d'experimentation sauvegarde -> {report_path}")
