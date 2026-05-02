import argparse
import os
import pandas as pd
from config import params
from models.model_registry import MODEL_REGISTRY
from utils.Functions import split_dataset, save_experiment_report
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



def main():
    parser = argparse.ArgumentParser(description="Pipeline d'entraînement et d'évaluation")
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_REGISTRY.keys(),
                        help=f"Modèle à utiliser : {list(MODEL_REGISTRY.keys())}")
    args = parser.parse_args()

    # 1. Instanciation du modèle
    model = MODEL_REGISTRY[args.model]()
    print(f"\n{'='*50}")
    print(f"  Modèle sélectionné : {args.model}")
    print(f"{'='*50}\n")

    # 2. Chargement des données brutes
    data_path = params['data_dir'] + params['data_name']
    print(f"Chargement des données : {data_path}")
    df = pd.read_csv(data_path)

    # 3. Préparation (preprocessing propre à chaque modèle)
    print("Préparation des données...")
    X, y = model.prepare(df)

    # 4. Séparation train / test
    print("Séparation train/test...")
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"  Train : {X_train.shape[0]} exemples | Test : {X_test.shape[0]} exemples")

    # On récupère la config
    grid_search_config = params.get('grid_search', {})
    param_grid = grid_search_config.get(args.model)

    # On lance l'optimisation si elle est activée et configurée
    do_grid_search = params.get('do_grid_search', False)
    if do_grid_search and param_grid:
        try:
            print(f"\n--- Optimisation des hyperparamètres pour {args.model} ---")
            best_params = model.fine_tune(X_train, y_train, param_grid)
            print(f"Meilleurs paramètres trouvés : {best_params}\n")
        except Exception as e:
            print(f"Erreur lors de l'optimisation : {e}")
    else:
        if not do_grid_search:
            print(f"\nAucune optimisation demandée (do_grid_search=False). Utilisation des paramètres par défaut de {args.model}.")
        else:
            print(f"\nAucune optimisation configurée pour {args.model} dans param.yaml. Utilisation des paramètres par défaut.")

    # 5. Validation croisée (sur train uniquement)
    print("\nValidation croisée (TimeSeriesSplit, 5 folds)...")
    cv_results = model.cross_val(X_train, y_train)
    print("\nRésultats cross-validation :")
    for metric, scores in cv_results.items():
        print(f"  {metric:12s} -> val: {scores['val_mean']:.4f} +/- {scores['val_std']:.4f}"
              f"  |  train: {scores['train_mean']:.4f} +/- {scores['train_std']:.4f}")

    # 6. Entraînement final sur tout le train set
    print("\nEntraînement final...")
    model.train(X_train, y_train)

    # 7. Évaluation sur le jeu de test
    metrics = model.evaluate(X_test, y_test)
    
    # 8. Sauvegarde du rapport d'expérimentation
    save_experiment_report(args.model, cv_results, metrics, params['model_output_dir'])

    # 9. Sauvegarde du modèle
    model_dir = os.path.join(params['model_output_dir'], args.model)
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, args.model + ".pkl")
    model.save(save_path)


if __name__ == "__main__":
    main()