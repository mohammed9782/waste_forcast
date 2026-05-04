import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path: str = "config.yaml"):
    """Charge la configuration depuis YAML"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_data(df: pd.DataFrame, features: list, target: str):
    """Prépare X et y selon les features du config"""
    X = df[features].copy()
    X = pd.get_dummies(X, columns=["borough"], drop_first=True)
    y = df[target]
    return X, y

def create_year_split(df: pd.DataFrame):
    """Split chronologique par année"""
    df['year'] = df['month'].dt.year
    years = sorted(df['year'].unique())
    
    n_years = len(years)
    train_end = int(n_years * 0.65)
    val_end = int(n_years * 0.80)
    
    train = df[df['year'] <= years[train_end]]
    val = df[(df['year'] > years[train_end]) & (df['year'] <= years[val_end])]
    test = df[df['year'] > years[val_end]]
    
    print(f" Train : {train['year'].min()}-{train['year'].max()} ({len(train)} obs)")
    print(f" Val   : {val['year'].min()}-{val['year'].max()} ({len(val)} obs)")
    print(f" Test  : {test['year'].min()}-{test['year'].max()} ({len(test)} obs)")
    
    return train, val, test

def main():
    # 1. Parser CLI
    parser = argparse.ArgumentParser(description="Entraîner le modèle Random Forest de prévision des tonnages")
    parser.add_argument("--config-path", "-c", default="config.yaml", 
                       help="Chemin vers le fichier config")
    parser.add_argument("--data-path", "-d", default="./data/processed/dsny_features.csv",
                       help="Chemin vers les données")
    parser.add_argument("--model-path", "-m", default="./models/random_forest_production.pkl",
                       help="Chemin de sauvegarde du modèle")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Mode verbeux")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🔧 Config : {args.config_path}")
        print(f"📊 Données : {args.data_path}")
        print(f"💾 Modèle : {args.model_path}")
    
    # 2. Charger config
    config = load_config(args.config_path)
    print("📋 Configuration chargée")
    
    # 3. Charger et préparer données
    df = pd.read_csv(args.data_path)
    df["month"] = pd.to_datetime(df["month"])
    
    features = config["features"]
    target = config["target"]
    
    # 4. Split chronologique
    train, val, test = create_year_split(df)
    
    X_train, y_train = prepare_data(train, features, target)
    X_val, y_val = prepare_data(val, features, target)
    X_test, y_test = prepare_data(test, features, target)
    
    # 5. Modèle avec hyperparamètres optimaux
    print("🌳 Entraînement Random Forest...")
    model = RandomForestRegressor(**config["model"]["params"])
    model.fit(X_train, y_train)
    
    # 6. Métriques
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    mae_val = mean_absolute_error(y_val, y_pred_val)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    mae_rel_test = mae_test / y_test.mean() * 100
    
    print(f"\n📊 Résultats finaux :")
    print(f"   MAE Validation : {mae_val:.1f} tonnes ({mae_val/y_val.mean()*100:.1f}%)")
    print(f"   MAE Test       : {mae_test:.1f} tonnes ({mae_rel_test:.1f}%)")
    
    # 7. Sauvegarde modèle
    joblib.dump(model, args.model_path)
    
    # 8. Mise à jour config avec métriques récentes
    config["metrics"]["mae_val"] = float(mae_val)
    config["metrics"]["mae_test"] = float(mae_test)
    config["metrics"]["mae_test_relative"] = float(mae_rel_test)
    
    with open(args.config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n🚀 Modèle sauvegardé : {args.model_path}")
    print(f"📈 Métriques mises à jour dans config")

if __name__ == "__main__":
    main()