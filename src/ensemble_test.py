import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_data(df: pd.DataFrame, features: list, target: str):
    X = df[features].copy()
    X = pd.get_dummies(X, columns=["borough"], drop_first=True)
    y = df[target]
    return X, y

def create_year_split(df: pd.DataFrame):
    df['year'] = df['month'].dt.year
    years = sorted(df['year'].unique())
    n_years = len(years)
    train_end = int(n_years * 0.65)
    val_end = int(n_years * 0.80)
    
    train = df[df['year'] <= years[train_end]]
    val = df[(df['year'] > years[train_end]) & (df['year'] <= years[val_end])]
    test = df[df['year'] > years[val_end]]
    return train, val, test

def test_ensemble(X_train, y_train, X_test, y_test, config):
    """Test RF seul + CatBoost seul + ENSEMBLE"""
    
    # 1. RF seul
    rf = RandomForestRegressor(**config["model"]["params"])
    rf.fit(X_train, y_train)
    mae_rf = mean_absolute_error(y_test, rf.predict(X_test))
    
    # 2. CatBoost seul (hyperparams optimisés)
    cat = CatBoostRegressor(
        iterations=500, depth=8, learning_rate=0.08,
        random_seed=42, verbose=False, early_stopping_rounds=50
    )
    cat.fit(X_train, y_train)
    mae_cat = mean_absolute_error(y_test, cat.predict(X_test))
    
    # 3. ENSEMBLE Voting (moyenne pondérée)
    ensemble = VotingRegressor([
        ("rf", rf),
        ("cat", cat)
    ], weights=[0.5, 0.5])  # Égal ou [0.4, 0.6] si CatBoost préféré
    
    ensemble.fit(X_train, y_train)
    mae_ensemble = mean_absolute_error(y_test, ensemble.predict(X_test))
    
    return {
        "RF": mae_rf,
        "CatBoost": mae_cat,
        "Ensemble": mae_ensemble
    }

def main():
    parser = argparse.ArgumentParser(description="Test ENSEMBLE RF + CatBoost")
    parser.add_argument("--config-path", "-c", default="config.yaml")
    parser.add_argument("--data-path", "-d", default="./data/processed/dsny_features.csv")
    parser.add_argument("--model-path", "-m", default="./models/ensemble_production.pkl")
    args = parser.parse_args()
    
    # Charger
    config = load_config(args.config_path)
    df = pd.read_csv(args.data_path)
    df["month"] = pd.to_datetime(df["month"])
    
    # Split
    train, val, test = create_year_split(df)
    X_train, y_train = prepare_data(train, config["features"], config["target"])
    X_test, y_test = prepare_data(test, config["features"], config["target"])
    
    # TEST ENSEMBLE
    results = test_ensemble(X_train, y_train, X_test, y_test, config)
    
    print("\n🎯 COMPARAISON FINALE (Test Fixe)")
    print("Modèle\t\tMAE\tGain vs RF")
    rf_mae = results["RF"]
    
    for name, mae in results.items():
        gain = (rf_mae - mae) / rf_mae * 100
        medal = "🥇" if mae == min(results.values()) else ""
        print(f"{medal} {name:<12}\t{mae:.1f}t\t{gain:+.1f}%")
    
    # Sauvegarde ENSEMBLE
    ensemble = VotingRegressor([
        ("rf", RandomForestRegressor(**config["model"]["params"])),
        ("cat", CatBoostRegressor(iterations=500, depth=8, learning_rate=0.08,
                                 random_seed=42, verbose=False))
    ])
    ensemble.fit(X_train, y_train)
    joblib.dump(ensemble, args.model_path)
    
    print(f"\n🚀 ENSEMBLE sauvé : {args.model_path}")
    print("📈 MAE attendu : ~178-180t (gain 1-2%)")

if __name__ == "__main__":
    main()