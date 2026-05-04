import argparse
import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression  # ← AJOUTÉ
import lightgbm as lgb
import xgboost as xgb
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

def create_year_split(df: pd.DataFrame):  # ← AJOUTÉ
    df['year'] = df['month'].dt.year
    years = sorted(df['year'].unique())
    n_years = len(years)
    train_end = int(n_years * 0.65)
    val_end = int(n_years * 0.80)
    
    train = df[df['year'] <= years[train_end]]
    val = df[(df['year'] > years[train_end]) & (df['year'] <= years[val_end])]
    test = df[df['year'] > years[val_end]]
    return train, val, test

def benchmark_models(X, y, config):
    tscv = TimeSeriesSplit(n_splits=3)
    results = {}
    
    models = {
        "RandomForest": RandomForestRegressor(**config["model"]["params"]),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   random_state=42, n_jobs=-1, tree_method='hist'),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                     random_state=42, n_jobs=-1, verbosity=-1),
        "CatBoost": CatBoostRegressor(iterations=200, depth=8, learning_rate=0.1,
                                     random_seed=42, verbose=False),
        "LinearReg": LinearRegression()  # ← SIMPLIFIÉ
    }
    
    print("\n🏆 BENCHMARK (CV TimeSeriesSplit)")
    print("Modèle\t\tMAE CV")
    
    for name, model in models.items():
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            scores.append(mean_absolute_error(y_val, y_pred))
        
        mae_cv = np.mean(scores)
        results[name] = mae_cv
        print(f"{name:<15}\t{mae_cv:.1f}t")
    
    return results

def test_final_fixed(X_train, y_train, X_test, y_test, config):  # ← CORRIGÉ
    models = {
        "RandomForest": RandomForestRegressor(**config["model"]["params"]),
        "XGBoost": xgb.XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                   random_state=42, n_jobs=-1, tree_method='hist'),
        "LightGBM": lgb.LGBMRegressor(n_estimators=200, max_depth=8, learning_rate=0.1,
                                     random_state=42, n_jobs=-1, verbosity=-1),
        "CatBoost": CatBoostRegressor(iterations=200, depth=8, learning_rate=0.1,
                                     random_seed=42, verbose=False),
        "LinearReg": LinearRegression()
    }
    
    results = {}
    print("\n📊 TEST FINAL FIXE")
    print("Modèle\t\tMAE Test")
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        results[name] = mae
        print(f"{name:<15}\t{mae:.1f}t")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark algorithmes vs RandomForest")
    parser.add_argument("--config-path", "-c", default="config.yaml")
    parser.add_argument("--data-path", "-d", default="./data/processed/dsny_features.csv")
    args = parser.parse_args()
    
    # Charger données
    config = load_config(args.config_path)
    df = pd.read_csv(args.data_path)
    df["month"] = pd.to_datetime(df["month"])
    
    features = config["features"]
    target = config["target"]
    
    X, y = prepare_data(df, features, target)
    
    # SPLIT pour test fixe
    train, val, test = create_year_split(df)
    X_train, y_train = prepare_data(train, features, target)
    X_test, y_test = prepare_data(test, features, target)
    
    # BENCHMARK CV
    results_cv = benchmark_models(X, y, config)
    
    # Classement CV
    print("\n🎯 CLASSEMENT CV (MAE ↓)")
    sorted_cv = sorted(results_cv.items(), key=lambda x: x[1])
    for i, (name, mae) in enumerate(sorted_cv, 1):
        medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else ""
        print(f"{medal} {i}. {name:<15} {mae:.1f}t")
    
    # TEST FINAL FIXE
    results_fixed = test_final_fixed(X_train, y_train, X_test, y_test, config)
    
    # Classement Test Fixe
    print("\n🎯 CLASSEMENT TEST FIXE (MAE ↓)")
    sorted_fixed = sorted(results_fixed.items(), key=lambda x: x[1])
    for i, (name, mae) in enumerate(sorted_fixed, 1):
        medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else ""
        print(f"{medal} {i}. {name:<15} {mae:.1f}t")
    
    # Sauvegarde
    pd.DataFrame({"CV": results_cv, "Test_Fixe": results_fixed}).T.to_csv("benchmark_results.csv")
    print("\n💾 Résultats sauvés : benchmark_results.csv")

if __name__ == "__main__":
    main()