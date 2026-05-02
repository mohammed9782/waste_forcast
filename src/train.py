import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import numpy as np

def prepare_data(df: pd.DataFrame, target: str = "refusetonscollected"):
    """Prépare X et y avec features robustes"""
    features = [
        "month_sin", "month_cos", "borough",
        "lag_1", "lag_12", "rolling_mean_3",
        "papertonscollected", "mgptonscollected"
    ]
    
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
    
    print(f"Train : {train['year'].min()}-{train['year'].max()} ({len(train)} obs)")
    print(f"Val   : {val['year'].min()}-{val['year'].max()} ({len(val)} obs)")
    print(f"Test  : {test['year'].min()}-{test['year'].max()} ({len(test)} obs)")
    
    return train, val, test

def main():
    # Chargement
    df = pd.read_csv("./data/processed/dsny_features.csv")
    df["month"] = pd.to_datetime(df["month"])
    
    # Split chronologique
    train, val, test = create_year_split(df)
    
    # Préparation features
    X_train, y_train = prepare_data(train)
    X_val, y_val = prepare_data(val)
    X_test, y_test = prepare_data(test)
    
    # Modèle Random Forest
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Prédictions et métriques
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    print("\n📊 Résultats :")
    print(f"MAE Validation : {mean_absolute_error(y_val, y_pred_val):.1f} tons")
    print(f"MAE Test       : {mean_absolute_error(y_test, y_pred_test):.1f} tons")

    # Erreur relative
    mae_relative_val = mean_absolute_error(y_val, y_pred_val) / y_val.mean() * 100
    mae_relative_test = mean_absolute_error(y_test, y_pred_test) / y_test.mean() * 100

    print(f"MAE relative Validation : {mae_relative_val:.1f}%")
    print(f"MAE relative Test       : {mae_relative_test:.1f}%")
    print(f"Nombre de camions erronés : ~{mae_relative_test/3:.1f} (30t/camion)")
    
    # Importance des variables
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\n🏆 Top 10 features importantes :")
    print(importance.head(10))
    
    # Sauvegarde
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./reports", exist_ok=True)
    joblib.dump(model, "./models/random_forest_district_v1.pkl")
    importance.to_csv("./reports/feature_importance_v1.csv", index=False)
    
    print("\n✅ Modèle sauvegardé + importance exportée")

if __name__ == "__main__":
    main()