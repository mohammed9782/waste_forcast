import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
import joblib

def prepare_data(df: pd.DataFrame, target: str = "refusetonscollected"):
    features = [
        "month_sin", "month_cos", "borough",
        "lag_1", "lag_12", "rolling_mean_3",
        "papertonscollected", "mgptonscollected"
    ]
    
    X = df[features].copy()
    X = pd.get_dummies(X, columns=["borough"], drop_first=True)
    y = df[target]
    
    return X, y

def main():
    # Chargement
    df = pd.read_csv("./data/processed/dsny_features.csv")
    df["month"] = pd.to_datetime(df["month"])
    
    # Train set pour tuning
    df_train = df[df['year'] <= df['year'].quantile(0.8)]
    X_train, y_train = prepare_data(df_train)
    
    print(f"Tuning sur {len(X_train)} observations")
    
    # Hyperparamètres prioritaires pour ton cas
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': ['sqrt', 0.3, 0.5]
    }
    
    # Cross-validation temporelle (3 folds)
    tscv = TimeSeriesSplit(n_splits=3)
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid = GridSearchCV(
        rf, param_grid, 
        cv=tscv, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )
    
    print("🚀 Lancement de GridSearch...")
    grid.fit(X_train, y_train)
    
    print("\n🏆 Meilleurs hyperparamètres :")
    print(grid.best_params_)
    print(f"Meilleur score CV : {-grid.best_score_:.1f} tons MAE")
    
    # Test sur données tenues à l'écart
    df_test = df[df['year'] > df['year'].quantile(0.8)]
    X_test, y_test = prepare_data(df_test)
    
    y_pred = grid.best_estimator_.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    
    print(f"\n📊 MAE Test final : {mae_test:.1f} tons ({mae_test/y_test.mean()*100:.1f}%)")
    
    # Sauvegarde
    joblib.dump(grid.best_estimator_, "./models/random_forest_tuned.pkl")
    pd.DataFrame(grid.cv_results_).to_csv("./reports/gridsearch_results.csv", index=False)
    
    print("\n✅ Modèle tuné sauvegardé !")

if __name__ == "__main__":
    main()