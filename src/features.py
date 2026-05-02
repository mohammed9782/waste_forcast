import pandas as pd
import numpy as np


def add_temporal_features(df: pd.DataFrame, target_col: str = "refusetonscollected") -> pd.DataFrame:
    df = df.sort_values("month")
    
    # Encodage cyclique du mois (sin/cos)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12)
    
    # Lags (retards)
    df["lag_1"] = df[target_col].shift(1)
    df["lag_12"] = df[target_col].shift(12)
    
    # Moyenne glissante
    df["rolling_mean_3"] = df[target_col].shift(1).rolling(window=3).mean()
    
    return df.dropna()


def prepare_features(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    df["month"] = pd.to_datetime(df["month"])
    
    # Créer identifiant unique = borough + communitydistrict
    df["district_id"] = df["borough"] + "_" + df["communitydistrict"].astype(str)
    
    print(f"Nombre de districts uniques : {df['district_id'].nunique()}")
    
    # Grouper par district_id pour calculer lags par zone
    df_feat = df.groupby("district_id").apply(add_temporal_features).reset_index(drop=True)
    
    df_feat.to_csv(output_path, index=False)
    print(f"✅ Features créées et sauvegardées dans {output_path}")
    print(f"Shape final : {df_feat.shape}")


if __name__ == "__main__":
    prepare_features("./data/processed/dsny_cleaned.csv", "./data/processed/dsny_features.csv")