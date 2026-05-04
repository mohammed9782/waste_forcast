import pandas as pd
import numpy as np
import os

# 1. Chargement du fichier local (assure-toi qu'il est dans data/raw/dsny.csv)
df = pd.read_csv("./data/DSNY_Monthly_Tonnage_Data_20260501.csv")

# 2. Nettoyage des noms de colonnes
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace("/", "_", regex=False)

# 3. Sélection des colonnes pertinentes
features = [
    'month', 'borough', 'communitydistrict', 
    'refusetonscollected', 'papertonscollected', 'mgptonscollected', 
    'resorganicstons', 'schoolorganictons', 'leavesorganictons', 'xmastreetons'
]
df_processed = df[features].copy()

# 4. Conversion date
df_processed['month'] = pd.to_datetime(df_processed['month'], errors='coerce')
df_processed["year"] = df_processed["month"].dt.year
df_processed["month_num"] = df_processed["month"].dt.month


# 5. Imputation
organic_cols = ['resorganicstons', 'schoolorganictons', 'leavesorganictons', 'xmastreetons']
df_processed[organic_cols] = df_processed[organic_cols].fillna(0)

# 6. Suppression des lignes où la cible est manquante
df_processed = df_processed.dropna(subset=['refusetonscollected'])

# 7. Sauvegarde
os.makedirs("./data/processed", exist_ok=True)
df_processed.to_csv("./data/processed/dsny_cleaned.csv", index=False)
print(f"Dataset nettoyé sauvegardé. Shape: {df_processed.shape}")