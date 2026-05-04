# 🗑️ **Waste Forecast**
**Prédiction tonnages déchets NYC** (MAE **176.9t** = 4.1% ! 🥇)

[![Stars](https://img.shields.io/github/stars/mohammed9782/waste_forcast)](https://github.com/mohammed9782/waste_forcast)
[![Forks](https://img.shields.io/github/forks/mohammed9782/waste_forcast)](https://github.com/mohammed9782/waste_forcast)
[![MAE Record](https://img.shields.io/badge/MAE-176.9t-brightgreen)](https://github.com/mohammed9782/waste_forcast)
[![Ensemble](https://img.shields.io/badge/Ensemble-RF%2BCatBoost-blueviolet)](https://catboost.ai/)
[![Python](https://img.shields.io/badge/Python-3.10+-orange)](https://python.org)
[![License](https://img.shields.io/github/license/mohammed9782/waste_forcast)](https://github.com/mohammed9782/waste_forcast/blob/main/LICENSE)

## 🎯 **Résultats SOTA** ⭐ **Nouveau Record !**
| Modèle | Test MAE | Gain vs RF | CV MAE |
|--------|----------|------------|--------|
| **🥇 Ensemble RF+CatBoost** | **176.9t** | **+2.7%** | - |
| 🥈 **CatBoost** | **180.1t** | **+0.9%** | 287.2t |
| 🥉 **RandomForest** | 181.7t | **0.0%** | **218.2t** |
| LightGBM | 183.5t | -1.0% | 226.7t |
| XGBoost | 196.0t | -7.6% | 235.4t |
| LinearReg | 194.5t | -6.7% | 232.2t |

**Ensemble RF+CatBoost = NOUVEAU RECORD** 🎉 **176.9t** (+2.7% vs RF)

## 🚀 **Quick Start**
```bash
# Clone + setup
git clone https://github.com/mohammed9782/waste_forcast
cd waste_forcast
uv sync    # ou pip install -r requirements.txt

# 1. ENSEMBLE PROD (recommandé !)
uv run python src/ensemble_test.py --model-path models/ensemble_prod.pkl

# 2. CatBoost seul
uv run python src/train_pro.py --model-path models/catboost_prod.pkl

# 3. Benchmark complet
uv run python src/benchmark_models.py
```

## 📁 **Structure du projet**

waste_forcast/
├── 📊 data/
│ └── processed/dsny_features.csv # NYC + lags/rolling
├── ⚙️ config/
│ └── config.yaml # Hyperparams + métriques
├── 🎯 src/
│ ├── train_pro.py # Pipeline PROD
│ ├── benchmark_models.py # 5 algos benchmark
│ └── ensemble_test.py # 🏆 RF+CatBoost 176.9t
├── 📈 models/
│ ├── ensemble_prod.pkl # RECORD 176.9t
│ ├── catboost_prod.pkl # 180.1t
│ └── random_forest_prod.pkl # 181.7t
├── 📊 results/
│ └── benchmark_results.csv
├── 📋 README.md
└── pyproject.toml

text

## ⚙️ **Configuration** (`config/config.yaml`)
```yaml
model:                          # CatBoost (ou ensemble)
  type: CatBoostRegressor
  params:
    iterations: 500
    depth: 8
    learning_rate: 0.08

ensemble:                       # NOUVEAU !
  rf_weight: 0.5
  cat_weight: 0.5

features:                       # 8 clés
  - month_sin
  - borough
  - lag_1
  - lag_12
  - rolling_mean_3
  - papertonscollected
  - mgptonscollected

metrics:                        # LIVE
  ensemble_mae: 176.9           # 🥇 RECORD !
  catboost_mae: 180.1
  rf_mae: 181.7
  relative_error: 4.1%
```

## 🏆 **Ensemble RF+CatBoost = SOTA**

VotingRegressor(weights=[0.5, 0.5])

    RF : 181.7t (stabilité)

    CatBoost : 180.1t (catégoriel)
    = ENSEMBLE : 176.9t (+2.7%) 🎉

text

## 🔬 **Pourquoi ça marche ?**

RF = stable + non-linéaire
CatBoost = catégoriel natif + anti-fuite
Ensemble = moyenne erreurs → variance ↓

text

## 🔮 **Production Ready**
- ✅ **CLI argparse** complet
- ✅ **joblib** + métriques YAML live
- ✅ **Split chrono** 65/15/20%
- ✅ **Config Git** versionnée
- ✅ **Ensemble 176.9t** prêt déploiement

## 💼 **Freelance Portfolio**

Use cases :

    Optimisation tournées camions (déchets)

    Prévision énergie (utilités)

    Demande retail saisonnière

Tarif : 1000-2000€ (modèle + API + dashboard)
Clients : Startups IA, PME logistique

text

## 🛠️ **Dépendances**
```bash
uv add scikit-learn catboost lightgbm xgboost pandas numpy pyyaml joblib
```

## 📈 **Métriques détaillées**

🏆 ENSEMBLE : 176.9t (4.1%) ← RECORD
RMSE : 240.2t
R² : 0.94
Baseline naïve : 450t (15%)

text

## 🤝 **Contribuer**
1. Fork → clone repo
2. `uv sync`
3. Améliore → MAE < 176t ?
4. **PR** bienvenu ! 🏆

## 📄 **Licence**
[MIT](LICENSE) © Mohammed 2026