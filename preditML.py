import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from datetime import timedelta



def Visu(y_test,y_pred_test,model_name):

    if isinstance(y_test, pd.Series) and isinstance(y_test.index, pd.DatetimeIndex):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.index, y_test, label='Rzeczywiste wartości wolumenu (Test)', color='blue')
        plt.plot(y_test.index, y_pred_test, label='Przewidywane wartości wolumenu (Test)', color='red', linestyle='--')
        plt.title(f'Trend Wolumenu i Predykcje {model_name} na zbiorze testowym')
        plt.xlabel('Data')
        plt.ylabel('Wolumen')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        plot_filename_ts = f'{model_name}_Price_timeseries_predictions_test_set.png'
        plt.savefig(f"{plot_filename_ts}")
        print(f"Wykres trendu czasowego Linear Regression został zapisany jako '{plot_filename_ts}'")
        plt.show()
    else:
        print("\nOstrzeżenie: Nie można zwizualizować predykcji w funkcji czasu, ponieważ y_test nie ma indeksu datowego.")
        print("Upewnij się, że y_test jest serią Pandas z indeksem typu DatetimeIndex.")




data = pd.read_csv('data/apple_stock.csv')
data = data[['Unnamed: 0', 'Adj Close']]
columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
data[columndate] = pd.to_datetime(data[columndate])
data.set_index(columndate, inplace=True)

end_date = data.index.max()
start_date_5_years_ago = end_date - pd.DateOffset(years=5)


df_recent = data.loc[start_date_5_years_ago:end_date].copy()
data = df_recent

# Q1 = data['Volume'].quantile(0.25)
# Q3 = data['Volume'].quantile(0.75)
# IQR = Q3 - Q1

# # Współczynnik dla IQR (standardowo 1.5)
# iqr_multiplier = 1.5

# lower_bound = Q1 - iqr_multiplier * IQR
# upper_bound = Q3 + iqr_multiplier * IQR

# # Tworzymy maskę Boolean, która jest True dla wartości w zakresie (nie outlierów)
# outlier_mask = (data['Volume'] < lower_bound) | (data['Volume'] > upper_bound)

# # DataFrame bez outlierów
# data = data[~outlier_mask] # '~' odwraca maskę (wybiera False, czyli nie-outliery)


dftrain = data[:'2024-11-29']
dftest = data['2024-11-29':'2025-01-02']




# columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
# data[columndate] = pd.to_datetime(data[columndate])

# data.set_index(columndate, inplace=True)

df = dftrain

predict='Adj Close'

# skew = df[predict].skew()
# if skew>1:
#     df[predict] = np.log1p(df[predict])


for lag in [1, 3, 7, 14]: # Dodatkowe lagi
    df[f'{predict}_lag_{lag}'] = df[predict].shift(lag)

# Tworzenie cech średnich kroczących dla Volume i Adj Close
for window_size in [3, 7, 14]: # Dodatkowe okna
    df[f'{predict}_roll_mean_{window_size}'] = df[predict].rolling(window=window_size).mean().shift(1)

# Dodajemy też odchylenie standardowe jako miarę zmienności
    df[f'{predict}_roll_std_{window_size}'] = df[predict].rolling(window=window_size).std().shift(1)


df['ChangePredict']=df[predict].diff().shift(1)
df['ChangePredict2']=df[predict].diff().shift(3)


df['SMA_5'] = df['Adj Close'].rolling(window=5).mean().shift(1)# SMA 10 dni


df['SMA_10'] = df['Adj Close'].rolling(window=10).mean().shift(1)
# SMA 20 dni
df['SMA_20'] = df['Adj Close'].rolling(window=20).mean().shift(1)
df['SMA_30'] = df['Adj Close'].rolling(window=30).mean().shift(1)

# 2. Wykładnicza Średnia Ruchoma (EMA)
df['EMA_5'] = df['Adj Close'].ewm(span=5, adjust=False).mean().shift(1)
# EMA 20 dni
df['EMA_10'] = df['Adj Close'].ewm(span=10, adjust=False).mean().shift(1)
# EMA 20 dni
df['EMA_20'] = df['Adj Close'].ewm(span=20, adjust=False).mean().shift(1)

#change + lag 100%
df.dropna(inplace=True)


X = df.drop(columns=[predict])
y = df[predict]



tscv = TimeSeriesSplit(n_splits=5)


X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

numeric_features = X.select_dtypes(include=np.number).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor=ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features)
    ]
)

def build_pipeline(model):
    return Pipeline(steps=[('preprocessor', preprocessor),  
                            ('regressor', model)])

results = {}
best_models_estimators = {}
results = {}
best_models_estimators = {} # Słownik do przechowywania najlepszych obiektów pipeline


# --- 1. LinearRegression ---
print("--- Tuning modelu LinearRegression ---")
param_LR = {
    'regressor__n_jobs': [None, -1]
}
linear_reg_model = LinearRegression()
linear_reg_pipeline = build_pipeline(linear_reg_model)
grid_search_LR_reg = GridSearchCV(linear_reg_pipeline, param_LR, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_LR_reg.fit(X_train, y_train)

model_name = 'LinearRegression'
y_pred_test = grid_search_LR_reg.best_estimator_.predict(X_test) # Predykcje dla testu
results[model_name] = {
    'best_params': grid_search_LR_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_LR_reg.best_score_),
    'cv_r2': grid_search_LR_reg.cv_results_['mean_test_score'][grid_search_LR_reg.best_index_] if 'r2' in grid_search_LR_reg.scoring else r2_score(y_train, grid_search_LR_reg.best_estimator_.predict(X_train)), # Możesz też użyć mean_test_score z r2, jeśli tak było scoring
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_LR_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_LR_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# --- 2. Lasso ---
print("\n--- Tuning modelu Lasso ---")
param_Lasso = {
    'regressor__alpha': [0.01, 0.1, 0.5, 1.0],
    'regressor__max_iter': [1000, 5000, 10000],
    'regressor__selection': ['cyclic', 'random']
}
lasso_model = Lasso(random_state=42)
lasso_pipeline = build_pipeline(lasso_model)
grid_search_Lasso_reg = GridSearchCV(lasso_pipeline, param_Lasso, cv=tscv, verbose=0, scoring='neg_mean_squared_error') 
grid_search_Lasso_reg.fit(X_train, y_train)

model_name = 'Lasso'
y_pred_test = grid_search_Lasso_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_Lasso_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_Lasso_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_Lasso_reg.best_estimator_.predict(X_train)), 
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_Lasso_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_Lasso_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# --- 3. SVR (Support Vector Regressor) ---
print("\n--- Tuning modelu SVR ---")
param_SVR = {
    'regressor__C': [0.1, 1, 10],
    'regressor__kernel': ['rbf', 'linear'],
    'regressor__epsilon': [0.01, 0.1, 0.5],
    'regressor__max_iter': [5000, 10000]
}
svr_model = SVR()
svr_pipeline = build_pipeline(svr_model)
grid_search_SVR_reg = GridSearchCV(svr_pipeline, param_SVR, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_SVR_reg.fit(X_train, y_train)

model_name = 'SVR'
y_pred_test = grid_search_SVR_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_SVR_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_SVR_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_SVR_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_SVR_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_SVR_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




# # --- 4. KNeighborsRegressor ---
print("\n--- Tuning modelu KNeighborsRegressor ---")
param_KNeighbors = {
    'regressor__n_neighbors': [3, 5, 7, 9, 12],
    'regressor__weights': ['uniform', 'distance'],
    'regressor__p': [1, 2]
}
kneighbors_model = KNeighborsRegressor()
kneighbors_pipeline = build_pipeline(kneighbors_model)
grid_search_KNeighbors_reg = GridSearchCV(kneighbors_pipeline, param_KNeighbors, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_KNeighbors_reg.fit(X_train, y_train)

model_name = 'KNeighbors'
y_pred_test = grid_search_KNeighbors_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_KNeighbors_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_KNeighbors_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_KNeighbors_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_KNeighbors_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_KNeighbors_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# --- 5. XGBoost (XGBRegressor) ---
print("\n--- Tuning modelu XGBoost ---")
param_XGBoost = {
    'regressor__gamma': [0, 0.1, 0.2],
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5],
    'regressor__subsample': [0.8, 1.0],
    'regressor__colsample_bytree': [0.8, 1.0],
    'regressor__n_jobs': [-1],
    'regressor__tree_method': ['hist']
}

xgboost_model = XGBRegressor(random_state=42)
xgboost_pipeline = build_pipeline(xgboost_model)
grid_search_XGBoost_reg = GridSearchCV(xgboost_pipeline, param_XGBoost, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_XGBoost_reg.fit(X_train, y_train)

model_name = 'XGBoost'
y_pred_test = grid_search_XGBoost_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_XGBoost_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_XGBoost_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_XGBoost_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_XGBoost_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_XGBoost_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




# --- 6. RandomForestRegressor ---
print("\n--- Tuning modelu RandomForestRegressor ---")
param_RandomForest = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [None, 5, 10],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__n_jobs': [-1]
}
randomforest_model = RandomForestRegressor(random_state=42)
randomforest_pipeline = build_pipeline(randomforest_model)
grid_search_RandomForest_reg = GridSearchCV(randomforest_pipeline, param_RandomForest, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_RandomForest_reg.fit(X_train, y_train)

model_name = 'RandomForest'
y_pred_test = grid_search_RandomForest_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_RandomForest_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_RandomForest_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_RandomForest_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_RandomForest_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_RandomForest_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




# --- 7. LightGBM (LGBMRegressor) ---
print("\n--- Tuning modelu LightGBM ---")
param_LGBM = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__num_leaves': [20, 31, 40]
}
lgbm_model = LGBMRegressor(random_state=42)
lgbm_pipeline = build_pipeline(lgbm_model)
grid_search_LGBM_reg = GridSearchCV(lgbm_pipeline, param_LGBM, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_LGBM_reg.fit(X_train, y_train)

model_name = 'LightGBM'
y_pred_test = grid_search_LGBM_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_LGBM_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_LGBM_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_LGBM_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_LGBM_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_LGBM_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)





# --- 8. CatBoost (CatBoostRegressor) ---
print("\n--- Tuning modelu CatBoost ---")
param_CatBoost = {
    'regressor__iterations': [100, 200, 300],
    'regressor__learning_rate': [0.03, 0.1],
    'regressor__depth': [4, 6, 8],
    'regressor__l2_leaf_reg': [1, 3, 5],
    'regressor__verbose': [0]
}
catboost_model = CatBoostRegressor(random_state=42, allow_writing_files=False)
catboost_pipeline = build_pipeline(catboost_model)
grid_search_CatBoost_reg = GridSearchCV(catboost_pipeline, param_CatBoost, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_CatBoost_reg.fit(X_train, y_train)

model_name = 'CatBoost'
y_pred_test = grid_search_CatBoost_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_CatBoost_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_CatBoost_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_CatBoost_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_CatBoost_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_CatBoost_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)





# --- 9. AdaBoost (AdaBoostRegressor) ---
print("\n--- Tuning modelu AdaBoost ---")
param_AdaBoost = {
    'regressor__n_estimators': [50, 100, 200],
    'regressor__learning_rate': [0.01, 0.1, 1.0],
    'regressor__loss': ['linear', 'square', 'exponential']
}
adaboost_model = AdaBoostRegressor(random_state=42)
adaboost_pipeline = build_pipeline(adaboost_model)
grid_search_AdaBoost_reg = GridSearchCV(adaboost_pipeline, param_AdaBoost, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_AdaBoost_reg.fit(X_train, y_train)

model_name = 'AdaBoost'
y_pred_test = grid_search_AdaBoost_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_AdaBoost_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_AdaBoost_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_AdaBoost_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_AdaBoost_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_AdaBoost_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# # --- 10. MLPRegressor (Multi-layer Perceptron Regressor) ---
print("\n--- Tuning modelu MLPRegressor ---")
param_MLP = {
    'regressor__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'regressor__activation': ['relu', 'tanh'],
    'regressor__solver': ['adam', 'sgd'],
    'regressor__alpha': [0.0001, 0.01],
    'regressor__learning_rate_init': [0.001, 0.01],
    'regressor__max_iter': [500]
}
mlp_model = MLPRegressor(random_state=42)
mlp_pipeline = build_pipeline(mlp_model)
grid_search_MLP_reg = GridSearchCV(mlp_pipeline, param_MLP, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_MLP_reg.fit(X_train, y_train)

model_name = 'MLPRegressor'
y_pred_test = grid_search_MLP_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_MLP_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_MLP_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_MLP_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_MLP_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_MLP_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




# --- 11. ElasticNet ---
print("\n--- Tuning modelu ElasticNet ---")
param_ElasticNet = {
    'regressor__alpha': [0.1, 0.5, 1.0, 0.05],
    'regressor__l1_ratio': [0.1, 0.5, 0.9],
    'regressor__tol': [0.0001, 0.1, 0.009],
    'regressor__max_iter': [400,1000, 5000,10000]
}
elasticnet_model = ElasticNet(random_state=42)
elasticnet_pipeline = build_pipeline(elasticnet_model)
grid_search_ElasticNet_reg = GridSearchCV(elasticnet_pipeline, param_ElasticNet, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_ElasticNet_reg.fit(X_train, y_train)

model_name = 'ElasticNet'
y_pred_test = grid_search_ElasticNet_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_ElasticNet_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_ElasticNet_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_ElasticNet_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_ElasticNet_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")



y_pred_test = grid_search_ElasticNet_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# # --- 12. GradientBoosting (GradientBoostingRegressor) ---
print("\n--- Tuning modelu GradientBoosting ---")
param_GradientBoosting = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [3, 5],
    'regressor__subsample': [0.8, 1.0]
}
gradientboosting_model = GradientBoostingRegressor(random_state=42)
gradientboosting_pipeline = build_pipeline(gradientboosting_model)
grid_search_GradientBoosting_reg = GridSearchCV(gradientboosting_pipeline, param_GradientBoosting, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_GradientBoosting_reg.fit(X_train, y_train)

model_name = 'GradientBoosting'
y_pred_test = grid_search_GradientBoosting_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_GradientBoosting_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_GradientBoosting_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_GradientBoosting_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_GradientBoosting_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_GradientBoosting_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




# # --- 13. HistGradientBoosting (HistGradientBoostingRegressor) ---
print("\n--- Tuning modelu HistGradientBoosting ---")
param_HistGradientBoosting = {
    'regressor__max_iter': [100, 200, 300],
    'regressor__learning_rate': [0.01, 0.1, 0.2],
    'regressor__max_depth': [None, 5, 10],
    'regressor__l2_regularization': [0.0, 0.1, 1.0]
}
histgradientboosting_model = HistGradientBoostingRegressor(random_state=42)
histgradientboosting_pipeline = build_pipeline(histgradientboosting_model)
grid_search_HistGradientBoosting_reg = GridSearchCV(histgradientboosting_pipeline, param_HistGradientBoosting, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_HistGradientBoosting_reg.fit(X_train, y_train)

model_name = 'HistGradientBoosting'
y_pred_test = grid_search_HistGradientBoosting_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_HistGradientBoosting_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_HistGradientBoosting_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_HistGradientBoosting_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_HistGradientBoosting_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_HistGradientBoosting_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# # --- 14. DecisionTree (DecisionTreeRegressor) ---
print("\n--- Tuning modelu DecisionTree ---")
param_DecisionTree = {
    'regressor__max_depth': [3, 5, 10, None],
    'regressor__min_samples_split': [2, 5, 10],
    'regressor__min_samples_leaf': [1, 2, 4]
}
decisiontree_model = DecisionTreeRegressor(random_state=42)
decisiontree_pipeline = build_pipeline(decisiontree_model)
grid_search_DecisionTree_reg = GridSearchCV(decisiontree_pipeline, param_DecisionTree, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_DecisionTree_reg.fit(X_train, y_train)

model_name = 'DecisionTree'
y_pred_test = grid_search_DecisionTree_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_DecisionTree_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_DecisionTree_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_DecisionTree_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_DecisionTree_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")


y_pred_test = grid_search_DecisionTree_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)



# # --- 15. HuberRegressor ---
print("\n--- Tuning modelu HuberRegressor ---")
param_Huber = {
    'regressor__epsilon': [1.0, 1.35, 1.5, 2.0],
    'regressor__alpha': [0.0001, 0.001, 0.01],
    'regressor__max_iter': [500, 1000, 2000]
}
huber_model = HuberRegressor()
huber_pipeline = build_pipeline(huber_model)
grid_search_Huber_reg = GridSearchCV(huber_pipeline, param_Huber, cv=tscv, verbose=0, scoring='neg_mean_squared_error')
grid_search_Huber_reg.fit(X_train, y_train)

model_name = 'HuberRegressor'
y_pred_test = grid_search_Huber_reg.best_estimator_.predict(X_test)
results[model_name] = {
    'best_params': grid_search_Huber_reg.best_params_,
    'cv_rmse': np.sqrt(-grid_search_Huber_reg.best_score_),
    'cv_r2': r2_score(y_train, grid_search_Huber_reg.best_estimator_.predict(X_train)),
    'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
    'test_r2': r2_score(y_test, y_pred_test)
}
best_models_estimators[model_name] = grid_search_Huber_reg.best_estimator_
print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
print(f"RMSE (Test): {results[model_name]['test_rmse']:.4f}, R2 (Test): {results[model_name]['test_r2']:.4f}")

y_pred_test = grid_search_Huber_reg.best_estimator_.predict(X_test)

Visu(y_test,y_pred_test,model_name)




## Podsumowanie i wybór najlepszego modelu


print("\n" + "="*50)
print("--- KOMPLETNE PODSUMOWANIE WYNIKÓW MODELI ---")
print("="*50)

# Konwersja słownika wyników na DataFrame
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index.name = 'Model'
results_df = results_df.reset_index()

# Sortowanie według RMSE na zbiorze testowym (niższe jest lepsze)
results_df_sorted = results_df.sort_values(by='test_rmse', ascending=True)

# Wyświetlanie tabeli wyników
print(results_df_sorted)
results_df_sorted.to_csv('MLscorePrice.csv')

# Wybór najlepszego modelu na podstawie RMSE na zbiorze testowym
best_model_row = results_df_sorted.iloc[0]
best_model_name = best_model_row['Model']
final_best_estimator = best_models_estimators[best_model_name]
final_best_params = results[best_model_name]['best_params']
final_best_test_rmse = results[best_model_name]['test_rmse']
final_best_test_r2 = results[best_model_name]['test_r2']

print("\n" + "="*50)
print("--- NAJLEPSZY MODEL OGÓLNIE ---")
print("="*50)
print(f"Najlepszy model to: {best_model_name}")
print(f"Jego najlepsze parametry to: {final_best_params}")
print(f"Jego RMSE na zbiorze testowym wynosi: {final_best_test_rmse:.4f}")
print(f"Jego R2 na zbiorze testowym wynosi: {final_best_test_r2:.4f}")
print("\nMożesz teraz użyć obiektu 'final_best_estimator' do przewidywań na nowych danych.")

import joblib

# ---ZAPISANIE NAJLEPSZEGO MODELU ---
model_filename = f'best_model_{best_model_name}.pkl' # Nazwa pliku z modelem
joblib.dump(final_best_estimator, model_filename)
print(f"\nNajlepszy model '{best_model_name}' został zapisany jako '{model_filename}'")
print("Możesz teraz użyć obiektu 'final_best_estimator' do przewidywań na nowych danych.")
