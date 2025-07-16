import joblib
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, HuberRegressor # Upewnij się, że masz zaimportowane
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt # Do funkcji Visu




def SaveTheBest():
    results = {}
    best_models_estimators = {}

    data = pd.read_csv('data/apple_stock.csv')
    data = data[['Unnamed: 0', 'Adj Close']]
    columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
    data[columndate] = pd.to_datetime(data[columndate])
    data.set_index(columndate, inplace=True)
    data = data[:'2024-11-29']


    df_with_features, _ = Features2(data, data.index.max() + pd.Timedelta(days=1))

  
    df_with_features = df_with_features.dropna()

    X = df_with_features.drop(columns='Adj Close')
    y = df_with_features['Adj Close'] 

    # --- Krok 2: Diagnostyka Y przed treningiem ---
    print("\n--- Diagnostyka zmiennej celu (y) przed treningiem ---")
    print(f"Typ zmiennej y: {type(y)}")
    if isinstance(y, pd.Series):
        print(f"Kształt y: {y.shape}")
        print(f"Przykładowe wartości y (pierwsze 5): {y.head().values}")
        print(f"Statystyki opisowe y:\n{y.describe()}")
        print(f"Minimum y: {y.min()}, Maksimum y: {y.max()}")
        print(f"Czy y zawiera wartości ujemne? {(y < 0).any()}")
    else:
        print("Y nie jest serią Pandas. Może to być problem.")
    print("----------------------------------------------------\n")




    X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)




    tscv = TimeSeriesSplit(n_splits=5)

    numeric_features = X.select_dtypes(include=np.number).columns

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ],
        remainder='passthrough'
    )

    def build_pipeline(model):
        return Pipeline(steps=[('preprocessor', preprocessor),
                                ('regressor', model)])

    # --- Krok 4: Tuning i trening modelu HuberRegressor ---
    print("--- Tuning modelu HuberRegressor ---")
    param_Huber = {
        'regressor__epsilon': [1.0, 1.35, 1.5, 2.0],
        'regressor__alpha': [0.0001, 0.001, 0.01],
        'regressor__max_iter': [500, 1000, 2000]
    }
    huber_model = HuberRegressor()
    huber_pipeline = build_pipeline(huber_model)

    # Sprawdź, czy X_train ma prawidłowe kolumny przed fit
    if not set(numeric_features).issubset(X_train.columns):
        print(f"Błąd: Nie wszystkie numeryczne cechy ({list(numeric_features)}) znajdują się w X_train. Dostępne: {list(X_train.columns)}")
        return results, best_models_estimators

    grid_search_Huber_reg = GridSearchCV(huber_pipeline, param_Huber, cv=tscv, verbose=0, scoring='neg_mean_squared_error')

    try:
        grid_search_Huber_reg.fit(X_train, y_train)
    except Exception as e:
        print(f"Błąd podczas treningu modelu: {e}")
        print("Sprawdź, czy dane X_train i y_train są prawidłowe i nie zawierają NaN po Features2 i podziale.")
        return results, best_models_estimators

    model_name = 'HuberRegressor'

    # 1. Zapisz najlepszy model (cały pipeline)
    BEST_MODEL_PATH = f'best_model_{model_name}.pkl'
    joblib.dump(grid_search_Huber_reg.best_estimator_, BEST_MODEL_PATH)
    print(f"\nNajlepszy model (Pipeline) dla {model_name} został zapisany do '{BEST_MODEL_PATH}'")

    # --- Krok 5: Predykcja i diagnostyka predykcji ---
    # Predykcja na zbiorze testowym
    y_pred_test_original_scale = grid_search_Huber_reg.best_estimator_.predict(X_test)

    print("\n--- Diagnostyka przewidywanych wartości (y_pred_test_original_scale) ---")
    print(f"Typ przewidywanej wartości: {type(y_pred_test_original_scale)}")
    print(f"Kształt przewidywanej wartości: {y_pred_test_original_scale.shape}")
    print(f"Przykładowe przewidywane wartości (pierwsze 5): {y_pred_test_original_scale[:5]}")
    print(f"Minimum predykcji: {np.min(y_pred_test_original_scale)}, Maksimum predykcji: {np.max(y_pred_test_original_scale)}")
    print(f"Średnia predykcji: {np.mean(y_pred_test_original_scale)}")
    print("----------------------------------------------------------------------\n")



    results[model_name] = {
        'best_params': grid_search_Huber_reg.best_params_,
        'cv_rmse': np.sqrt(-grid_search_Huber_reg.best_score_),
  
        'cv_r2': r2_score(y_train, grid_search_Huber_reg.best_estimator_.predict(X_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_original_scale)),
        'test_r2': r2_score(y_test, y_pred_test_original_scale)
    }
    best_models_estimators[model_name] = grid_search_Huber_reg.best_estimator_
    print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
    print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
    print(f"RMSE (Test - w oryginalnej skali): {results[model_name]['test_rmse']:.4f}, R2 (Test - w oryginalnej skali): {results[model_name]['test_r2']:.4f}")




    return results, best_models_estimators




def Features2(df_original, day):
    """
    Generuje cechy (features) dla DataFrame na podstawie kolumny 'Adj Close'.
    Funkcja tworzy nowe kolumny cech bezpośrednio w modyfikowanym DataFrame.

    Args:
        df_original (pd.DataFrame): Oryginalny DataFrame zawierający kolumnę 'Adj Close'
                                    z indeksem typu DatetimeIndex.
        days_lookback_value (int, optional): Wartość 'day' używana np. do określenia
                                             okna średniej kroczącej. W tej funkcji nie jest
                                             bezpośrednio używana w każdej cesze,
                                             ale może być przydatna dla innych cech.
                                             Domyślnie None.

    Returns:
        pd.DataFrame: DataFrame z dodanymi kolumnami cech.
                      Jeśli brakuje danych do obliczenia cechy, będzie tam NaN.
    """

    df = df_original.copy() # Pracujemy na kopii, aby nie modyfikować oryginalnego DataFrame'u

    topredict = 'Adj Close'

    # --- Cechy oparte na konkretnych lagach (wartość sprzed X dni) ---
    # shift() przesuwa wartości w dół. shift(1) oznacza wartość z poprzedniego dnia.
    df[f'{topredict}_lag_1'] = df[topredict].shift(1)
    df[f'{topredict}_lag_3'] = df[topredict].shift(3)
    df[f'{topredict}_lag_7'] = df[topredict].shift(7)
    df[f'{topredict}_lag_14'] = df[topredict].shift(14)

    # --- Cechy oparte na średnich kroczących (średnia z okna X dni) ---
    # .rolling(window=X).mean().shift(Y) - średnia z X dni kończąca się Y dni temu
    # shift(1) tutaj oznacza, że średnia jest liczona do poprzedniego dnia włącznie (nie bierze pod uwagę bieżącego dnia)
    df[f'{topredict}_roll_mean_3'] = df[topredict].rolling(window=3).mean().shift(1)
    df[f'{topredict}_roll_mean_7'] = df[topredict].rolling(window=7).mean().shift(1)
    df[f'{topredict}_roll_mean_14'] = df[topredict].rolling(window=14).mean().shift(1)

    df[f'{topredict}_roll_std_3'] = df[topredict].rolling(window=3).std().shift(1)
    df[f'{topredict}_roll_std_7'] = df[topredict].rolling(window=7).std().shift(1)
    df[f'{topredict}_roll_std_14'] = df[topredict].rolling(window=14).std().shift(1)

    # --- Nowe cechy: Różnice między wartościami ---
    # .diff() oblicza różnicę między bieżącym a poprzednim elementem
    # .shift(1) powoduje, że różnica jest z dnia poprzedniego (lag 1)

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
    df['ChangePredict'] = df[topredict].diff().shift(1) # Różnica Adj Close z dnia 't-1' minus 't-2'

    # Różnica z 3 dni wstecz (czyli wartość z dnia 't-3' minus 't-4')
    df['ChangePredict2'] = df[topredict].diff(periods=1).shift(3) # Różnica Adj Close z dnia 't-3' minus 't-4'

    # Możesz dodać więcej cech, np. średnie dzienne/tygodniowe, wolumen, itp.
    # Upewnij się, że Twoje 'df' wchodzi z indeksem DatetimeIndex, inaczej rolling/shift nie zadziałają poprawnie.
    df=df.dropna()


    dumy = pd.DataFrame()




    # skew = df[predict].skew()
    # if skew>1:
    #     df[predict] = np.log1p(df[predict])
    # 1. Zbieramy wszystkie wartości do jednego słownika
    single_row_data = {
    # --- Cechy oparte na konkretnych lagach (wartość sprzed X dni) ---
    f'{topredict}_lag_1': df[topredict].iloc[-1], # Ostatnia wartość z df
    f'{topredict}_lag_3': df[topredict].iloc[-3] if len(df) >= 3 else np.nan,
    f'{topredict}_lag_7': df[topredict].iloc[-7] if len(df) >= 7 else np.nan,
    f'{topredict}_lag_14': df[topredict].iloc[-14] if len(df) >= 14 else np.nan,

    # --- Cechy oparte na średnich kroczących (średnia z okna X dni, kończąca się Y dni temu) ---
    # `_roll_mean_3` dla `new_date` to średnia z ostatnich 3 wartości w `df[topredict]`
    # kończących się 3 dni przed `new_date`. Czyli średnia z wartości `.iloc[-4:-1]`.
    f'{topredict}_roll_mean_3': df[topredict].iloc[-4:-1].mean() if len(df) >= 4 else np.nan,
    f'{topredict}_roll_mean_7': df[topredict].iloc[-8:-1].mean() if len(df) >= 8 else np.nan,
    f'{topredict}_roll_mean_14': df[topredict].iloc[-15:-1].mean() if len(df) >= 15 else np.nan,

    f'{topredict}_roll_std_3': df[topredict].iloc[-4:-1].std() if len(df) >= 4 else np.nan,
    f'{topredict}_roll_std_7': df[topredict].iloc[-8:-1].std() if len(df) >= 8 else np.nan,
    f'{topredict}_roll_std_14': df[topredict].iloc[-15:-1].std() if len(df) >= 15 else np.nan,
        # --- DODANE: Simple Moving Average (SMA) dla ostatniego punktu ---
    # Obliczamy SMA dla całego ciągu i bierzemy ostatnią wartość,
    # która odpowiada SMA z poprzedniego dnia dla "bieżącego" dnia predykcji.
    'SMA_5': df['Adj Close'].rolling(window=5).mean().iloc[-1] if len(df) >= 5 else np.nan, # Dodano SMA_5
    'SMA_10': df['Adj Close'].rolling(window=10).mean().iloc[-1] if len(df) >= 10 else np.nan,
    'SMA_20': df['Adj Close'].rolling(window=20).mean().iloc[-1] if len(df) >= 20 else np.nan,
    'SMA_30': df['Adj Close'].rolling(window=30).mean().iloc[-1] if len(df) >= 30 else np.nan,

    # --- DODANE: Exponential Moving Average (EMA) dla ostatniego punktu ---
    # Obliczamy EMA dla całego ciągu i bierzemy ostatnią wartość.
    'EMA_5': df['Adj Close'].ewm(span=5, adjust=False).mean().iloc[-1] if len(df) >= 5 else np.nan,
    'EMA_10': df['Adj Close'].ewm(span=10, adjust=False).mean().iloc[-1] if len(df) >= 10 else np.nan,
    'EMA_20': df['Adj Close'].ewm(span=20, adjust=False).mean().iloc[-1] if len(df) >= 20 else np.nan,


    # --- Nowe cechy: Różnice między konkretnymi, ostatnimi wartościami ---
    # Różnica pomiędzy ostatnią a przedostatnią wartością w df
    # Jest to odpowiednik df[topredict].diff().shift(1) dla nowego rekordu
    'ChangePredict': df[topredict].iloc[-1] - df[topredict].iloc[-2] if len(df) >= 2 else np.nan,

    # Różnica pomiędzy przedostatnią a trzecią od końca wartością w df
    # Jest to odpowiednik df[topredict].diff().shift(X) dla X > 1 dla nowego rekordu
    'ChangePredict2': df[topredict].iloc[-3] - df[topredict].iloc[-4] if len(df) >= 3 else np.nan,

    }

    # 2. Tworzymy DataFrame z tego słownika.
    # Słownik staje się danymi, a indeks jest listą zawierającą naszą pojedynczą datę.
    dumy = pd.DataFrame([single_row_data], index=[day])







    return df, dumy



def Features(df_original, day):
    """
    Generuje cechy (features) dla DataFrame na podstawie kolumny 'Adj Close'.
    Funkcja tworzy nowe kolumny cech bezpośrednio w modyfikowanym DataFrame.

    Args:
        df_original (pd.DataFrame): Oryginalny DataFrame zawierający kolumnę 'Adj Close'
                                    z indeksem typu DatetimeIndex.
        days_lookback_value (int, optional): Wartość 'day' używana np. do określenia
                                             okna średniej kroczącej. W tej funkcji nie jest
                                             bezpośrednio używana w każdej cesze,
                                             ale może być przydatna dla innych cech.
                                             Domyślnie None.

    Returns:
        pd.DataFrame: DataFrame z dodanymi kolumnami cech.
                      Jeśli brakuje danych do obliczenia cechy, będzie tam NaN.
    """

    df = df_original.copy() # Pracujemy na kopii, aby nie modyfikować oryginalnego DataFrame'u

    topredict = 'Adj Close'

    # --- Cechy oparte na konkretnych lagach (wartość sprzed X dni) ---
    # shift() przesuwa wartości w dół. shift(1) oznacza wartość z poprzedniego dnia.
    df[f'{topredict}_lag_1'] = df[topredict].shift(1)
    df[f'{topredict}_lag_3'] = df[topredict].shift(3)
    df[f'{topredict}_lag_7'] = df[topredict].shift(7)
    df[f'{topredict}_lag_14'] = df[topredict].shift(14)

    # --- Cechy oparte na średnich kroczących (średnia z okna X dni) ---
    # .rolling(window=X).mean().shift(Y) - średnia z X dni kończąca się Y dni temu
    # shift(1) tutaj oznacza, że średnia jest liczona do poprzedniego dnia włącznie (nie bierze pod uwagę bieżącego dnia)
    df[f'{topredict}_roll_mean_3'] = df[topredict].rolling(window=3).mean().shift(1)
    df[f'{topredict}_roll_mean_7'] = df[topredict].rolling(window=7).mean().shift(1)
    df[f'{topredict}_roll_mean_14'] = df[topredict].rolling(window=14).mean().shift(1)

    df[f'{topredict}_roll_std_3'] = df[topredict].rolling(window=3).std().shift(1)
    df[f'{topredict}_roll_std_7'] = df[topredict].rolling(window=7).std().shift(1)
    df[f'{topredict}_roll_std_14'] = df[topredict].rolling(window=14).std().shift(1)

    # --- Nowe cechy: Różnice między wartościami ---
    # .diff() oblicza różnicę między bieżącym a poprzednim elementem
    # .shift(1) powoduje, że różnica jest z dnia poprzedniego (lag 1)

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
    df['ChangePredict'] = df[topredict].diff().shift(1) # Różnica Adj Close z dnia 't-1' minus 't-2'

    # Różnica z 3 dni wstecz (czyli wartość z dnia 't-3' minus 't-4')
    df['ChangePredict2'] = df[topredict].diff(periods=1).shift(3) # Różnica Adj Close z dnia 't-3' minus 't-4'

    # Możesz dodać więcej cech, np. średnie dzienne/tygodniowe, wolumen, itp.
    # Upewnij się, że Twoje 'df' wchodzi z indeksem DatetimeIndex, inaczej rolling/shift nie zadziałają poprawnie.
    df=df.dropna()


    df= df[:day]

    print(df)



    return df









x, model=SaveTheBest()


data = pd.read_csv('data/apple_stock.csv')
data = data[['Unnamed: 0', 'Adj Close']]
columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
data[columndate] = pd.to_datetime(data[columndate])
data.set_index(columndate, inplace=True)

end_date = data.index.max()
start_date_5_years_ago = end_date - pd.DateOffset(years=5)


df = data.loc[start_date_5_years_ago:end_date].copy()

dftrain = df[:'2024-11-29']
dftest = df.loc['2024-12-02':][:-2]
days=dftest.index

MODEL_FILE_PATH = 'best_model_HuberRegressor.pkl' # Upewnij się, że ta ścieżka jest poprawna

# --- Krok 1: Załaduj wytrenowany model Pipeline ---
loaded_model = None
if os.path.exists(MODEL_FILE_PATH):
    try:
        loaded_model = joblib.load(MODEL_FILE_PATH)
        print(f"Model '{MODEL_FILE_PATH}' został pomyślnie załadowany.")
        if not isinstance(loaded_model, Pipeline):
            print("Ostrzeżenie: Załadowany obiekt nie jest obiektem Pipeline. Upewnij się, że zapisałeś cały Pipeline.")
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        print("Upewnij się, że plik .pkl jest poprawny i nie jest uszkodzony.")





#######################

Maindata = pd.read_csv('data/apple_stock.csv')
data = Maindata[['Unnamed: 0', 'Adj Close']]
columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
data[columndate] = pd.to_datetime(data[columndate])
data.set_index(columndate, inplace=True)

dataframes={}
dataframes[0]=data
predictdata=[]

i=0
for i in range(len(days)):


    dfFeature = Features(data,days[i])

    y = dfFeature.tail(1)
    y = y.drop(columns='Adj Close')
    predicted_value = loaded_model.predict(y)


    predictdata.append(predicted_value[0]) #= pd.DataFrame({'Adj Close':predicted_value}, index=[days[i]])

    dataframes[i+1]=dfFeature


y_pred = pd.DataFrame({'Adj Close':predictdata}, index=dftest.index)






results =  {"R²": [], "MAE": [], "RMSE": []}
results['R²'].append(r2_score(dftest, y_pred))
results['MAE'].append(mean_absolute_error(dftest, y_pred))
results['RMSE'].append(np.sqrt(mean_squared_error(dftest, y_pred))) # Poprawione na MSE

dftest['Price_Up_Binary'] = dftest['Adj Close'].diff().apply(lambda x: 0 if x <= 0 else 1)

y_pred['Price_Up_Binary'] = y_pred['Adj Close'].diff().apply(lambda x: 0 if x <= 0 else 1)

final = dftest.copy()

final['yClose']=y_pred['Adj Close']
final['yPrice_Up_Binary']=y_pred['Price_Up_Binary']
final
###################

def Visu(y_test,y_pred_test,model_name):

    if isinstance(y_test, pd.Series) and isinstance(y_test.index, pd.DatetimeIndex):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test.index, y_test, label='Rzeczywiste wartości wolumenu (Test)', color='blue')
        plt.plot(y_test.index, y_pred_test, label='Przewidywane wartości wolumenu (Test)', color='red', linestyle='--')
        plt.title(f'Predykcje Ceny na modelu {model_name} na zbiorze testowym')
        plt.xlabel('Data')
        plt.ylabel('Cena')
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




Visu(dftest['Adj Close'], y_pred['Adj Close'], 'HuberRegressor')

