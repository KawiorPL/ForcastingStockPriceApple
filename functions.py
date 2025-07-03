# functions.py
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import matplotlib.pyplot as plt
import io
import base64

from sklearn.linear_model import LinearRegression
# Importy dla prognozowania
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore") # Ignoruj ostrzeżenia ARIMA/statsmodels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import pandas_ta as ta






def EDA():
    """
    Wykonuje podstawową analizę eksploracyjną danych (EDA) dla danych Apple Stock.
    Zwraca podsumowanie danych w formacie słownikowym (listy słowników)
    oraz wykresy (histogramy i box ploty) w formacie Base64,
    bez przeprowadzania logarytmizacji danych.
    """

    # Słownik do przechowywania wszystkich wyników
    results = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    results["text_logs"] = []

    results["text_logs"].append("--- Rozpoczynamy EDA dla danych Apple Stock ---")

    data = pd.read_csv('data/apple_stock.csv')

    # --- Przechowywanie head() i tail() w formacie słownikowym ---
    results["data_head"] = data.head().to_dict(orient='records')
    results["text_logs"].append("\nPierwsze 5 wierszy danych zostało przetworzone do formatu tabelarycznego (`data_head`).")

    results["data_tail"] = data.tail().to_dict(orient='records')
    results["text_logs"].append("Ostatnie 5 wierszy danych zostało przetworzone do formatu tabelarycznego (`data_tail`).")

    # --- Przechowywanie describe() w formacie słownikowym ---
    results["data_description"] = data.describe().reset_index().to_dict(orient='records')
    results["text_logs"].append("\nPodsumowanie statystyczne danych zostało przetworzone do formatu tabelarycznego (`data_description`).")

    # Przechwytywanie wyjścia data.info()
    info_buffer = io.StringIO()
    data.info(buf=info_buffer)
    results["text_logs"].append("\nInformacje o DataFrame (data.info()):\n" + info_buffer.getvalue())
    info_buffer.close()

    # Identyfikacja kolumny z datami
    if not data.empty and len(data.select_dtypes(include=['object']).columns) > 0:
        columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
        results["text_logs"].append(f"\nZidentyfikowana kolumna z datami: '{columndate}'")

        # Konwersja kolumny do datetime
        data[columndate] = pd.to_datetime(data[columndate])
        results["text_logs"].append(f"Kolumna '{columndate}' skonwertowana do formatu datetime.")

        # Sprawdzenie i usunięcie duplikatów w datach
        if data[columndate].duplicated().sum() > 0:
            results["text_logs"].append('Uwaga: Duplikaty w datach!\nUsuwanie duplikatów...')
            data = data.drop_duplicates(subset=[columndate])
            results["text_logs"].append('Duplikaty usunięte.')
        else:
            results["text_logs"].append('Brak duplikatów w datach.')

        # Ustawienie dat jako indeksów.
        data.set_index(columndate, inplace=True)
        results["text_logs"].append(f"Kolumna '{columndate}' ustawiona jako indeks.")
    else:
        results["text_logs"].append("\nOstrzeżenie: Brak kolumn typu 'object' do zidentyfikowania jako daty lub DataFrame jest pusty.")


    # Sprawdzenie czy są wartości NaN
    nan_count = data.isnull().sum().sum()
    results["text_logs"].append(f'\nSprawdzenie czy są wartości NaN: {nan_count} (suma wszystkich NaN w DataFrame).')

    # Oblicz datę początkową dla ostatnich 4 lat od najnowszej daty w danych
    end_date = data.index.max()
    start_date_4_years_ago = end_date - pd.DateOffset(years=4)
    results["text_logs"].append(f"Filtrowanie danych do ostatnich 4 lat: od {start_date_4_years_ago.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}.")

    # Filtrowanie DataFrame
    df = data.loc[start_date_4_years_ago:end_date].copy()
    results["text_logs"].append(f"DataFrame df po filtrowaniu ma {df.shape[0]} wierszy i {df.shape[1]} kolumn.")

    # Usunięcie NaN (jeśli są jakieś w oryginalnych danych po filtrowaniu)
    initial_rows_after_filter = df.shape[0]
    df.dropna(inplace=True)
    if initial_rows_after_filter - df.shape[0] > 0:
        results["text_logs"].append(f"Usunięto {initial_rows_after_filter - df.shape[0]} wierszy z NaN po filtrowaniu danych.")
        results["text_logs"].append(f"DataFrame po usunięciu NaN ma {df.shape[0]} wierszy.")
    else:
        results["text_logs"].append("Brak wartości NaN w danych po filtrowaniu.")

    # --- Usunięto sekcję logarytmizacji ---
    results["text_logs"].append("\n--- Pominięto logarytmizację kolumn zgodnie z instrukcją. ---")

    # Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns

    # --- Analiza Rozkładów: Skośność (Skewness) i Kurtoza (Kurtosis) ---
    results["text_logs"].append("\n--- Analiza Rozkładów: Skośność (Skewness) i Kurtoza (Kurtosis) ---")

    results["skewness_data"] = df[numerical_cols].skew().reset_index().rename(columns={'index': 'Feature', 0: 'Skewness'}).to_dict(orient='records')
    results["text_logs"].append("Dane o skośności wszystkich kolumn numerycznych zostały przetworzone do formatu tabelarycznego (`skewness_data`).")

    #okreslenie skośnosci
    skew = df[numerical_cols].skew()
    #indetyfikacja kolumn potrzbenych do logartytmowania. Wieksze niz 1
    log = skew[skew>1].index.values.tolist()
    #tworzenie slownika z kolumnami.
    d = {'log': log, 'date':columndate}
    results["text_logs"].append(f"Dane o skośności większej niż 1 zostaną zlogmatryzowane podczas features. Kolumna/y :{log}.")
    pd.DataFrame(d).to_csv('steps/EDA.csv')

    results["kurtosis_data"] = df[numerical_cols].kurt().reset_index().rename(columns={'index': 'Feature', 0: 'Kurtosis'}).to_dict(orient='records')
    results["text_logs"].append("Dane o kurtozie wszystkich kolumn numerycznych zostały przetworzone do formatu tabelarycznego (`kurtosis_data`).")

    results["text_logs"].append("\n--- Rekomendacja na podstawie skośności: Brak transformacji logarytmicznej.")
    results["text_logs"].append("--- Z reguły dla danych akcyjnych, Volume oraz ceny (Open, High, Low, Close, Adj Close) są często prawoskośne.")



    # --- Dodawanie wskaźników technicznych ---
    results["text_logs"].append("\n--- Dodawanie wskaźników technicznych (RSI, MACD, Stochastic) ---")

    # 1. Relative Strength Index (RSI)
    df.ta.rsi(close='Close', length=14, append=True)
    results["text_logs"].append("Dodano wskaźnik RSI (RSI_14).")


    # 3. Stochastic Oscillator (Stoch)
    df.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)
    results["text_logs"].append("Dodano wskaźnik Stochastic (STOCHk_14_3_3, STOCHd_14_3_3).")

    df.dropna(inplace=True)
    # --- Generowanie wykresów ---

    # Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns


    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy



    results["korelacja_DataTable"] = df[numerical_cols].corr().to_dict(orient='records')


# Utwórz figurę i osie dla wykresu
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8)) # Przypisz figurę do zmiennej
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr) # Użyj osi
    ax_corr.set_title('Macierz Korelacji Kolumn Numerycznych') # Ustaw tytuł na osi

    # Zapisz wykres do bufora i zakoduj do Base64
    buf_corr = io.BytesIO()
    plt.savefig(buf_corr, format='png', bbox_inches='tight')
    plt.close(fig_corr) # Zamknij obiekt figury, a nie bufor
    results["korelacja_plot_base64"] = base64.b64encode(buf_corr.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano Macierz Korelacji Kolumn Numerycznych (`korelacja_plot_base64`).")


    # Wykresy Histogramów
    results["text_logs"].append("\n--- Generowanie wykresów: Histogramy ---")
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 10))
    axes_hist = axes_hist.flatten()
    cols_to_plot_hist = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']

    for i, col in enumerate(cols_to_plot_hist):
        if i >= len(axes_hist):
            break
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes_hist[i])
            axes_hist[i].set_title(f'Histogram of {col}')
            axes_hist[i].set_xlabel(col)
            axes_hist[i].set_ylabel('Frequency')
        else:
            results["text_logs"].append(f"Ostrzeżenie: Kolumna '{col}' nie istnieje dla histogramu. Subplot zostanie pusty.")
            axes_hist[i].set_visible(False)
    plt.tight_layout()
    buf_hist = io.BytesIO()
    plt.savefig(buf_hist, format='png', bbox_inches='tight')
    plt.close(fig_hist)
    results["histogram_plot_base64"] = base64.b64encode(buf_hist.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano histogramy.")

    # Wykresy Box Plotów
    results["text_logs"].append("\n--- Generowanie wykresów: Box Ploty ---")
    fig_box, axes_box = plt.subplots(2, 3, figsize=(15, 10))
    axes_box = axes_box.flatten()
    for i, col in enumerate(cols_to_plot_hist):
        if i >= len(axes_box):
            break
        if col in df.columns:
            sns.boxplot(y=df[col], ax=axes_box[i])
            axes_box[i].set_title(f'Box Plot of {col}')
            axes_box[i].set_ylabel(col)
        else:
            results["text_logs"].append(f"Ostrzeżenie: Kolumna '{col}' nie istnieje dla box plota. Subplot zostanie pusty.")
            axes_box[i].set_visible(False)
    plt.tight_layout()
    buf_box = io.BytesIO()
    plt.savefig(buf_box, format='png', bbox_inches='tight')
    plt.close(fig_box)
    results["boxplot_plot_base64"] = base64.b64encode(buf_box.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano Box Ploty.")


    # Tworzymy figurę z wieloma subplotsami (cena + 3 wskaźniki)
    # Share x-axis to align dates
    fig_combined, (ax_prices, ax_rsi, ax_stoch) = plt.subplots(
        nrows=3, ncols=1, figsize=(18, 14), sharex=True, # Zmieniono nrows na 3, zmniejszono figsize
        gridspec_kw={'height_ratios': [3, 1, 1]} # Zmieniono height_ratios
    )

    # 1. Wykres Ceny Zamknięcia
    if 'Close' in df.columns:
        ax_prices.plot(df.index, df['Close'], label='Cena zamknięcia (Close)', linewidth=1.5, color='blue')
        ax_prices.set_title('Cena akcji Apple i wskaźniki techniczne')
        ax_prices.set_ylabel('Cena')
        ax_prices.legend(loc='upper left')
        ax_prices.grid(True, linestyle='--', alpha=0.7)
    else:
        results["text_logs"].append("Kolumna 'Close' nie znaleziona, wykres cen pominięty.")

    # 2. Wykres RSI
    if 'RSI_14' in df.columns and not df['RSI_14'].isnull().all():
        ax_rsi.plot(df.index, df['RSI_14'], label='RSI (14)', linewidth=1, color='purple')
        ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8, label='Oversold (70)')
        ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.8, label='Overbought (30)')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.legend(loc='upper left')
        ax_rsi.grid(True, linestyle='--', alpha=0.7)
        ax_rsi.set_ylim(0, 100) # Standardowe granice RSI
    else:
        results["text_logs"].append("Kolumna 'RSI_14' nie znaleziona lub zawiera tylko NaN, wykres RSI pominięty.")

    # Usunięto kod do rysowania MACD

    # 3. Wykres Stochastic Oscillator (numeracja zmieniona, bo MACD usunięto)
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns and \
       not df['STOCHk_14_3_3'].isnull().all() and \
       not df['STOCHd_14_3_3'].isnull().all():
        ax_stoch.plot(df.index, df['STOCHk_14_3_3'], label='%K Line', linewidth=1, color='blue')
        ax_stoch.plot(df.index, df['STOCHd_14_3_3'], label='%D Line', linewidth=1, color='red')
        ax_stoch.axhline(80, color='red', linestyle='--', linewidth=0.8, label='Overbought (80)')
        ax_stoch.axhline(20, color='green', linestyle='--', linewidth=0.8, label='Oversold (20)')
        ax_stoch.set_ylabel('Stochastic')
        ax_stoch.set_xlabel('Data') # Tylko ostatni wykres ma etykietę X
        ax_stoch.legend(loc='upper left')
        ax_stoch.grid(True, linestyle='--', alpha=0.7)
        ax_stoch.set_ylim(0, 100) # Standardowe granice Stochastic
    else:
        results["text_logs"].append("Kolumny Stochastic nie znaleziona lub zawiera tylko NaN, wykres Stochastic pominięty.")

    plt.tight_layout() # Dopasowanie layoutu
    buf_combined_prices = io.BytesIO()
    plt.savefig(buf_combined_prices, format='png', bbox_inches='tight')
    plt.close(fig_combined)
    results["combined_price_indicators_chart_base64"] = base64.b64encode(buf_combined_prices.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano wykres cen akcji z wskaźnikami technicznymi (bez MACD) (`combined_price_indicators_chart_base64`).")

    results["description"] = "\n".join(results["text_logs"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del results["text_logs"]

    return results




def add_features():
    """
    Wykonuje podstawową analizę eksploracyjną danych (EDA) dla danych Apple Stock.
    Zwraca podsumowanie danych w formacie słownikowym (listy słowników)
    oraz wykresy (histogramy i box ploty) w formacie Base64,
    bez przeprowadzania logarytmizacji danych.
    """
    results = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    results["text_features"] = []

    results["text_features"].append("--- Rozpoczynamy data modeling Features ---\n")

        # --- Przechowywanie head() i tail() w formacie słownikowym ---




    csv_file_path = 'data/apple_stock.csv'

    df = pd.read_csv(csv_file_path)


    eda=pd.read_csv('steps/EDA.csv')

    results["text_features"].append("Pobranie kolumny Daty przygotowane podczas EDA.\n")
    columndate = eda['date'][0]


    results["text_features"].append("Conversja kolumny daty do datetime frame\n")
    df[columndate].drop_duplicates()
    df[columndate]=pd.to_datetime(df[columndate])

    results["text_features"].append("Ustiawnie Kolumny daty jako Index\n")

    df.set_index(columndate, inplace=True)

    topredict='Adj Close'


    results["text_features"].append("Filtrowanie 5 ostanich lat.\n")
    end_date = df.index.max()
    start_date_5_years_ago = end_date - pd.DateOffset(years=5)


    df_recent = df.loc[start_date_5_years_ago:end_date].copy()

    df=df_recent

    #Indetyfyikacja kolumyn numerycznych do Standaryzacji.
    numerical_cols  = df.select_dtypes(include=['int64','float64']).columns

    pd.DataFrame({'stada':numerical_cols,'Predict':topredict}).to_csv('steps/feature.csv')

    results["text_features"].append("Ustawianie dodatkowych kolumnn: rok, miesiac, numer dnia, numer tygodnia.\n")
    df['year']=df.index.year
    df['month']=df.index.month
    df['day']=df.index.day
    df['weekday']=df.index.weekday

    results["text_features"].append("Dodawaniae sin i cos aby .\n")
    df['month_sin'] = np.sin(2 * np.pi * df['month']/ 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/ 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/ 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/ 7)


    results["text_features"].append("Dodawanie lag 1, 3 7, 14 dniowy.\n")
    for lag in [1, 3, 7, 14]:
        df[f'value_lag_{lag}'] = df[topredict].shift(lag)
        if lag != 1:
            df[f'value_roll_mean_{lag}'] = df[topredict].rolling(window=lag).mean().shift(lag)

    results["text_features"].append("Dodawnaie roznicy pomiedzy dniami w wartosci \n")
    df['ChangePredict']=df[topredict].diff()


    results["text_features"].append("Dodawaniae wskiaznikow jak RSI, MACD, STOCH \n")
    df.ta.rsi(close=topredict, length=14, append=True)

    # 2. Moving Average Convergence Divergence (MACD)
    # Standardowo MACD używa okresów 12 (szybki EMA), 26 (wolny EMA) i 9 (sygnał EMA).
    # pandas_ta doda 3 kolumny: 'MACD_12_26_9', 'MACDH_12_26_9' (histogram), 'MACDS_12_26_9' (linia sygnału)
    df.ta.macd(close=topredict, fast=12, slow=26, signal=9, append=True)

    # 3. Stochastic Oscillator (Stoch)
    # Standardowo Stochastic używa okresów 14 (K), 3 (D) i 3 (spowolnienie).
    # Wymaga kolumn 'high', 'low', 'close'.
    # pandas_ta doda 2 kolumny: 'STOCHk_14_3_3' (%K), 'STOCHd_14_3_3' (%D)
    df.ta.stoch(high='High', low='Low', predict=topredict , k=14, d=3, append=True)


    eda = pd.read_csv('steps/EDA.csv')

    results["text_features"].append("Logarytmizowanie kolumn ktore sa prawo skosne \n")
    for i in eda['log']:
        df[i] = np.log1p(df[i])


    df.dropna(inplace=True)


    results["data_featur"] = df.head(10).to_dict(orient='records')

    #Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns


    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy



    fig_corr, ax_corr = plt.subplots(figsize=(20, 16)) # Przypisz figurę do zmiennej
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr) # Użyj osi
    ax_corr.set_title('Macierz Korelacji Kolumn Numerycznych') # Ustaw tytuł na osi
    plt.tight_layout() # Dopasowanie layoutu
    # Zapisz wykres do bufora i zakoduj do Base64
    feat_corr = io.BytesIO()
    plt.savefig(feat_corr, format='png', bbox_inches='tight')
    plt.close(fig_corr) # Zamknij obiekt figury, a nie bufor
    results["korelacja_featt_base64"] = base64.b64encode(feat_corr.getvalue()).decode('utf-8')
    results["text_features"].append("Wygenerowano Macierz Korelacji Kolumn Numerycznych do podaniue features")



    results["descriptionfe"] = "\n".join(results["text_features"])


    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del results["text_features"]

    df.to_csv('steps/modelData.csv')

    return results



def Model():

    resultsM = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    resultsM["text_model"] = []

    resultsM["text_model"].append("Pobranie danych do przetworzenia przez model \n")
    df = pd.read_csv('steps/modelData.csv')

    feature = pd.read_csv('steps/feature.csv')
    resultsM["text_model"].append("Pobranie kolumny do predykcji\n")
    predict = feature['Predict'][0]


    eda=pd.read_csv('steps/EDA.csv')

    resultsM["text_model"].append("Pobranie kolumny Daty przygotowane podczas EDA.\n")
    columndate = eda['date'][0]


    resultsM["text_model"].append("Conversja kolumny daty do datetime frame\n")
    df[columndate].drop_duplicates()
    df[columndate]=pd.to_datetime(df[columndate])

    resultsM["text_model"].append("Ustiawnie Kolumny daty jako Index\n")

    df.set_index(columndate, inplace=True)


    resultsM["text_model"].append("Budowa X i y\n")
    X = df.drop(columns=predict)
    y = df[predict]




    resultsM["text_model"].append("ustawienie n split na 5\n")
    ts_split = TimeSeriesSplit(n_splits=5)

    resultsM["text_model"].append("konfiguracja modeli\n")


    models = {
        'LinearRegression': LinearRegression(),
        'Lasso (alpha: 0.1)': Lasso(alpha=0.05, random_state=42),
        'SVR': SVR( C=3, kernel='rbf', epsilon=0.1),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        #'XGBoost': XGBRegressor(max_depth=4, subsample=0.8, random_state=42),
        'RandomForest': RandomForestRegressor(max_depth=4, random_state=42)
    }




    results = {name: {"R²": [], "MAE": [], "RMSE": []} for name in models}
    predictions = {name: [] for name in models}

    # Skaler
    scaler = StandardScaler()

    # Trenowanie modeli i ewaluacja

    resultsM["text_model"].append("Trenowanie modeli i ewaluacja\n")
    # Trenowanie modeli i ewaluacja
    for name, model in models.items():
        for i, (train_index, test_index) in enumerate(ts_split.split(X)):
            # Podział na dane treningowe i testowe
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if i ==0:
                X_train_scaled=scaler.fit_transform(X_train)
            else:
                X_train_scaled=scaler.transform(X_train)

            X_test_scaled = scaler.transform(X_test)


            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            results[name]['R²'].append(r2_score(y_test, y_pred))
            results[name]['MAE'].append(mean_absolute_error(y_test, y_pred))
            results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred))) # Poprawione na MSE
            predictions[name].append((df.index[test_index], y_pred))


    # Poniżej umieść swoje rozwiązanie
    final_results = {name: {metric: np.mean(values) for metric, values in metrics.items()} for name, metrics in results.items()}

    results_df = pd.DataFrame(final_results).T
    results_df
    resultsM["text_model"].append("Wyniki\n")



    resultsM["results_models"] = results_df.reset_index().to_dict(orient='records')

    resultsM["descriptionmodel"] = "\n".join(resultsM["text_model"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del resultsM["text_model"]

    return resultsM



def EDA2(brand):
    """
    Wykonuje podstawową analizę eksploracyjną danych (EDA) dla danych Apple Stock.
    Zwraca podsumowanie danych w formacie słownikowym (listy słowników)
    oraz wykresy (histogramy i box ploty) w formacie Base64,
    bez przeprowadzania logarytmizacji danych.
    """

    # Słownik do przechowywania wszystkich wyników
    results = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    results["text_logs"] = []

    results["text_logs"].append("--- Rozpoczynamy EDA dla danych Apple Stock ---")

    data = pd.read_csv('data/World-Stock-Prices-Dataset.csv')

    data= data.sort_values('Date')

    data = data.loc[data['Brand_Name']==brand]

    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    data=data.rename(columns={'Close':'Adj Close'})

    # --- Przechowywanie head() i tail() w formacie słownikowym ---
    results["data_head2"] = data.head().to_dict(orient='records')
    results["text_logs"].append("\nPierwsze 5 wierszy danych zostało przetworzone do formatu tabelarycznego (`data_head`).")

    results["data_tail2"] = data.tail().to_dict(orient='records')
    results["text_logs"].append("Ostatnie 5 wierszy danych zostało przetworzone do formatu tabelarycznego (`data_tail`).")

    # --- Przechowywanie describe() w formacie słownikowym ---
    results["data_description2"] = data.describe().reset_index().to_dict(orient='records')
    results["text_logs"].append("\nPodsumowanie statystyczne danych zostało przetworzone do formatu tabelarycznego (`data_description`).")

    # Przechwytywanie wyjścia data.info()
    info_buffer = io.StringIO()
    data.info(buf=info_buffer)
    results["text_logs"].append("\nInformacje o DataFrame (data.info()):\n" + info_buffer.getvalue())
    info_buffer.close()

    # Identyfikacja kolumny z datami
    if not data.empty and len(data.select_dtypes(include=['object']).columns) > 0:
        columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
        results["text_logs"].append(f"\nZidentyfikowana kolumna z datami: '{columndate}'")

        # Konwersja kolumny do datetime
        data[columndate] = pd.to_datetime(data[columndate])
        results["text_logs"].append(f"Kolumna '{columndate}' skonwertowana do formatu datetime.")

        # Sprawdzenie i usunięcie duplikatów w datach
        if data[columndate].duplicated().sum() > 0:
            results["text_logs"].append('Uwaga: Duplikaty w datach!\nUsuwanie duplikatów...')
            data = data.drop_duplicates(subset=[columndate])
            results["text_logs"].append('Duplikaty usunięte.')
        else:
            results["text_logs"].append('Brak duplikatów w datach.')

        # Ustawienie dat jako indeksów.
        data.set_index(columndate, inplace=True)
        results["text_logs"].append(f"Kolumna '{columndate}' ustawiona jako indeks.")
    else:
        results["text_logs"].append("\nOstrzeżenie: Brak kolumn typu 'object' do zidentyfikowania jako daty lub DataFrame jest pusty.")


    # Sprawdzenie czy są wartości NaN
    nan_count = data.isnull().sum().sum()
    results["text_logs"].append(f'\nSprawdzenie czy są wartości NaN: {nan_count} (suma wszystkich NaN w DataFrame).')

    # Oblicz datę początkową dla ostatnich 4 lat od najnowszej daty w danych
    end_date = data.index.max()
    start_date_4_years_ago = end_date - pd.DateOffset(years=4)
    results["text_logs"].append(f"Filtrowanie danych do ostatnich 4 lat: od {start_date_4_years_ago.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}.")

    # Filtrowanie DataFrame
    df = data.loc[start_date_4_years_ago:end_date].copy()
    results["text_logs"].append(f"DataFrame df po filtrowaniu ma {df.shape[0]} wierszy i {df.shape[1]} kolumn.")

    # Usunięcie NaN (jeśli są jakieś w oryginalnych danych po filtrowaniu)
    initial_rows_after_filter = df.shape[0]
    df.dropna(inplace=True)
    if initial_rows_after_filter - df.shape[0] > 0:
        results["text_logs"].append(f"Usunięto {initial_rows_after_filter - df.shape[0]} wierszy z NaN po filtrowaniu danych.")
        results["text_logs"].append(f"DataFrame po usunięciu NaN ma {df.shape[0]} wierszy.")
    else:
        results["text_logs"].append("Brak wartości NaN w danych po filtrowaniu.")

    # --- Usunięto sekcję logarytmizacji ---
    results["text_logs"].append("\n--- Pominięto logarytmizację kolumn zgodnie z instrukcją. ---")

    # Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns

    # --- Analiza Rozkładów: Skośność (Skewness) i Kurtoza (Kurtosis) ---
    results["text_logs"].append("\n--- Analiza Rozkładów: Skośność (Skewness) i Kurtoza (Kurtosis) ---")

    results["skewness_data2"] = df[numerical_cols].skew().reset_index().rename(columns={'index': 'Feature', 0: 'Skewness'}).to_dict(orient='records')
    results["text_logs"].append("Dane o skośności wszystkich kolumn numerycznych zostały przetworzone do formatu tabelarycznego (`skewness_data`).")

    #okreslenie skośnosci
    skew = df[numerical_cols].skew()
    #indetyfikacja kolumn potrzbenych do logartytmowania. Wieksze niz 1
    log = skew[skew>1].index.values.tolist()
    #tworzenie slownika z kolumnami.
    d = {'log': log, 'date':columndate}
    results["text_logs"].append(f"Dane o skośności większej niż 1 zostaną zlogmatryzowane podczas features. Kolumna/y :{log}.")
    pd.DataFrame(d).to_csv('steps/EDA.csv')

    results["kurtosis_data2"] = df[numerical_cols].kurt().reset_index().rename(columns={'index': 'Feature', 0: 'Kurtosis'}).to_dict(orient='records')
    results["text_logs"].append("Dane o kurtozie wszystkich kolumn numerycznych zostały przetworzone do formatu tabelarycznego (`kurtosis_data`).")

    results["text_logs"].append("\n--- Rekomendacja na podstawie skośności: Brak transformacji logarytmicznej.")
    results["text_logs"].append("--- Z reguły dla danych akcyjnych, Volume oraz ceny (Open, High, Low, Close, Adj Close) są często prawoskośne.")



    # --- Dodawanie wskaźników technicznych ---
    results["text_logs"].append("\n--- Dodawanie wskaźników technicznych (RSI, MACD, Stochastic) ---")

    # 1. Relative Strength Index (RSI)
    df.ta.rsi(close='Adj Close', length=14, append=True)
    results["text_logs"].append("Dodano wskaźnik RSI (RSI_14).")


    # 3. Stochastic Oscillator (Stoch)
    df.ta.stoch(high='High', low='Low', close='Adj Close', k=14, d=3, append=True)
    results["text_logs"].append("Dodano wskaźnik Stochastic (STOCHk_14_3_3, STOCHd_14_3_3).")

    df.dropna(inplace=True)
    # --- Generowanie wykresów ---

    # Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns


    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy



    results["korelacja_DataTable2"] = df[numerical_cols].corr().to_dict(orient='records')


# Utwórz figurę i osie dla wykresu
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8)) # Przypisz figurę do zmiennej
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr) # Użyj osi
    ax_corr.set_title('Macierz Korelacji Kolumn Numerycznych') # Ustaw tytuł na osi

    # Zapisz wykres do bufora i zakoduj do Base64
    buf_corr = io.BytesIO()
    plt.savefig(buf_corr, format='png', bbox_inches='tight')
    plt.close(fig_corr) # Zamknij obiekt figury, a nie bufor
    results["korelacja2_plot_base64"] = base64.b64encode(buf_corr.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano Macierz Korelacji Kolumn Numerycznych (`korelacja_plot_base64`).")


    # Wykresy Histogramów
    results["text_logs"].append("\n--- Generowanie wykresów: Histogramy ---")
    fig_hist, axes_hist = plt.subplots(2, 3, figsize=(15, 10))
    axes_hist = axes_hist.flatten()
    cols_to_plot_hist = ['Open', 'High', 'Low', 'Volume', 'Adj Close']

    for i, col in enumerate(cols_to_plot_hist):
        if i >= len(axes_hist):
            break
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes_hist[i])
            axes_hist[i].set_title(f'Histogram of {col}')
            axes_hist[i].set_xlabel(col)
            axes_hist[i].set_ylabel('Frequency')
        else:
            results["text_logs"].append(f"Ostrzeżenie: Kolumna '{col}' nie istnieje dla histogramu. Subplot zostanie pusty.")
            axes_hist[i].set_visible(False)
    plt.tight_layout()
    buf_hist = io.BytesIO()
    plt.savefig(buf_hist, format='png', bbox_inches='tight')
    plt.close(fig_hist)
    results["histogram2_plot_base64"] = base64.b64encode(buf_hist.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano histogramy.")

    # Wykresy Box Plotów
    results["text_logs"].append("\n--- Generowanie wykresów: Box Ploty ---")
    fig_box, axes_box = plt.subplots(2, 3, figsize=(15, 10))
    axes_box = axes_box.flatten()
    for i, col in enumerate(cols_to_plot_hist):
        if i >= len(axes_box):
            break
        if col in df.columns:
            sns.boxplot(y=df[col], ax=axes_box[i])
            axes_box[i].set_title(f'Box Plot of {col}')
            axes_box[i].set_ylabel(col)
        else:
            results["text_logs"].append(f"Ostrzeżenie: Kolumna '{col}' nie istnieje dla box plota. Subplot zostanie pusty.")
            axes_box[i].set_visible(False)
    plt.tight_layout()
    buf_box = io.BytesIO()
    plt.savefig(buf_box, format='png', bbox_inches='tight')
    plt.close(fig_box)
    results["boxplot2_plot_base64"] = base64.b64encode(buf_box.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano Box Ploty.")


    # Tworzymy figurę z wieloma subplotsami (cena + 3 wskaźniki)
    # Share x-axis to align dates
    fig_combined, (ax_prices, ax_rsi, ax_stoch) = plt.subplots(
        nrows=3, ncols=1, figsize=(18, 14), sharex=True, # Zmieniono nrows na 3, zmniejszono figsize
        gridspec_kw={'height_ratios': [3, 1, 1]} # Zmieniono height_ratios
    )

    # 1. Wykres Ceny Zamknięcia
    if 'Adj Close' in df.columns:
        ax_prices.plot(df.index, df['Adj Close'], label='Cena zamknięcia (Adj Close)', linewidth=1.5, color='blue')
        ax_prices.set_title('Cena akcji Apple i wskaźniki techniczne')
        ax_prices.set_ylabel('Cena')
        ax_prices.legend(loc='upper left')
        ax_prices.grid(True, linestyle='--', alpha=0.7)
    else:
        results["text_logs"].append("Kolumna 'Adj Close' nie znaleziona, wykres cen pominięty.")

    # 2. Wykres RSI
    if 'RSI_14' in df.columns and not df['RSI_14'].isnull().all():
        ax_rsi.plot(df.index, df['RSI_14'], label='RSI (14)', linewidth=1, color='purple')
        ax_rsi.axhline(70, color='red', linestyle='--', linewidth=0.8, label='Oversold (70)')
        ax_rsi.axhline(30, color='green', linestyle='--', linewidth=0.8, label='Overbought (30)')
        ax_rsi.set_ylabel('RSI')
        ax_rsi.legend(loc='upper left')
        ax_rsi.grid(True, linestyle='--', alpha=0.7)
        ax_rsi.set_ylim(0, 100) # Standardowe granice RSI
    else:
        results["text_logs"].append("Kolumna 'RSI_14' nie znaleziona lub zawiera tylko NaN, wykres RSI pominięty.")

    # Usunięto kod do rysowania MACD

    # 3. Wykres Stochastic Oscillator (numeracja zmieniona, bo MACD usunięto)
    if 'STOCHk_14_3_3' in df.columns and 'STOCHd_14_3_3' in df.columns and \
       not df['STOCHk_14_3_3'].isnull().all() and \
       not df['STOCHd_14_3_3'].isnull().all():
        ax_stoch.plot(df.index, df['STOCHk_14_3_3'], label='%K Line', linewidth=1, color='blue')
        ax_stoch.plot(df.index, df['STOCHd_14_3_3'], label='%D Line', linewidth=1, color='red')
        ax_stoch.axhline(80, color='red', linestyle='--', linewidth=0.8, label='Overbought (80)')
        ax_stoch.axhline(20, color='green', linestyle='--', linewidth=0.8, label='Oversold (20)')
        ax_stoch.set_ylabel('Stochastic')
        ax_stoch.set_xlabel('Data') # Tylko ostatni wykres ma etykietę X
        ax_stoch.legend(loc='upper left')
        ax_stoch.grid(True, linestyle='--', alpha=0.7)
        ax_stoch.set_ylim(0, 100) # Standardowe granice Stochastic
    else:
        results["text_logs"].append("Kolumny Stochastic nie znaleziona lub zawiera tylko NaN, wykres Stochastic pominięty.")

    plt.tight_layout() # Dopasowanie layoutu
    buf_combined_prices = io.BytesIO()
    plt.savefig(buf_combined_prices, format='png', bbox_inches='tight')
    plt.close(fig_combined)
    results["combined_price_indicators2_chart_base64"] = base64.b64encode(buf_combined_prices.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano wykres cen akcji z wskaźnikami technicznymi (bez MACD) (`combined_price_indicators_chart_base64`).")

    results["description_eda"] = "\n".join(results["text_logs"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del results["text_logs"]

    return results

def add_features2(brand):
    """
    Wykonuje podstawową analizę eksploracyjną danych (EDA) dla danych Apple Stock.
    Zwraca podsumowanie danych w formacie słownikowym (listy słowników)
    oraz wykresy (histogramy i box ploty) w formacie Base64,
    bez przeprowadzania logarytmizacji danych.
    """
    results = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    results["text_features"] = []

    results["text_features"].append("--- Rozpoczynamy data modeling Features ---\n")

        # --- Przechowywanie head() i tail() w formacie słownikowym ---




    csv_file_path = 'data/World-Stock-Prices-Dataset.csv'

    df = pd.read_csv(csv_file_path)


    df = df.loc[df['Brand_Name']==brand]

    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]



    df=df.rename(columns={'Close':'Adj Close'})
    df= df.sort_values('Date')

    eda=pd.read_csv('steps/EDA.csv')

    results["text_features"].append("Pobranie kolumny Daty przygotowane podczas EDA.\n")
    columndate = eda['date'][0]


    results["text_features"].append("Conversja kolumny daty do datetime frame\n")
    df[columndate].drop_duplicates()
    df[columndate]=pd.to_datetime(df[columndate], utc=True)
    #df[columndate]=df[columndate].strftime('%Y-%m-%d')


    results["text_features"].append("Ustiawnie Kolumny daty jako Index\n")
    df.drop_duplicates(subset=[columndate], keep='last', inplace=True)

    df.set_index(columndate, inplace=True)




    topredict='Adj Close'


    results["text_features"].append("Filtrowanie 5 ostanich lat.\n")
    end_date = df.index.max()
    start_date_5_years_ago = end_date - pd.DateOffset(years=4)


    df_recent = df.loc[start_date_5_years_ago:end_date].copy()

    df=df_recent

    #Indetyfyikacja kolumyn numerycznych do Standaryzacji.
    numerical_cols  = df.select_dtypes(include=['int64','float64']).columns

    pd.DataFrame({'stada':numerical_cols,'Predict':topredict}).to_csv('steps/feature.csv')

    results["text_features"].append("Ustawianie dodatkowych kolumnn: rok, miesiac, numer dnia, numer tygodnia.\n")
    df['year']=df.index.year
    df['month']=df.index.month
    df['day']=df.index.day
    df['weekday']=df.index.weekday

    results["text_features"].append("Dodawaniae sin i cos aby .\n")
    df['month_sin'] = np.sin(2 * np.pi * df['month']/ 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/ 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday']/ 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday']/ 7)


    results["text_features"].append("Dodawanie lag 1, 3 7, 14 dniowy.\n")
    for lag in [1, 3, 7, 14]:
        df[f'value_lag_{lag}'] = df[topredict].shift(lag)
        if lag != 1:
            df[f'value_roll_mean_{lag}'] = df[topredict].rolling(window=lag).mean().shift(lag)

    results["text_features"].append("Dodawnaie roznicy pomiedzy dniami w wartosci \n")
    df['ChangePredict']=df[topredict].diff()





    results["text_features"].append("Dodawaniae wskiaznikow jak RSI, MACD, STOCH \n")
    df.ta.rsi(close=df[topredict], length=14, append=True)

    # 2. Moving Average Convergence Divergence (MACD)
    # Standardowo MACD używa okresów 12 (szybki EMA), 26 (wolny EMA) i 9 (sygnał EMA).
    # pandas_ta doda 3 kolumny: 'MACD_12_26_9', 'MACDH_12_26_9' (histogram), 'MACDS_12_26_9' (linia sygnału)
    df.ta.macd(close=topredict, fast=12, slow=26, signal=9, append=True)

    # 3. Stochastic Oscillator (Stoch)
    # Standardowo Stochastic używa okresów 14 (K), 3 (D) i 3 (spowolnienie).
    # Wymaga kolumn 'high', 'low', 'close'.
    # pandas_ta doda 2 kolumny: 'STOCHk_14_3_3' (%K), 'STOCHd_14_3_3' (%D)
    df.ta.stoch(high='High', low='Low', predict=topredict , k=14, d=3, append=True)


    eda = pd.read_csv('steps/EDA.csv')

    results["text_features"].append("Logarytmizowanie kolumn ktore sa prawo skosne \n")
    for i in eda['log']:
        df[i] = np.log1p(df[i])


    df.dropna(inplace=True)


    results["data_featur2"] = df.head(10).to_dict(orient='records')

    #Aktualizacja listy kolumn numerycznych
    numerical_cols = df.select_dtypes(include=np.number).columns


    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy



    fig_corr, ax_corr = plt.subplots(figsize=(20, 16)) # Przypisz figurę do zmiennej
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr) # Użyj osi
    ax_corr.set_title('Macierz Korelacji Kolumn Numerycznych') # Ustaw tytuł na osi
    plt.tight_layout() # Dopasowanie layoutu
    # Zapisz wykres do bufora i zakoduj do Base64
    feat_corr = io.BytesIO()
    plt.savefig(feat_corr, format='png', bbox_inches='tight')
    plt.close(fig_corr) # Zamknij obiekt figury, a nie bufor
    results["korelacja2_featt_base64"] = base64.b64encode(feat_corr.getvalue()).decode('utf-8')
    results["text_features"].append("Wygenerowano Macierz Korelacji Kolumn Numerycznych do podaniue features")



    results["description_features"] = "\n".join(results["text_features"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del results["text_features"]

    df.to_csv('steps/modelData.csv')

    return results




def Model2():

    resultsM = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    resultsM["text_model"] = []

    resultsM["text_model"].append("Pobranie danych do przetworzenia przez model \n")
    df = pd.read_csv('steps/modelData.csv')

    feature = pd.read_csv('steps/feature.csv')
    resultsM["text_model"].append("Pobranie kolumny do predykcji\n")
    predict = feature['Predict'][0]


    eda=pd.read_csv('steps/EDA.csv')

    resultsM["text_model"].append("Pobranie kolumny Daty przygotowane podczas EDA.\n")
    columndate = eda['date'][0]


    resultsM["text_model"].append("Conversja kolumny daty do datetime frame\n")
    df[columndate].drop_duplicates()
    df[columndate]=pd.to_datetime(df[columndate])

    resultsM["text_model"].append("Ustiawnie Kolumny daty jako Index\n")

    df.set_index(columndate, inplace=True)


    resultsM["text_model"].append("Budowa X i y\n")
    X = df.drop(columns=predict)
    y = df[predict]




    resultsM["text_model"].append("ustawienie n split na 5\n")
    ts_split = TimeSeriesSplit(n_splits=5)

    resultsM["text_model"].append("konfiguracja modeli\n")


    models = {
        'LinearRegression': LinearRegression(),
        'Lasso (alpha: 0.1)': Lasso(alpha=0.05, random_state=42),
        'SVR': SVR( C=3, kernel='rbf', epsilon=0.1),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5, weights='distance'),
        #'XGBoost': XGBRegressor(max_depth=4, subsample=0.8, random_state=42),
        'RandomForest': RandomForestRegressor(max_depth=4, random_state=42)
    }




    results = {name: {"R²": [], "MAE": [], "RMSE": []} for name in models}
    predictions = {name: [] for name in models}

    # Skaler
    scaler = StandardScaler()

    # Trenowanie modeli i ewaluacja

    resultsM["text_model"].append("Trenowanie modeli i ewaluacja\n")
    # Trenowanie modeli i ewaluacja
    for name, model in models.items():
        for i, (train_index, test_index) in enumerate(ts_split.split(X)):
            # Podział na dane treningowe i testowe
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            if i ==0:
                X_train_scaled=scaler.fit_transform(X_train)
            else:
                X_train_scaled=scaler.transform(X_train)

            X_test_scaled = scaler.transform(X_test)


            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            results[name]['R²'].append(r2_score(y_test, y_pred))
            results[name]['MAE'].append(mean_absolute_error(y_test, y_pred))
            results[name]['RMSE'].append(np.sqrt(mean_squared_error(y_test, y_pred))) # Poprawione na MSE
            predictions[name].append((df.index[test_index], y_pred))


    # Poniżej umieść swoje rozwiązanie
    final_results = {name: {metric: np.mean(values) for metric, values in metrics.items()} for name, metrics in results.items()}

    results_df = pd.DataFrame(final_results).T
    results_df
    resultsM["text_model"].append("Wyniki\n")



    resultsM["model_results"] = results_df.reset_index().to_dict(orient='records')

    resultsM["description_model"] = "\n".join(resultsM["text_model"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del resultsM["text_model"]

    return resultsM

    # --- 1.5. Analiza Szeregów Czasowych ---

    print("\n--- 1.5. Analiza Szeregów Czasowych: Wykresy Liniowe Cen i Wolumenu ---")

    # Wykresy dla kolumn cenowych
    plt.figure(figsize=(15, 7))
    plt.plot(df.index, df['Open'], label='Open')
    plt.plot(df.index, df['High'], label='High')
    plt.plot(df.index, df['Low'], label='Low')
    plt.plot(df.index, df['Close'], label='Close')
    plt.plot(df.index, df['Adj Close'], label='Adj Close')
    plt.title('Ceny Akcji Apple w Czasie')
    plt.xlabel('Data')
    plt.ylabel('Cena')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Wykres dla wolumenu
    plt.figure(figsize=(15, 5))
    plt.plot(df.index, df['Volume'], label='Volume', color='purple')
    plt.title('Wolumen Handlu Akcjami Apple w Czasie')
    plt.xlabel('Data')
    plt.ylabel('Wolumen')
    plt.legend()
    plt.grid(True)
    plt.show()


    # Wskazówka dotycząca stacjonarności:
    print("\n--- 1.5. Analiza Szeregów Czasowych: Sprawdzenie Stacjonarności ---")
    print("Ceny akcji (Open, High, Low, Close, Adj Close) zazwyczaj NIE SĄ stacjonarne (ich średnia i wariancja zmieniają się w czasie).")
    print("Dzienne zwroty (Daily_Return) są znacznie bliżej stacjonarności, co jest często wykorzystywane w modelach szeregów czasowych.")
    print("Dla regresji liniowej brak stacjonarności może być problemem (np. wariancja reszt nie jest stała), ale jest to często akceptowane dla prostych prognoz krótkoterminowych.")


    # --- 1.6. Analiza Korelacji ---

    print("\n--- 1.6. Analiza Korelacji ---")
    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy
    print("Macierz Korelacji:\n", correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz Korelacji Kolumn Numerycznych')
    plt.show()

    print("\n--- Wnioski z Analizy Korelacji ---")
    print("Kolumny cenowe (Open, High, Low, Close, Adj Close) są zazwyczaj silnie skorelowane ze sobą (blisko 1.0). To normalne.")
    print("Może to prowadzić do problemów z multicollinearnością w regresji liniowej, ale często 'Adj Close' jest używane jako reprezentatywna cena.")
    print("Sprawdź korelacje z kolumną 'Close' (Twoja zmienna docelowa), aby zobaczyć, które cechy mają największy wpływ.")




    print("\nSkewness for numeric columns in the filtered DataFrame:")
    print(df.select_dtypes(include=np.number).skew())

    skew = df.select_dtypes(include=np.number).skew()

    log = skew[skew>1].index.values.tolist()
    d = {'log': log}
    pd.DataFrame(d).to_csv('steps/EDA.csv')


    plt.figure(figsize=(10, 8))
    sns.jointplot(data=df, x='High', y='Open')
    sns.jointplot(data=df, x='Volume', y='Close')
    sns.jointplot(data=df, x=df.index, y='Close')
    sns.jointplot(data=df, x=df.index, y='Volume')
    plt.show()


    # 1. Relative Strength Index (RSI)
    # Standardowo RSI obliczane jest dla okresu 14 dni.
    # pandas_ta automatycznie doda kolumnę 'RSI_14'
    df.ta.rsi(close='Close', length=14, append=True)

    # 2. Moving Average Convergence Divergence (MACD)
    # Standardowo MACD używa okresów 12 (szybki EMA), 26 (wolny EMA) i 9 (sygnał EMA).
    # pandas_ta doda 3 kolumny: 'MACD_12_26_9', 'MACDH_12_26_9' (histogram), 'MACDS_12_26_9' (linia sygnału)
    df.ta.macd(close='Close', fast=12, slow=26, signal=9, append=True)

    # 3. Stochastic Oscillator (Stoch)
    # Standardowo Stochastic używa okresów 14 (K), 3 (D) i 3 (spowolnienie).
    # Wymaga kolumn 'high', 'low', 'close'.
    # pandas_ta doda 2 kolumny: 'STOCHk_14_3_3' (%K), 'STOCHd_14_3_3' (%D)
    df.ta.stoch(high='High', low='Low', close='Close', k=14, d=3, append=True)

    df.dropna(inplace=True)
    print("\nDataFrame po dodaniu wskaźników technicznych:")
    print(df.tail(10)) # Pokaż ostatnie 10 wierszy, aby zobaczyć wskaźniki
    print(f"\nLiczba kolumn po: {df.shape[1]}")

    # Sprawdź, ile jest brakujących wartości po dodaniu wskaźników
    # Zazwyczaj pierwsze wiersze będą miały NaN, ponieważ wskaźniki wymagają danych historycznych
    print("\nBrakujące wartości po dodaniu wskaźników:")
    print(df.isnull().sum())



    # --- Tworzenie figury i siatki podwykresów ---
    # Mamy teraz 4 podwykresy: Ceny, RSI, MACD, Stochastic.
    # Dostosowujemy proporcje wysokości, aby ceny zajmowały więcej miejsca.
    fig, (ax_prices, ax_rsi, ax_macd, ax_stoch) = plt.subplots(4, 1, figsize=(15, 18), # Zwiększ wysokość dla 4 wykresów
                                                            sharex=True,
                                                            gridspec_kw={'height_ratios': [3, 1, 1, 1]})


    # --- Podwykres 1: Ceny Akcji (bezpośrednio na ax_prices) ---
    ax_prices.plot(df.index, df['Open'], label='Open', linewidth=1)
    ax_prices.plot(df.index, df['High'], label='High', linewidth=1)
    ax_prices.plot(df.index, df['Low'], label='Low', linewidth=1)
    ax_prices.plot(df.index, df['Close'], label='Close', linewidth=2, color='blue') # Cena zamknięcia wyraźniej

    ax_prices.set_title('Ceny Akcji, RSI, MACD i Stochastic Oscillator w Czasie') # Tytuł główny na górnym wykresie
    ax_prices.set_ylabel('Cena')
    ax_prices.legend(loc='upper left')
    ax_prices.grid(True)


    # --- Podwykres 2: RSI (na osobnym podwykresie ax_rsi) ---
    ax_rsi.plot(df.index, df['RSI_14'], label='RSI (14)', color='red', linewidth=1.5)
    ax_rsi.axhline(70, color='gray', linestyle=':', linewidth=1, label='RSI Overbought (70)')
    ax_rsi.axhline(30, color='gray', linestyle=':', linewidth=1, label='RSI Oversold (30)')
    ax_rsi.set_ylabel('RSI Value')
    ax_rsi.set_title('RSI (Relative Strength Index)') # Tytuł dla podwykresu RSI
    ax_rsi.legend(loc='upper left')
    ax_rsi.grid(True)
    ax_rsi.set_ylim(0, 100) # RSI jest zawsze między 0 a 100


    # --- Podwykres 3: MACD (na osobnym podwykresie ax_macd) ---
    ax_macd.plot(df.index, df['MACD_12_26_9'], label='MACD Line', color='blue', linewidth=1.5)
    ax_macd.plot(df.index, df['MACDS_12_26_9'], label='Signal Line', color='red', linestyle='--', linewidth=1.5)
    ax_macd.bar(df.index, df['MACDH_12_26_9'], color=np.where(df['MACDH_12_26_9'] > 0, 'green', 'red'), label='MACD Histogram', alpha=0.6)
    ax_macd.set_ylabel('MACD Value')
    ax_macd.set_title('MACD (Moving Average Convergence Divergence)') # Tytuł dla podwykresu MACD
    ax_macd.legend(loc='upper left')
    ax_macd.grid(True)
    ax_macd.axhline(0, color='gray', linestyle='-', linewidth=0.8) # Linia zerowa dla MACD


    # --- Podwykres 4: Stochastic Oscillator (na osobnym podwykresie ax_stoch) ---
    ax_stoch.plot(df.index, df['STOCHk_14_3_3'], label='%K Line', color='purple', linewidth=1.5)
    ax_stoch.plot(df.index, df['STOCHd_14_3_3'], label='%D Line', color='orange', linestyle='--', linewidth=1.5)
    ax_stoch.axhline(80, color='gray', linestyle=':', linewidth=1, label='Overbought (80)')
    ax_stoch.axhline(20, color='gray', linestyle=':', linewidth=1, label='Oversold (20)')
    ax_stoch.set_xlabel('Data') # Etykieta X tylko na ostatnim wykresie
    ax_stoch.set_ylabel('Stoch Value')
    ax_stoch.set_title('Stochastic Oscillator') # Tytuł dla podwykresu Stochastic
    ax_stoch.legend(loc='upper left')
    ax_stoch.grid(True)
    ax_stoch.set_ylim(0, 100) # Stochastic jest zawsze między 0 a 100

    plt.tight_layout() # Dopasuj układ, aby uniknąć nakładania się elementów
    plt.show()


    numerical_cols  = df.select_dtypes(include=['int64','float64']).columns
    print("\n--- 1.6. Analiza Korelacji ---")
    # Wybierz tylko kolumny numeryczne do obliczenia korelacji
    correlation_matrix = df[numerical_cols].corr() # Dodajemy Daily_Return jeśli stworzyliśmy
    print("Macierz Korelacji:\n", correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Macierz Korelacji Kolumn Numerycznych')
    plt.show()









