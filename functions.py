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

    # Oblicz datę początkową dla ostatnich 5 lat od najnowszej daty w danych
    end_date = data.index.max()
    start_date_4_years_ago = end_date - pd.DateOffset(years=5)
    results["text_logs"].append(f"Filtrowanie danych do ostatnich 5 lat: od {start_date_4_years_ago.strftime('%Y-%m-%d')} do {end_date.strftime('%Y-%m-%d')}.")

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
    results["text_logs"].append("Wygenerowano wykres cen akcji z wskaźnikami technicznymi (bez MACD)")


    df['Month'] = df.index.month
    df['Year'] = df.index.year
    average_price_per_month = df.groupby('Month')['Adj Close'].mean()

    month_names = {
        1: 'Styczeń', 2: 'Luty', 3: 'Marzec', 4: 'Kwiecień',
        5: 'Maj', 6: 'Czerwiec', 7: 'Lipiec', 8: 'Sierpień',
        9: 'Wrzesień', 10: 'Październik', 11: 'Listopad', 12: 'Grudzień'
    }
    average_price_per_month.index = average_price_per_month.index.map(month_names)

    fig1, ax1 = plt.subplots(figsize=(12, 6)) 
    sns.barplot(x=average_price_per_month.index, y=average_price_per_month.values, palette='viridis', ax=ax1)
    ax1.set_title('Średnia Cena "Adj Close" w Poszczególnych Miesiącach (Uśredniona przez Lata)')
    ax1.set_xlabel('Miesiąc')
    ax1.set_ylabel('Średnia Cena "Adj Close"')
    ax1.set_xticklabels(average_price_per_month.index, rotation=45, ha='right')
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buf1 = io.BytesIO()
    fig1.savefig(buf1, format='png', bbox_inches='tight')
    plt.close(fig1) # Ważne, aby zamknąć figurę, aby zwolnić pamięć
    results["avg_monthly_price_plot_base64"] = base64.b64encode(buf1.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano wykres cen akcji w poszczególnych miesiącach (Uśredniona przez Lata)")



    december_data = df[df['Month'] == 12]
    average_price_december_per_year = december_data.groupby('Year')['Adj Close'].mean()

    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.lineplot(x=average_price_december_per_year.index, y=average_price_december_per_year.values, marker='o', color='red', ax=ax2)
    ax2.set_title('Średnia Cena "Adj Close" w Grudniu na Przestrzeni Lat')
    ax2.set_xlabel('Rok')
    ax2.set_ylabel('Średnia Cena "Adj Close"')
    ax2.set_xticks(average_price_december_per_year.index) # Ustawia wszystkie lata jako etykiety
    ax2.set_xticklabels(average_price_december_per_year.index, rotation=45, ha='right')
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buf2 = io.BytesIO()
    fig2.savefig(buf2, format='png', bbox_inches='tight')
    plt.close(fig2)
    results["avg_december_price_plot_base64"] = base64.b64encode(buf2.getvalue()).decode('utf-8')
    results["text_logs"].append("Wygenerowano wykres Średnia Cena Adj Close w Grudniu na Przestrzeni Lat")


    last_prices = {'Jan-Nov': {}, 'Dec': {}}

    for year in df['Year'].unique():
        df_year = df[df['Year'] == year]

        jan_nov_data = df_year[(df_year['Month'] >= 1) & (df_year['Month'] <= 11)]
        if not jan_nov_data.empty:
            last_prices['Jan-Nov'][year] = jan_nov_data['Adj Close'].iloc[-1]

        dec_data = df_year[df_year['Month'] == 12]
        if not dec_data.empty:
            last_prices['Dec'][year] = dec_data['Adj Close'].iloc[-1]

    percent_changes = {'Jan-Nov': [], 'Dec': []}
    years = sorted(df['Year'].unique())

    for i in range(1, len(years)):
        current_year = years[i]
        previous_year = years[i-1]

        if previous_year in last_prices['Jan-Nov'] and current_year in last_prices['Jan-Nov']:
            prev_price = last_prices['Jan-Nov'][previous_year]
            curr_price = last_prices['Jan-Nov'][current_year]
            if prev_price != 0:
                change = ((curr_price - prev_price) / prev_price) * 100
                percent_changes['Jan-Nov'].append({'Year': current_year, 'Change': change})

        if previous_year in last_prices['Dec'] and current_year in last_prices['Dec']:
            prev_price = last_prices['Dec'][previous_year]
            curr_price = last_prices['Dec'][current_year]
            if prev_price != 0:
                change = ((curr_price - prev_price) / prev_price) * 100
                percent_changes['Dec'].append({'Year': current_year, 'Change': change})

    df_jan_nov_change = pd.DataFrame(percent_changes['Jan-Nov'])
    df_dec_change = pd.DataFrame(percent_changes['Dec'])

    # Wykres dla Styczeń-Listopad
    if not df_jan_nov_change.empty:
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        sns.barplot(x='Year', y='Change', data=df_jan_nov_change, palette='coolwarm', ax=ax3)
        ax3.set_title('Procentowa Zmiana Wartości Spółki Rok do Roku (Styczeń-Listopad)')
        ax3.set_xlabel('Rok')
        ax3.set_ylabel('Procentowa Zmiana (%)')
        ax3.axhline(0, color='grey', linewidth=0.8)
        ax3.set_xticklabels(df_jan_nov_change['Year'], rotation=45, ha='right')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        buf3 = io.BytesIO()
        fig3.savefig(buf3, format='png', bbox_inches='tight')
        plt.close(fig3)
        results["percent_change_jan_nov_plot_base64"] = base64.b64encode(buf3.getvalue()).decode('utf-8')
        results["text_logs"].append("Wygenerowano wykres Procentowa Zmiana Wartości Spółki Rok do Roku (Styczeń-Listopad)")
    else:
        results["percent_change_jan_nov_plot_base64"] = "Brak wystarczających danych do obliczenia zmian rok do roku dla okresu Styczeń-Listopad."

    # Wykres dla Grudnia
    if not df_dec_change.empty:
        fig4, ax4 = plt.subplots(figsize=(14, 7))
        sns.barplot(x='Year', y='Change', data=df_dec_change, palette='plasma', ax=ax4)
        ax4.set_title('Procentowa Zmiana Wartości Spółki Rok do Roku (Grudzień)')
        ax4.set_xlabel('Rok')
        ax4.set_ylabel('Procentowa Zmiana (%)')
        ax4.axhline(0, color='grey', linewidth=0.8)
        ax4.set_xticklabels(df_dec_change['Year'], rotation=45, ha='right')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        buf4 = io.BytesIO()
        fig4.savefig(buf4, format='png', bbox_inches='tight')
        plt.close(fig4)
        results["percent_change_dec_plot_base64"] = base64.b64encode(buf4.getvalue()).decode('utf-8')
        results["text_logs"].append("Wygenerowano wykres Procentowa Zmiana Wartości Spółki Rok do Roku (Grudzień)")
    else:
        results["percent_change_dec_plot_base64"] = "Brak wystarczających danych do obliczenia zmian rok do roku dla Grudnia."





    results["description"] = "\n".join(results["text_logs"])

    # Usunięcie "text_logs" jeśli nie chcemy go jako oddzielnego klucza w finalnym JSON-ie
    del results["text_logs"]

    return results



def SaveTheBest():
    results = {}
    best_models_estimators = {}

    data = pd.read_csv('data/apple_stock.csv')
    data = data[['Unnamed: 0', 'Adj Close']]
    columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
    data[columndate] = pd.to_datetime(data[columndate])
    data.set_index(columndate, inplace=True)
    data = data[:'2024-11-29']

    # --- Krok 1: Generowanie cech i etykiet (X, y) ---
    # To jest kluczowe! Upewnij się, że 'Adj Close' w 'data' nie jest modyfikowane PRZED tym krokiem
    # i że Features2 zwraca oczekiwane wartości.
    df_with_features, _ = Features2(data, data.index.max() + pd.Timedelta(days=1))

    # Usuń wiersze z NaN, które Features2 mogło wygenerować na początku
    df_with_features = df_with_features.dropna()

    X = df_with_features.drop(columns='Adj Close')
    y = df_with_features['Adj Close'] # Y POWINNO BYĆ TUTAJ W ORYGINALNEJ SKALI CEN AKCJI

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

    # Zgodnie z założeniami, że y_train nie było skalowane,
    # y_pred_test_original_scale JUŻ POWINNO BYĆ w oryginalnej skali.
    # Brak tutaj inverse_transform, chyba że diagnostyka pokaże inaczej.

    results[model_name] = {
        'best_params': grid_search_Huber_reg.best_params_,
        'cv_rmse': np.sqrt(-grid_search_Huber_reg.best_score_),
        # Ważne: r2_score wymaga, aby obie wartości były w tej samej skali
        'cv_r2': r2_score(y_train, grid_search_Huber_reg.best_estimator_.predict(X_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test_original_scale)),
        'test_r2': r2_score(y_test, y_pred_test_original_scale)
    }
    best_models_estimators[model_name] = grid_search_Huber_reg.best_estimator_
    print(f"Najlepsze parametry dla {model_name}: {results[model_name]['best_params']}")
    print(f"RMSE (CV): {results[model_name]['cv_rmse']:.4f}, R2 (CV): {results[model_name]['cv_r2']:.4f}")
    print(f"RMSE (Test - w oryginalnej skali): {results[model_name]['test_rmse']:.4f}, R2 (Test - w oryginalnej skali): {results[model_name]['test_r2']:.4f}")

    # --- Krok 6: Wizualizacja ---
    # Wizualizacja na wartościach w oryginalnej skali


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





def Model():

    resultsM = {}

    # Lista do zbierania ogólnych komunikatów tekstowych (nie tabelarycznych)
    resultsM["text_model"] = []


    



    #x, model=SaveTheBest()




    resultsM["text_model"].append("Pobranie danych do przetworzenia przez model. \n")
    Maindata = pd.read_csv('data/apple_stock.csv')

    
    resultsM["text_model"].append("Pobranie kolumny Adj Close do predykcji.\n")
    data = Maindata[['Unnamed: 0', 'Adj Close']]




    resultsM["text_model"].append("Konwersja kolumny daty na typ datetime i ustawienie jako index.\n")
    
    columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
    data[columndate] = pd.to_datetime(data[columndate])
    data.set_index(columndate, inplace=True)

    resultsM["text_model"].append("Pobieranie dat do predykcji ceny - Grudzień 2024\n")
    
    dftest = data.loc['2024-12-02':][:-2]
    days=dftest.index

    resultsM["text_model"].append("Budowa slownika, ktory jest zbiorem wszystkich predykcji na grudzień\n")
    dataframes={}
    dataframes[0]=data
    resultsM["text_model"].append("Stworznie listy z prognozowanymi cenami\n")
    predictdata=[]




    resultsM["text_model"].append("Pobieranie najlepszego modelu z pliku best_model_HuberRegressor.pkl.\n")
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








    
    for i in range(len(days)):


        dfFeature = Features(data,days[i])

        y = dfFeature.tail(1)
        y = y.drop(columns='Adj Close')
        predicted_value = loaded_model.predict(y)


        predictdata.append(predicted_value[0])

        dataframes[i+1]=dfFeature

    resultsM["text_model"].append("Utworznie pentli na forcasting Ceny na grudzień.\n")
    y_pred = pd.DataFrame({'Adj Close':predictdata}, index=dftest.index)
    resultsM["text_model"].append("Tworznie DataFrame z wartościami przewidywabynu.\n")

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
            plt.savefig(f"static/{plot_filename_ts}")
            #print(f"Wykres trendu czasowego Linear Regression został zapisany jako '{plot_filename_ts}'")
            #plt.show()
        else:
            print("\nOstrzeżenie: Nie można zwizualizować predykcji w funkcji czasu, ponieważ y_test nie ma indeksu datowego.")
            print("Upewnij się, że y_test jest serią Pandas z indeksem typu DatetimeIndex.")



    resultsM["text_model"].append("Tworzenie wizualizacji.\n")
    Visu(dftest['Adj Close'], y_pred['Adj Close'], 'HuberRegressor')

    resultsM["text_model"].append("Tworzenie metryk\n")
    results =  {"R²": [], "MAE": [], "RMSE": []}
    results['R²'].append(r2_score(dftest, y_pred))
    results['MAE'].append(mean_absolute_error(dftest, y_pred))
    results['RMSE'].append(np.sqrt(mean_squared_error(dftest, y_pred))) # Poprawione na MSE

    dftest['Price_Up_Binary'] = dftest['Adj Close'].diff().apply(lambda x: 0 if x <= 0 else 1)

    y_pred['Price_Up_Binary'] = y_pred['Adj Close'].diff().apply(lambda x: 0 if x <= 0 else 1)

    final = dftest.copy()
    resultsM["text_model"].append("Dodanie do orginalnych danych wartosci przewidywane.\n")
    final['yClose']=y_pred['Adj Close']
    final['yPrice_Up_Binary']=y_pred['Price_Up_Binary']
    resultsM["text_model"].append("yClose to przewidzyania wartosc Adj Close, a yPrice_Up_Binary to przewidziany wzrost lub spadek wartosci.\n")
    df_bez_indeksu = final.reset_index(drop=True)

    resultsM["final_data"] = df_bez_indeksu.to_dict(orient='records')
        

    results_df = pd.DataFrame(results)
    results_df
    resultsM["text_model"].append("Wyniki\n")



    resultsM["results_models"] = results_df.reset_index().to_dict(orient='records')

    resultsM["descriptionmodel"] = "\n".join(resultsM["text_model"])

    
    del resultsM["text_model"]

    return resultsM




