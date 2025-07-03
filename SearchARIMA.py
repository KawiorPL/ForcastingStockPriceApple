import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import warnings
import itertools
import joblib # Importuj joblib do zapisywania i ładowania modeli

# Import pmdarima do automatycznego strojenia ARIMA
import pmdarima as pm

# Ignoruj ostrzeżenia, które czasami pojawiają się przy modelach ARIMA
warnings.filterwarnings("ignore")

# --- Definicja funkcji Visu ---
def Visu(y_test_series, y_pred_test_series, model_name):
    """
    Wizualizuje rzeczywiste i przewidywane wartości na wykresie punktowym
    oraz, jeśli dane mają indeks datowy, na wykresie szeregu czasowego.
    Zapisuje wykresy do plików PNG.

    Args:
        y_test_series (pd.Series): Rzeczywiste wartości testowe (z indeksem datowym).
        y_pred_test_series (pd.Series): Przewidywane wartości testowe (z indeksem datowym).
        model_name (str): Nazwa modelu do użycia w tytułach wykresów i nazwach plików.
    """
    # Wykres punktowy (Rzeczywiste vs. Przewidywane)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_series, y_pred_test_series, alpha=0.6, color='blue', label='Przewidywane vs Rzeczywiste (Test)')

    # Upewnij się, że min/max są używane z wartościami, które mają sens dla obu serii
    min_val = min(y_test_series.min(), y_pred_test_series.min())
    max_val = max(y_test_series.max(), y_pred_test_series.max())

    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Idealna Predykcja (y=x)')
    plt.title(f'Rzeczywiste vs Przewidywane wartości dla {model_name}')
    plt.xlabel('Rzeczywiste wartości')
    plt.ylabel('Przewidywane wartości')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    scatter_filename = f'{model_name}_scatter_predictions_test_set.png'
    plt.savefig(scatter_filename)
    print(f"Wykres punktowy dla {model_name} został zapisany jako '{scatter_filename}'")
    plt.show()

    # Wykres szeregu czasowego (tylko jeśli indeks jest datowy)
    if isinstance(y_test_series.index, pd.DatetimeIndex):
        plt.figure(figsize=(15, 6))
        plt.plot(y_test_series.index, y_test_series, label='Rzeczywiste wartości wolumenu (Test)', color='blue')
        plt.plot(y_pred_test_series.index, y_pred_test_series, label='Przewidywane wartości wolumenu (Test)', color='red', linestyle='--')
        plt.title(f'Trend Wolumenu i Predykcje {model_name} na zbiorze testowym')
        plt.xlabel('Data')
        plt.ylabel('Wolumen')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        timeseries_filename = f'{model_name}_timeseries_predictions_test_set.png'
        plt.savefig(timeseries_filename)
        print(f"Wykres trendu czasowego dla {model_name} został zapisany jako '{timeseries_filename}'")
        plt.show()
    else:
        print(f"\nOstrzeżenie: Nie można zwizualizować predykcji w funkcji czasu dla {model_name}, ponieważ indeks nie jest datowy.")
        print("Upewnij się, że indeks serii wejściowych jest typu DatetimeIndex.")

# --- 1. Ładowanie i wstępne przygotowanie danych ---
try:
    data = pd.read_csv('data/apple_stock.csv')
except FileNotFoundError:
    print("Błąd: Plik 'data/apple_stock.csv' nie został znaleziony.")
    print("Upewnij się, że plik znajduje się w katalogu 'data/' lub podaj pełną ścieżkę.")
    exit()

data = data[['Unnamed: 0', 'Volume']]
columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
data[columndate] = pd.to_datetime(data[columndate])
data.set_index(columndate, inplace=True)
data.sort_index(inplace=True) # Zawsze sortuj indeks czasowy!

# Odkomentuj, jeśli chcesz filtrować do ostatnich 4-5 lat
# # --- 2. Filtrowanie danych do ostatnich 5 lat ---
# end_date = data.index.max()
# start_date_5_years_ago = end_date - pd.DateOffset(years=4)
# df_recent = data.loc[start_date_5_years_ago:end_date].copy()
# data = df_recent

print(f"Dane od: {data.index.min().date()} do {data.index.max().date()}")
print(f"Liczba obserwacji: {len(data)}")
print("-" * 30)

# --- 3. Usuwanie outlierów z 'Volume' ---
Q1 = data['Volume'].quantile(0.25)
Q3 = data['Volume'].quantile(0.75)
IQR = Q3 - Q1
iqr_multiplier = 1.5

lower_bound = Q1 - iqr_multiplier * IQR
upper_bound = Q3 + iqr_multiplier * IQR

outlier_mask = (data['Volume'] < lower_bound) | (data['Volume'] > upper_bound)
data = data[~outlier_mask].copy()

print(f"Liczba obserwacji po usunięciu outlierów: {len(data)}")
print("-" * 30)

# Upewnij się, że 'data' ma wartości do pracy
if data.empty:
    print("Błąd: Brak danych po preprocessingu do uruchomienia ARIMA.")
    exit()

# Przygotowanie danych do walidacji krokowej
X = data['Volume'].values

# --- 4. Automatyczne strojenie hiperparametrów ARIMA za pomocą pmdarima ---
print("Rozpoczynam automatyczne strojenie hiperparametrów ARIMA za pomocą pmdarima...")

best_model_auto = pm.auto_arima(X,
                                start_p=1, start_q=1,
                                max_p=5, max_q=5, # Zwiększone zakresy do przeszukiwania
                                d=None, # Pozwól auto_arima znaleźć optymalne d (lub ustaw na stałe np. d=1)
                                max_d=2, # Maksymalna wartość d, jeśli d=None
                                seasonal=False, # Ustaw na False dla ARIMA, True dla SARIMA
                                stepwise=True,
                                trace=True,
                                suppress_warnings=True,
                                error_action="ignore",
                                approximation=False, # Użyj dokładnej metody, wolniej, ale dokładniej
                                n_fits=50 # Ile modeli próbować (stepwise zwykle nie dobija do max_order)
                               )

best_order = best_model_auto.order # To zwróci (p, d, q) dla najlepszego modelu

print("-" * 30)
print(f"\nNajlepszy Order ARIMA (z pmdarima): {best_order}")
print(f"Najlepsze kryterium informacyjne (AIC): {best_model_auto.aic():.3f}")
print("-" * 30)

# --- ZAPIS MODELU ARIMA (pmdarima) ---
model_filename = 'best_arima_model.joblib'
joblib.dump(best_model_auto, model_filename)
print(f"Model pmdarima.auto_arima został zapisany jako '{model_filename}'")
# Aby wczytać model: loaded_model = joblib.load('best_arima_model.joblib')

# --- 5. Uruchomienie modelu z najlepszymi hiperparametrami na danych testowych (Walk-Forward Validation) ---
print(f"Uruchamiam walk-forward validation z najlepszymi parametrami: {best_order}")

# Przygotuj dane ponownie dla najlepszego modelu
X = data['Volume'].values
size = int(len(X) * 0.8) # Podział na zbiór treningowy i testowy do oceny
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions_best_model = list()

for t in range(len(test)):
    model = ARIMA(history, order=best_order)
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions_best_model.append(yhat)
    obs = test[t]
    history.append(obs)

# Ocena prognoz dla najlepszego modelu
rmse_best = sqrt(mean_squared_error(test, predictions_best_model))
r2_best = r2_score(test, predictions_best_model)
print(f'Test RMSE (najlepszy model): {rmse_best:.3f}')
print(f'Test R2 (najlepszy model): {r2_best:.3f}')

# --- PRZYGOTOWANIE DANYCH DLA FUNKCJI VISU ---
# y_test_series to rzeczywiste wartości z testu, z indeksem datowym
test_dates = data.index[size:len(X)]
y_test_series = pd.Series(test, index=test_dates)

# y_pred_test_series to przewidywane wartości z testu, również z indeksem datowym
y_pred_test_series = pd.Series(predictions_best_model, index=test_dates)

# --- WYWOŁANIE FUNKCJI VISU DLA OCENY MODELU ARIMA NA ZBIORZE TESTOWYM ---
Visu(y_test_series, y_pred_test_series, 'ARIMA_Test_Evaluation')


# --- 7. Prognozowanie wolumenu '2024-12-01' do '2025-01-02' przy użyciu najlepszego modelu ---
print("\n--- Prognozowanie przyszłego wolumenu (2024-12-01 do 2025-01-02) przy użyciu najlepszego modelu ---")

# Trening modelu na CAŁYCH dostępnych danych, aby uzyskać najlepszą prognozę na przyszłość
final_training_data_for_forecast = data['Volume'].copy() # Całe dane po preprocessingu

# Użyjemy obiektu 'best_model_auto', który już został dopasowany do całego 'X' w kroku 4.
# Jest to bardziej efektywne niż ponowne tworzenie i dopasowywanie ARIMA z statsmodels.
model_fit_final_forecast_pm = best_model_auto

forecast_start_date = pd.to_datetime('2024-12-01')
forecast_end_date = pd.to_datetime('2025-01-02')

# Wygeneruj zakres dat dla prognozy, uwzględniając tylko dni robocze
future_dates_range = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='B')

if future_dates_range.empty:
    print("Błąd: Zakres dat do prognozy jest pusty po zastosowaniu częstotliwości dni roboczych.")
    forecasted_volume_future = pd.Series()
else:
    num_steps_to_forecast = len(future_dates_range)

    try:
        # Prognozowanie bezpośrednio z obiektu pmdarima.ARIMA
        forecasted_volume_future = model_fit_final_forecast_pm.predict(n_periods=num_steps_to_forecast)
        forecasted_volume_future = pd.Series(forecasted_volume_future, index=future_dates_range) # Przypisz indeks datowy

        print(f"\nPrognozowany Wolumen (od {forecast_start_date.date()} do {forecast_end_date.date()}):")
        print(forecasted_volume_future)

    except Exception as e:
        print(f"Błąd podczas prognozowania przyszłego wolumenu: {e}")
        forecasted_volume_future = pd.Series() # Pusta seria w razie błędu

# --- 8. Wizualizacja finalna (dane historyczne + przyszła prognoza) ---
plt.figure(figsize=(18, 8))

# Rysowanie rzeczywistego wolumenu od 2024-11-02
start_plot_date_str = '2024-11-02'
try:
    start_plot_date = pd.to_datetime(start_plot_date_str)
    # Sprawdzamy, czy data start_plot_date jest w indeksie lub przed ostatnią datą danych
    if start_plot_date <= data.index.max():
        actual_volume_slice = data.loc[start_plot_date:, 'Volume']
        plt.plot(actual_volume_slice.index, actual_volume_slice,
                 label='Rzeczywisty Wolumen (od ' + start_plot_date_str + ')',
                 color='blue', alpha=0.7)
    else:
        print(f"Ostrzeżenie: Data startu wykresu rzeczywistego wolumenu ({start_plot_date_str}) jest poza zakresem dostępnych danych historycznych. Wyświetlam cały dostępny wolumen.")
        plt.plot(data.index, data['Volume'], label='Rzeczywisty Wolumen (pełny zakres)', color='blue', alpha=0.7)
except KeyError:
    print(f"Ostrzeżenie: Data {start_plot_date_str} nie istnieje w indeksie danych. Wyświetlam cały dostępny wolumen.")
    plt.plot(data.index, data['Volume'], label='Rzeczywisty Wolumen (pełny zakres)', color='blue', alpha=0.7)
except Exception as e:
    print(f"Wystąpił błąd podczas cięcia danych dla wizualizacji: {e}. Wyświetlam cały dostępny wolumen.")
    plt.plot(data.index, data['Volume'], label='Rzeczywisty Wolumen (pełny zakres)', color='blue', alpha=0.7)


if not forecasted_volume_future.empty:
    plt.plot(forecasted_volume_future.index, forecasted_volume_future, label='Prognoza Wolumenu (Przyszłość)', color='red', linestyle='-.')

plt.title('Rzeczywisty Wolumen i Przyszła Prognoza ARIMA (Optymalny Model)')
plt.xlabel('Data')
plt.ylabel('Wolumen')
plt.legend()
plt.grid(True)

# Zapisywanie wykresu do pliku PNG
model_name_forecast_plot = "ARIMA_Optimal_Forecast"
plt.savefig(f'{model_name_forecast_plot}.png')
print(f"\nWykres został zapisany jako '{model_name_forecast_plot}.png'")

plt.show()