
import keras
import tensorflow as tf

# Ustawienie seedów dla powtarzalności
keras.utils.set_random_seed(43)
tf.config.experimental.enable_op_determinism()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch
from tensorflow.keras.callbacks import EarlyStopping
import joblib
# --- Wczytanie i przygotowanie danych (część niezmienna) ---

data = pd.read_csv('data/apple_stock.csv')
data = data[['Unnamed: 0', 'Volume']]
columndate = data.dtypes.loc[data.dtypes == 'object'].index[0]
data[columndate] = pd.to_datetime(data[columndate])
data.set_index(columndate, inplace=True)

series = data[['Volume']].values

# Podział na zbiór treningowy i testowy (jednorazowy podział danych surowych)
train_data, test_data = train_test_split(series , test_size=0.2, shuffle=False)

# Skalowanie danych (jednorazowe dopasowanie na zbiorze treningowym)
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

joblib.dump(scaler, 'volume_MinMaxScaler.pkl')
print("MinMaxScaler dla 'Volume' został zapisany jako 'volume_MinMaxScaler.pkl'.")
# --- KONIEC ZAPISU SCALERA ---

# Funkcja do tworzenia sekwencji (zależy od seq_length)
def create_sequences(data, seq_lenght):
    X, y = [], []
    for i in range(len(data) - seq_lenght):
        X.append(data[i:i + seq_lenght])
        y.append(data[i + seq_lenght])
    return np.array(X), np.array(y)

# Definicje modeli do tuningu (będą przyjmować seq_length jako argument)
def build_lstm_model(hp, seq_lenght_val): # Dodano seq_lenght_val
    model = Sequential()
    model.add(LSTM(hp.Int('lstml_units', min_value=32, max_value=128, step=32),
                   return_sequences=True, input_shape=(seq_lenght_val,1))) # Użycie seq_lenght_val
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(LSTM(hp.Int('lstm2_units', min_value=32, max_value=128, step=32)))
    model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
    model.add(Dense(1))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
    return model
# --- ZAKOMENTOWANY MODEL GRU ---
# def build_gru_model(hp, seq_lenght_val): # Dodano seq_lenght_val
#     model = Sequential()
#     model.add(GRU(hp.Int('gru1_units', min_value=32, max_value=128, step=32),
#                   return_sequences=True,input_shape=(seq_lenght_val,1))) # Użycie seq_lenght_val
#     model.add(BatchNormalization())
#     model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
#     model.add(GRU(hp.Int('gru2_units', min_value=32, max_value=128, step=32)))
#     model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
#     model.add(Dense(1))
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
#     model.compile(optimizer=Adam(learning_rate=hp_learning_rate), loss='mse')
#     return model

# --- Lista długości sekwencji do przetestowania ---
seq_lengths_to_test = [80]
all_results = {} # Słownik do przechowywania wyników dla każdej długości sekwencji

# Callback dla Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# --- Pętla po różnych długościach sekwencji ---
for current_seq_length in seq_lengths_to_test:
    print(f"\n{'='*60}")
    print(f"--- ROZPOCZYNAM TESTOWANIE DLA seq_length = {current_seq_length} ---")
    print(f"{'='*60}\n")

    # Tworzenie sekwencji dla bieżącej długości
    X_train, y_train = create_sequences(train_scaled, current_seq_length)
    X_test, y_test = create_sequences(test_scaled, current_seq_length)

    # Sprawdzenie, czy po utworzeniu sekwencji jest wystarczająco dużo danych
    if len(X_train) == 0 or len(X_test) == 0:
        print(f"Brak wystarczającej liczby próbek dla seq_length={current_seq_length}. Pomijam.")
        continue

    # Przygotowanie danych walidacyjnych
    val_split_ratio = 0.1
    val_split_index = int(len(X_train) * (1 - val_split_ratio))

    X_train_for_tuning, y_train_for_tuning = X_train[:val_split_index], y_train[:val_split_index]
    X_val, y_val = X_train[val_split_index:], y_train[val_split_index:]

    if len(X_train_for_tuning) == 0 or len(X_val) == 0:
        print(f"Brak wystarczającej liczby próbek treningowych/walidacyjnych po podziale dla seq_length={current_seq_length}. Pomijam.")
        continue

    # --- Tuning modelu LSTM ---
    print(f"\n--- Rozpoczynam tuning modelu LSTM dla seq_length={current_seq_length} ---")
    tuner_lstm = RandomSearch(
       lambda hp: build_lstm_model(hp, current_seq_length), # Przekazanie seq_length do funkcji build
       objective='val_loss',
       max_trials=50,
       executions_per_trial=2,
       directory=f'lstm_tuning_seq{current_seq_length}', # Dynamiczna nazwa katalogu
       project_name='lstm_volume_prediction',
       seed=42
    )

    tuner_lstm.search(X_train_for_tuning, y_train_for_tuning,
                      validation_data=(X_val, y_val),
                      epochs=50,
                      batch_size=32,
                      callbacks=[early_stopping],
                      verbose=1)

    best_lstm = tuner_lstm.get_best_models(1)[0]
    best_lstm_params = tuner_lstm.get_best_hyperparameters(1)[0].values
    print(f"Najlepsze hiperparametry dla LSTM (seq_length={current_seq_length}): {best_lstm_params}")
    best_lstm.save(f'best_lstm_volume_tuned_seq{current_seq_length}.h5')
    print(f"Najlepszy model LSTM dla seq_length={current_seq_length} zapisany.")


    # --- ZAKOMENTOWANY KOD TUNINGU MODELU GRU --
    # --- Tuning modelu GRU ---
    # print(f"\n--- Rozpoczynam tuning modelu GRU dla seq_length={current_seq_length} ---")
    # tuner_gru = RandomSearch(
    #    lambda hp: build_gru_model(hp, current_seq_length), # Przekazanie seq_length do funkcji build
    #    objective='val_loss',
    #    max_trials=50,
    #    executions_per_trial=2,
    #    directory=f'gru_tuning_seq{current_seq_length}', # Dynamiczna nazwa katalogu
    #    project_name='gru_volume_prediction',
    #    seed=42
    # )

    # tuner_gru.search(X_train_for_tuning, y_train_for_tuning,
    #                  validation_data=(X_val, y_val),
    #                  epochs=50,
    #                  batch_size=32,
    #                  callbacks=[early_stopping],
    #                  verbose=1)

    # best_gru = tuner_gru.get_best_models(1)[0]
    # best_gru_params = tuner_gru.get_best_hyperparameters(1)[0].values
    # print(f"Najlepsze hiperparametry dla GRU (seq_length={current_seq_length}): {best_gru_params}")
    # best_gru.save(f'best_gru_volume_tuned_seq{current_seq_length}.h5')
    # print(f"Najlepszy model GRU dla seq_length={current_seq_length} zapisany.")

    # --- Predykcje i ocena modeli dla bieżącej seq_length ---
    pred_lstm_scaled = best_lstm.predict(X_test)
    # pred_gru_scaled = best_gru.predict(X_test)

    # Odwrócenie skalowania
    pred_lstm = scaler.inverse_transform(pred_lstm_scaled)
    # pred_gru = scaler.inverse_transform(pred_gru_scaled)
    y_test_inv = scaler.inverse_transform(y_test)

    # Obliczenie metryk
    rmse_lstm = np.sqrt(mean_squared_error(y_test_inv, pred_lstm))
    r2_lstm = r2_score(y_test_inv, pred_lstm)
    # rmse_gru = np.sqrt(mean_squared_error(y_test_inv, pred_gru))
    # r2_gru = r2_score(y_test_inv, pred_gru)

    print(f'\n--- Wyniki Oceny Modeli dla seq_length={current_seq_length} ---')
    print(f'RMSE LSTM (Volume): {rmse_lstm:.4f}')
    print(f'R2 LSTM (Volume): {r2_lstm:.4f}')
    # print(f'RMSE GRU (Volume): {rmse_gru:.4f}')
    # print(f'R2 GRU (Volume): {r2_gru:.4f}')

    # Zapis wyników do słownika
    all_results[f'LSTM_seq{current_seq_length}'] = {'RMSE': rmse_lstm, 'R2': r2_lstm, 'Best Params': best_lstm_params}
    # all_results[f'GRU_seq{current_seq_length}'] = {'RMSE': rmse_gru, 'R2': r2_gru, 'Best Params': best_gru_params}


    # --- Wizualizacja Wyników ---
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 7))
    plt.plot(y_test_inv, label='Rzeczywiste Volume')
    plt.plot(pred_lstm, label=f'LSTM Predykcja Volume (seq_len={current_seq_length})', linestyle='--')
    # plt.plot(pred_gru, label=f'GRU Predykcja Volume (seq_len={current_seq_length})', linestyle=':')
    plt.title(f'Porównanie modeli: LSTM vs GRU (Prognoza Volume, seq_length={current_seq_length})')
    plt.xlabel('Krok Czasowy w Zbierze Testowym')
    plt.ylabel('Volume (oryginalna skala)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'Volume_LSTM_GRU_predictions_seq{current_seq_length}.png')
    plt.show()

# --- Końcowe podsumowanie wszystkich wyników ---
print(f"\n{'='*60}")
print("--- KOMPLETNE PODSUMOWANIE WYNIKÓW DLA RÓŻNYCH DŁUGOŚCI SEKWENCJI ---")
print(f"{'='*60}")

results_df_dl = pd.DataFrame.from_dict(all_results, orient='index')
results_df_dl.index.name = 'Model_SeqLength'
results_df_dl_sorted = results_df_dl.sort_values(by='RMSE', ascending=True)

print(results_df_dl_sorted)
print("\n")

results_df_dl_sorted.to_csv('DLscoreLSTM80.csv')
