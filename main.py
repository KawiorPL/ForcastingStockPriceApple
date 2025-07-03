# main.py
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from functions import  EDA, add_features, Model, EDA2, add_features2,Model2 # Importujemy nową funkcję
import pandas as pd

#uvicorn main:app --reload
app = FastAPI()

templates = Jinja2Templates(directory="templates")

GLOBAL_MARKET_DATA = None


def load_Market_data():
    global GLOBAL_MARKET_DATA
    if GLOBAL_MARKET_DATA is None:
        try:
            # Upewnij się, że ścieżka do pliku jest poprawna
            # Zmieniona linia: dodano date_parser i lambda z utc=True
            GLOBAL_MARKET_DATA = pd.read_csv(
                'data/World-Stock-Prices-Dataset.csv',
                parse_dates=["Date"],
                dayfirst=True,
                index_col="Date",
                # DODAJ TEN ARGUMENT:
                date_parser=lambda col: pd.to_datetime(col, utc=True)
            )
            # Konwersja indeksu na DatetimeIndex, jeśli nie jest
            # Ten blok nadal jest dobry, ale teraz pd.read_csv powinien to już zrobić poprawnie
            if not isinstance(GLOBAL_MARKET_DATA.index, pd.DatetimeIndex):
                GLOBAL_MARKET_DATA.index = pd.to_datetime(GLOBAL_MARKET_DATA.index, utc=True) # Ewentualnie dodaj utc=True tutaj też
            GLOBAL_MARKET_DATA.sort_index(inplace=True)
            print("Pomyślnie załadowano World-Stock-Prices-Dataset.csv")
        except FileNotFoundError:
            print("BŁĄD: World-Stock-Prices-Dataset.csv nie znaleziono. Upewnij się, że jest w katalogu 'data/'.")
            GLOBAL_MARKET_DATA = pd.DataFrame()
        except Exception as e:
            print(f"BŁĄD podczas ładowania World-Stock-Prices-Dataset.csv: {e}")
            GLOBAL_MARKET_DATA = pd.DataFrame()
    return GLOBAL_MARKET_DATA


# ZModyfikowana funkcja: pobieranie nazw firm z kolumny 'NEMO' z data.csv
def get_company_names_from_data():
    """Zwraca listę unikalnych nazw firm z kolumny 'NEMO' z pliku data.csv."""
    df_nemo = load_Market_data()
    print(f"test get_company_names_from_data {df_nemo}")
    if 'Brand_Name' in df_nemo.columns:
        return df_nemo['Brand_Name'].unique().tolist()


    return []

load_Market_data()


# Nowy endpoint do pobierania nazw firm dla dropdownu
@app.get("/get-company-names", response_class=JSONResponse)
async def get_company_names_endpoint():
    """
    Zwraca listę unikalnych nazw firm z pliku World-Stock-Prices-Dataset.csv,
    pobranych z kolumny 'Brand_Name'.
    """
    company_names = get_company_names_from_data()
    if not company_names:
        return JSONResponse(content={"error": "Nie udało się załadować nazw firm lub plik jest pusty."}, status_code=500)
    return JSONResponse(content=company_names)



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/run-demo", response_class=JSONResponse)
async def get_data_science_demo_results():
    #results = run_data_science_demo()
    eda_results = EDA()
    feature_results = add_features()
    model_results = Model()
    combined_results = {**eda_results, **feature_results, **model_results}
    return JSONResponse(content=combined_results)

@app.get("/analyze-company", response_class=JSONResponse)
async def analyze_company_data(company_name: str = Query(..., description="Nazwa firmy do analizy")):
    """
    Endpoint do przeprowadzania kompleksowej analizy (EDA, Feature Engineering, Modelowanie)
    dla wybranej spółki z World-Stock-Prices-Dataset.csv.
    """


    try:
        # Wywołaj poszczególne etapy analizy, PRZEKAZUJĄC przefiltrowany DataFrame
        eda_results = EDA2(company_name) # Przekaż kopię, aby EDA mogła ją modyfikować bez wpływu na resztę
        feature_results = add_features2(company_name) # Przekaż kopię
        model_results = Model2() # Przekaż kopię

        combined_results = {
            **eda_results,
            **feature_results,
            **model_results
        }
        return JSONResponse(content=combined_results)
    except Exception as e:
        # Obsługa błędów, jeśli coś pójdzie nie tak w EDA/FE/Model
        print(f"Błąd podczas przetwarzania analizy dla firmy {company_name}: {e}")
        import traceback
        traceback.print_exc() # Wydrukuj pełny traceback do konsoli serwera
        return JSONResponse(content={"error": f"Wystąpił błąd podczas analizy danych dla firmy '{company_name}'. Szczegóły: {str(e)}"}, status_code=500)
# Aby uruchomić aplikację, zapisz ten plik jako main.py i uruchom w terminalu:
# uvicorn main:app --reload