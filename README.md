# ForcastingStockPriceApple

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)


ForcastingStockPriceApple ($AAPL)
Ten projekt ma na celu opracowanie i wdrożenie robustnego modelu uczenia maszynowego (ML) do prognozowania przyszłej ceny akcji spółki Apple Inc. ($AAPL).

Ewolucja Projektu i Zakres
Początkowo projekt zakładał eksplorację zarówno metodologii Uczenia Maszynowego (ML), jak i Głębokiego Uczenia (DL). Jednakże, po gruntownej analizie i eksperymentach, modele Głębokiego Uczenia zostały ostatecznie odrzucone ze względu na ich niezadowalającą wydajność w tym konkretnym zadaniu prognostycznym. 
Warto zaznaczyć, że użytkownik w aplikacji ma dostęp do wizualizacji modeli Głębokiego Uczenia – LSTM i GRU. Prace nad ich implementacją i optymalizacją będą kontynuowane w przyszłości.

Ponadto, wczesna hipoteza zakładała dwuetapową predykcję: najpierw prognozowanie wolumenu obrotu, a następnie wykorzystanie tej prognozy do przewidywania ceny zamknięcia. Obszerne analizy i testy wykazały, że to podejście nie było skuteczne i negatywnie wpływało na dokładność prognoz cen. W związku z tym, komponent prognozowania wolumenu został usunięty z głównego zakresu projektu.

Finalna wersja projektu koncentruje się wyłącznie na prognozowaniu ceny akcji za pomocą metod ML. Użytkownicy zainteresowani mogą jednak wykorzystać te wstępne fragmenty kodu DL do własnych eksperymentów lub innych celów.


Kluczowe Funkcjonalności
Modelowanie Ceny Akcji: Głównym rezultatem projektu jest wytrenowany model uczenia maszynowego, zdolny do bezpośredniej predykcji cen zamknięcia akcji.
Horyzont Predykcji: Model oferuje prognozy ceny na horyzoncie 30 dni. Należy jednak zaznaczyć, że jest to prognoza dla następnego dnia handlowego, którą można odświeżać codziennie.

**Zastosowanie w Handlu**: 
Dzięki zdolności modelu do przewidywania jutrzejszej ceny oraz informowania o przewidywanym wzroście lub spadku ceny, projekt ten może znaleźć zastosowanie zarówno dla daytraderów oraz innych inwestorów w spółkę Apple, jak i spekulantów.



## Instalacja

Poniższe kroki opisują, jak zainstalować i uruchomić aplikację na Twoim lokalnym komputerze.

### Wymagania wstępne

* **Python 3.10** (zalecana wersja) - Możesz pobrać go ze strony [python.org](https://www.python.org/downloads/).
* **Git** - Do sklonowania repozytorium. Możesz pobrać go ze strony [git-scm.com](https://git-scm.com/downloads).
* **Anaconda** (opcjonalnie, dla środowisk Conda) - Możesz pobrać ją ze strony [anaconda.com](https://www.anaconda.com/download/). Lżejszą alternatywą jest [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
* **Virtualenv** (opcjonalnie, dla zwykłych środowisk wirtualnych) - Zazwyczaj jest wbudowany w nowsze wersje Pythona (`python -m venv`).

### Kroki instalacji

1.  **Sklonuj repozytorium z GitHub:**

    Otwórz terminal lub wiersz poleceń i przejdź do folderu, w którym chcesz zainstalować aplikację. Następnie wykonaj polecenie:

    ```bash
    git clone https://github.com/KawiorPL/ForcastingStockPriceApple.git
     ```
    nastepnie
    ```bash
    cd ForcastingStockPriceApple
     ```
   

3.  **Utwórz i aktywuj środowisko wirtualne:**

    Wybierz jedną z poniższych opcji w zależności od preferowanego narzędzia:

    #### Opcja A: Użycie Conda (zalecane)

    Jeśli używasz Anacondy lub Minicondy:

    ```bash
    conda create --name Apple_env python=3.10
    conda activate Apple_env
    ```

    #### Opcja B: Użycie venv

    Jeśli nie używasz Condy:

    ```bash
    python -m venv venv
    # Aktywacja środowiska:
    # Windows (CMD):
    .\venv\Scripts\activate
    # Windows (PowerShell):
    .\venv\Scripts\Activate.ps1
    # macOS/Linux:
    source venv/bin/activate
    ```

    Po aktywacji środowiska nazwa środowiska (`(Apple_env)` lub `(venv)`) powinna pojawić się na początku linii poleceń.

4.  **Zainstaluj biblioteki:**

    Po aktywowaniu środowiska przejdź do folderu `ForcastingStockPriceApple` (jeśli jeszcze tam nie jesteś) i zainstaluj wymagane biblioteki z pliku `requirements.txt`:

    ```bash
    cd ForcastingStockPriceApple
    pip install -r requirements.txt
    ```

    To polecenie zainstaluje wszystkie pakiety wymienione w pliku `requirements.txt`.

### Uruchomienie aplikacji

Po pomyślnej instalacji wymaganych bibliotek możesz uruchomić aplikację FastAPI. W terminalu użyj następującej komendy:

```bash
uvicorn main:app --reload
```
Gdy zobaczysz poniższe komunikaty w konsoli, oznacza to, że aplikacja jest gotowa do użycia. Następnie kliknij link http://127.0.0.1:8000, aby otworzyć aplikację w przeglądarce.

<span style="color: green;">INFO</span>: Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)<br>
<span style="color: green;">INFO</span>: Started reloader process [11420] using WatchFiles<br>
<span style="color: green;">INFO</span>: Started server process [3940]<br>
<span style="color: green;">INFO</span>: Waiting for application startup.<br>
<span style="color: green;">INFO</span>: Application startup complete.<br>


```
echo "# ForcastingStockPriceApple" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:KawiorPL/ForcastingStockPriceApple.git
git push -u origin main
```
