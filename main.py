# main.py
from fastapi import FastAPI, Request, UploadFile, File, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from functions import  EDA, Model

#uvicorn main:app --reload
app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static") 


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/run-demo", response_class=JSONResponse)
async def get_data_science_demo_results():
    #results = run_data_science_demo()
    eda_results = EDA()
    model_results = Model()
    combined_results = {**eda_results, **model_results}
    return JSONResponse(content=combined_results)


# Aby uruchomić aplikację, zapisz ten plik jako main.py i uruchom w terminalu:
# uvicorn main:app --reload