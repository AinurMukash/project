from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import re
import os

app = FastAPI()

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели и векторизатора
BASE_DIR = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE_DIR, "models", "DecisionTreeClassifier_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"))

class TextIn(BaseModel):
    text: str

# Обработка POST-запроса
@app.post("/predict")
def predict(data: TextIn):
    text = data.text
    text = re.sub(r'[^\w\s]', '', text.lower())
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)
    return {"prediction": int(prediction[0])}

# Отдаём frontend (React) из dist или build
frontend_path = os.path.join(BASE_DIR, "frontend", "dist")  # если Vite
# frontend_path = os.path.join(BASE_DIR, "frontend", "build")  # если CRA

app.mount("/static", StaticFiles(directory=frontend_path), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_path, "index.html"))

# 🚀 Запуск через Railway (uvicorn)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
