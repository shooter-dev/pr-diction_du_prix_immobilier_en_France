# Point d’entrée FastAPI, routes des prédic
from fastapi import FastAPI

from app.routes.predict_route import router

app = FastAPI(
    title="API Agence Prediction Prix",
    description="",
    version="1.0.0")

app.include_router(router)