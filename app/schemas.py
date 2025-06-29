# Modèles Pydantic pour validation des requêtes
from enum import Enum

import sklearn
from nltk import SklearnClassifier
from pydantic import BaseModel


class TypeDeBien(Enum):
    MAISON = "maison"
    APPARTEMENT = "appart"

class RequestPredict(BaseModel):
    ville_name: str
    surface_bati: float
    nombre_pieces: int
    type_local: str
    surface_terrain: float
    nombre_lots: int

class InputPredict(BaseModel):
    surface_bati: float
    nombre_pieces: int
    type_local: str
    surface_terrain: float
    nombre_lots: int

class OutputPredict(BaseModel):
    prix_m2_estime: float
    ville_modele: str
    model: str

class ModelLoaderEntity:

    def __init__(self, model, scaler_x, scaler_y):
        self.model: sklearn.ensemble = model
        self.scaler_x: sklearn.preprocessing = scaler_x
        self.scaler_y: sklearn.preprocessing = scaler_y

class ModelLLM(BaseModel):
    name_model: str
    type_bien: TypeDeBien
