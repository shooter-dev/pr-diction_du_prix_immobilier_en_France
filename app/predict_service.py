import time

import numpy as np

from app.model_loader import ModelLoader
from app.schemas import OutputPredict, RequestPredict, TypeDeBien, ModelLoaderEntity, ModelLLM

model_loader: ModelLoader = ModelLoader()

class PredictService:
    def get_predict_ville(self, features: RequestPredict, model_llm: ModelLLM):

        model: ModelLoaderEntity = model_loader.load(
            model_name=model_llm.name_model,
            type_bien=model_llm.type_bien
        )

        data_np = np.array(
            [[features.surface_bati,
              features.nombre_pieces,
              0 if features.nombre_pieces == "Maison" else 1,
              features.surface_terrain,
              features.nombre_lots
              ]])

        # Prédiction
        entree_scaled = model.scaler_x.transform(data_np)
        pred_scaled = model.model.predict(entree_scaled)
        prix_m2 = model.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

        # Calcul du prix total

        print(f"prix: {prix_m2} €")

        return OutputPredict(
            prix_m2_estime=prix_m2,
            ville_modele=features.ville_name,
            model="RandomForestRegressor"
        )