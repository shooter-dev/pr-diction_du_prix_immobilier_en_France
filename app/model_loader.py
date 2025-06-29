# Chargement des modèles ML
import os
import joblib

from app.schemas import TypeDeBien, ModelLoaderEntity


class ModelLoader:
    def __init__(self):
        self.model_map = {
            "LinearRegression": "linear_regression",
            "DecisionTreeRegressor": "decision_tree_regressor",
            "RandomForestRegressor": "random_forest_regressor"
        }

    def load(self, model_name: str, type_bien: TypeDeBien) -> ModelLoaderEntity:
        model_name = model_name.strip()

        if model_name not in self.model_map:
            raise ValueError(f"Modèle non reconnu : {model_name}")

        mapped_name = self.model_map[model_name]

        dossier = os.path.join('./models')
        if not os.path.exists(dossier):
            raise FileNotFoundError(f"Dossier introuvable : {dossier}")

        path_model = os.path.join(dossier, f"{mapped_name}_{type_bien.value}.pkl")
        path_scaler_x = os.path.join(dossier, f"{mapped_name}_{type_bien.value}_scaler_X.pkl")
        path_scaler_y = os.path.join(dossier, f"{mapped_name}_{type_bien.value}_scaler_y.pkl")


        model = joblib.load(path_model)
        scaler_x = joblib.load(path_scaler_x)
        scaler_y = joblib.load(path_scaler_y)

        print('tupe model', type(model))
        print('tupe scaler_x', type(scaler_x))
        print('tupe scaler_y', type(scaler_y))

        return ModelLoaderEntity(
            model=model,
            scaler_x=scaler_x,
            scaler_y=scaler_y
        )