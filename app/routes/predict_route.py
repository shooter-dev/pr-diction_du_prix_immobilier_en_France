from fastapi import APIRouter, Request
from httpx import request

from app.predict_service import PredictService
from app.schemas import RequestPredict, OutputPredict, ModelLLM, TypeDeBien

router = APIRouter(
    prefix="/predict"
)

service_predict: PredictService = PredictService()

@router.post("/", response_model=OutputPredict, summary="", description="Renvoie la prediction du prix du bien.")
async def get_ville(request: RequestPredict) -> OutputPredict:
    """
    :ville_name: name_model ville
    :type request: Request
    """
    #features: RequestPredict(**request)
    #return await service_predict.get_predict_ville(ville_name, features)

    model = ModelLLM(name_model="RandomForestRegressor",
                     type_bien=TypeDeBien.MAISON if request.type_local == "Maison" else TypeDeBien.APPARTEMENT)
    return service_predict.get_predict_ville(request, model)

