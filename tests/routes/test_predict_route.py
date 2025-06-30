from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)

def test_post_ville():
    body = {
        "ville_name": "lille",
        "surface_bati": 3000,
        "nombre_pieces": 4,
        "type_local": "Maison",
        "surface_terrain": 400,
        "nombre_lots": 10
    }
    resp = client.post(f"/predict", json=body)
    assert resp.status_code == 200