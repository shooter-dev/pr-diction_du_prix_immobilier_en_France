from fastapi.testclient import TestClient
from app.main import app
from app.schemas import OutputPredict

client = TestClient(app)

def test_post_ville():
    ville = 'lille'
    resp = client.post(f"/predict/{ville}")
    assert resp.status_code == 200
    assert resp.json() == OutputPredict