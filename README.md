# ml-ops-dvml8-course-project
MLOps course project

## FastAPI deployment and monitoring

The deployed cats-vs-dogs model can be served through a FastAPI service. The
service loads `deployment/cat_dog_cnn.pth`, exposes a prediction endpoint, and
exports Prometheus metrics for request counts, prediction counts, model load
status, errors, and inference latency.

Run the API locally:

```powershell
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health` checks whether the deployed model is available.
- `POST /predict` accepts an uploaded image file and returns the predicted class,
  confidence, and inference latency.
- `GET /metrics` exposes Prometheus metrics.

Run the API together with Prometheus and Grafana:

```powershell
docker compose up --build
```

Services:

- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana uses the default credentials `admin` / `admin` and automatically loads
the `Cats vs Dogs API Monitoring` dashboard from `monitoring/grafana/dashboards`.
