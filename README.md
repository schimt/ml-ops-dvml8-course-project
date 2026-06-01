# ml-ops-dvml8-course-project
MLOps course project

## Project structure

The repository separates the main MLOps pipeline from standalone course
experiments.

```text
src/
  model.py
  train.py
  api.py

experiments/
  scalable_training/
    train_ddp.py
    train_deepspeed.py
  scalable_inference/
    infer.py
    quantize.py
    prune.py
    finetune_prune.py
    timing.py
    evaluate_model.py
  monitoring/
    drift.py
  post_deployment/
    continual.py
    unlearning.py
```

The main pipeline is implemented in `src/`: model definition, config-based
training, MLflow logging, deployment, and FastAPI serving. The `experiments/`
directory contains module-specific scripts used to document scalable training,
scalable inference, monitoring, continual learning, and unlearning.

See `docs/run_commands.md` for the most common commands.

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

## Limitations

- The Docker image is built in CI, but it is not pushed to a registry because
  registry credentials and setup are outside the project scope.
- GitHub Actions is used for CI/CD instead of Jenkins. This is valid for the
  course because Jenkins, GitHub Actions, or another preferred CI/CD framework
  are allowed.
- Branch protection is a GitHub repository setting and is not configured in
  code.
- Android phone deployment was not performed. Local quantization experiments
  and FastAPI deployment were used as a fallback.
- EWC was implemented in the MM7 experiment. Replay + EWC was evaluated
  against naive sequential training and replay. EWC did not clearly outperform
  replay alone, but it demonstrates the regularization-based continual learning
  method required by the exercise.
- DDP and DeepSpeed experiments require the AI-Lab/GPU setup to run properly.
