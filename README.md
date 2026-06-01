# ml-ops-dvml8-course-project

This is my MLOps course project for cats-vs-dogs image classification. The
classifier is a small PyTorch CNN. The model is not the interesting part by
itself; the point of the project is to build the pipeline around it.

The repo covers config-based training, MLflow tracking, a simple model
acceptance step, deployment of an accepted model, a FastAPI inference service,
and local monitoring with Prometheus and Grafana. It also includes separate
experiment scripts for the course modules.

## Pipeline

The normal workflow is:

```text
configs/config.yaml
  -> python -m src.train
  -> MLflow logging
  -> model acceptance check
  -> deployment/cat_dog_cnn.pth
  -> FastAPI service
  -> Prometheus/Grafana monitoring
```

`configs/config.yaml` controls the training parameters. `src.train` trains and
evaluates the model, logs the run to MLflow, and checks whether the model meets
the configured threshold. If the model is accepted, the trained weights are
copied to `deployment/cat_dog_cnn.pth`.

The FastAPI service loads that deployed model and exposes prediction and
monitoring endpoints.

## Repository Layout

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

docs/
  model_card.md
  run_commands.md

monitoring/
  prometheus.yml
  grafana/

tests/
artifacts/
configs/
```

`src/` is the actual pipeline code: model, training script, and API.

`experiments/` is for the module-specific work, such as DDP, DeepSpeed,
quantization, pruning, drift detection, continual learning, EWC, and
unlearning. These scripts are not part of the serving path.

`monitoring/` has the Prometheus and Grafana setup. `docs/` has the model card
and a longer command list. `artifacts/` has generated plots and result files.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

If the DVC-tracked data/model files are missing locally:

```powershell
dvc pull
```

## Training and MLflow

Run the training pipeline:

```powershell
python -m src.train
```

Open MLflow:

```powershell
mlflow ui --backend-store-uri mlruns
```

The current threshold is:

```text
accuracy_threshold: 0.50
```

This threshold is low on purpose. It lets the project show the acceptance and
deployment flow even though the CNN is only a baseline model. A threshold such
as `0.80` would make more sense for a stronger classifier, but it would often
reject this model and skip the deployment step.

## API

Run the FastAPI service:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Useful endpoints:

- FastAPI docs: http://127.0.0.1:8000/docs
- `GET /health`
- `POST /predict`
- `GET /metrics`

The API loads:

```text
deployment/cat_dog_cnn.pth
```

## Monitoring

Run the API with Prometheus and Grafana:

```powershell
docker compose up --build
```

Services:

- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana login:

```text
admin / admin
```

The dashboard is loaded from `monitoring/grafana/dashboards`.

## Tests and CI

Run tests:

```powershell
pytest -q
```

Run pre-commit checks:

```powershell
pre-commit run --all-files
```

Build the Docker image locally:

```powershell
docker build -t mlops-catdog:test .
```

GitHub Actions is used as the CI/CD tool. The workflow runs pre-commit, tests,
coverage, and a Docker build. The Docker image is only built as a check; it is
not pushed to a registry.

## Experiments

The extra scripts are not part of the serving pipeline. They are used to
document the course module experiments.

Scalable inference:

```powershell
python -m experiments.scalable_inference.prune
python -m experiments.scalable_inference.timing
python -m experiments.scalable_inference.infer
```

Drift detection:

```powershell
python -m experiments.monitoring.drift
```

Post-deployment:

```powershell
python -m experiments.post_deployment.continual
python -m experiments.post_deployment.unlearning
```

The continual learning script compares naive sequential training, experience
replay, and replay with EWC. The unlearning script tests class-specific
unlearning on MNIST.

More commands are listed in `docs/run_commands.md`.

## Limitations

- The model accuracy is limited. The project is mainly about the MLOps pipeline,
  not about building the best cats-vs-dogs classifier.
- `accuracy_threshold = 0.50` is used to show the deployment flow.
- The Docker image is built in CI but not pushed to a registry.
- Android phone deployment was not completed. Local quantization experiments
  and FastAPI deployment were used as the fallback.
- Multi-node, DDP, and DeepSpeed experiments require the AI-Lab/GPU setup.
- The monitoring setup is local and does not include authentication, TLS, rate
  limiting, or alert rules.
