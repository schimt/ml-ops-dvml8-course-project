# Run Commands

Common commands used in the project.

## Main Pipeline

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train the model, evaluate it, log the run to MLflow, and copy the accepted
model to `deployment/`:

```powershell
python -m src.train
```

Open the MLflow UI:

```powershell
mlflow ui --backend-store-uri mlruns
```

Run the FastAPI service:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

FastAPI URLs:

- Docs: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/health
- Metrics endpoint: http://127.0.0.1:8000/metrics

Run tests:

```powershell
pytest -q
```

Run tests with coverage:

```powershell
pytest --cov=src --cov-report=term-missing
```

Run pre-commit checks:

```powershell
pre-commit run --all-files
```

Run Flake8 directly:

```powershell
python -m flake8 src experiments tests
```

Build the Docker image locally:

```powershell
docker build -t mlops-catdog:test .
```

This checks that the project can be built as a container. GitHub Actions runs
the same build check, but the image is not pushed to a registry.

## Scalable Training Experiments

Run Distributed Data Parallel training with `torchrun`:

```powershell
torchrun --nproc_per_node=2 -m experiments.scalable_training.train_ddp
```

Run DeepSpeed training:

```powershell
deepspeed -m experiments.scalable_training.train_deepspeed
```

## Scalable Inference Experiments

Quantize the trained model:

```powershell
python -m experiments.scalable_inference.quantize
```

Measure batch inference:

```powershell
python -m experiments.scalable_inference.infer
```

Run pruning experiment:

```powershell
python -m experiments.scalable_inference.prune
```

Fine-tune a pruned model:

```powershell
python -m experiments.scalable_inference.finetune_prune
```

Measure FP32 vs INT8 latency:

```powershell
python -m experiments.scalable_inference.timing
```

Evaluate the trained model:

```powershell
python -m experiments.scalable_inference.evaluate_model
```

## Monitoring Experiments

Run the local drift detection experiment:

```powershell
python -m experiments.monitoring.drift
```

Run the API, Prometheus, and Grafana stack:

```powershell
docker compose up --build
```

Stop the stack:

```powershell
docker compose down
```

Monitoring URLs:

- FastAPI: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

Grafana default login:

```text
admin / admin
```

## Post-Deployment Experiments

Run continual learning experiment:

```powershell
python -m experiments.post_deployment.continual
```

Run machine unlearning experiment:

```powershell
python -m experiments.post_deployment.unlearning
```
