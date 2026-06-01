# Run Commands

Common commands for the MLOps course project.

## Main Pipeline

Train, evaluate, log to MLflow, and deploy the accepted model:

```powershell
python -m src.train
```

Run the FastAPI service:

```powershell
uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Run tests:

```powershell
pytest -q
```

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

## Post-Deployment Experiments

Run continual learning experiment:

```powershell
python -m experiments.post_deployment.continual
```

Run machine unlearning experiment:

```powershell
python -m experiments.post_deployment.unlearning
```
