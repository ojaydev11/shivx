# ShivX MLOps - Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Prerequisites

- Docker & Docker Compose installed
- Python 3.10+ (for local development)
- 8GB+ RAM recommended
- 20GB+ disk space

### Step 1: Clone & Setup

```bash
cd shivx/
cp .env.example .env

# Edit .env with your passwords
nano .env
```

### Step 2: Start MLOps Infrastructure

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps

# Expected output:
# ✓ postgres
# ✓ redis
# ✓ mlflow
# ✓ celery-worker
# ✓ celery-beat
# ✓ prometheus
# ✓ grafana
# ✓ api
```

### Step 3: Verify Installation

```bash
# Check API health
curl http://localhost:8000/health

# Check MLflow
curl http://localhost:5000/health

# Check Prometheus
curl http://localhost:9090/-/healthy

# Check Grafana
curl http://localhost:3000/api/health
```

### Step 4: Access Dashboards

| Service | URL | Default Login |
|---------|-----|---------------|
| API Docs | http://localhost:8000/docs | N/A |
| MLflow | http://localhost:5000 | N/A |
| Grafana | http://localhost:3000 | admin/admin |
| Prometheus | http://localhost:9090 | N/A |

### Step 5: Train Your First Model

```python
import asyncio
from app.ml.training import TrainingPipeline, TrainingConfig
from app.ml.registry import ModelRegistry

async def train_first_model():
    # Initialize components
    pipeline = TrainingPipeline()
    registry = ModelRegistry()

    # Configure training
    config = TrainingConfig(
        model_name="my_first_trading_model",
        model_type="supervised",
        dataset_path="/data/training.csv",
        hyperparameters={
            "input_size": 10,
            "hidden_size": 64,
            "output_size": 2
        },
        epochs=50,
        batch_size=32
    )

    # Train (replace with your data)
    import numpy as np
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, 1000)

    result = await pipeline.train_model(config, X, y)

    print(f"Training completed!")
    print(f"Model version: {result.model_version}")
    print(f"Training time: {result.training_time_seconds:.2f}s")
    print(f"Validation metrics: {result.validation_metrics}")

# Run training
asyncio.run(train_first_model())
```

### Step 6: Make Your First Prediction

```python
import asyncio
from app.ml.inference import MLInferenceService

async def make_prediction():
    service = MLInferenceService()
    await service.initialize()

    # Make prediction
    result = await service.predict_async(
        model_id="my_first_trading_model",
        features={
            "rsi": 65.2,
            "macd": 1.5,
            "volume": 1000000,
            "price_change": 2.3
        }
    )

    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Latency: {result['latency_ms']:.2f}ms")

asyncio.run(make_prediction())
```

### Step 7: Monitor Your Models

1. **Open Grafana**: http://localhost:3000
2. **Login**: admin/admin
3. **Navigate to**: Dashboards → ShivX MLOps Dashboard
4. **View metrics**:
   - Prediction rate
   - Latency (P95, P99)
   - Model accuracy
   - Drift scores

### Step 8: Set Up Drift Detection

```python
import asyncio
from app.ml.monitor import ModelMonitor

async def setup_monitoring():
    monitor = ModelMonitor(drift_threshold=0.25)
    await monitor.initialize()

    # Set baseline distribution
    baseline_features = [
        {"rsi": 60, "macd": 1.0, "volume": 900000},
        {"rsi": 65, "macd": 1.5, "volume": 1100000},
        {"rsi": 70, "macd": 2.0, "volume": 1200000},
        # ... more samples ...
    ]

    await monitor.set_baseline_distribution(
        model_id="my_first_trading_model",
        features=baseline_features
    )

    print("Drift detection configured!")

asyncio.run(setup_monitoring())
```

## 🎯 Common Tasks

### Deploy Model to Production

```python
from app.ml.registry import ModelRegistry

registry = ModelRegistry()

# Promote to production
registry.promote_model(
    model_name="my_first_trading_model",
    version="1",
    stage="Production",
    archive_existing=True
)
```

### Rollback Model

```python
registry.rollback_model(
    model_name="my_first_trading_model",
    target_version="1"  # previous version
)
```

### Schedule Automated Retraining

```python
from app.ml.training import TrainingPipeline

pipeline = TrainingPipeline()

await pipeline.schedule_retraining(
    model_name="my_first_trading_model",
    schedule="daily",  # or "weekly", "monthly"
    config=training_config
)
```

### Optimize Model for Production

```python
from app.ml.serving import ModelServingOptimizer

optimizer = ModelServingOptimizer(quantize=True)

# Convert to ONNX
onnx_path = optimizer.convert_to_onnx(
    model=pytorch_model,
    model_name="my_first_trading_model",
    input_shape=(1, 10)
)

# Load and benchmark
session = optimizer.load_onnx_model(onnx_path, "my_first_trading_model")
stats = optimizer.benchmark_models(pytorch_model, session, (1, 10))

print(f"ONNX speedup: {stats['speedup']:.2f}x")
```

### Get Model Explanation

```python
from app.ml.explainability import XAISystem

xai = XAISystem()

explanation = xai.explain_prediction(
    model=model,
    model_id="my_first_trading_model",
    prediction_id="pred_123",
    features=input_features,
    feature_names=["rsi", "macd", "volume"],
    prediction="BUY",
    confidence=0.82,
    method="lime"
)

print(explanation.explanation_text)
```

## 🔧 Troubleshooting

### Services Not Starting

```bash
# Check Docker
docker --version
docker-compose --version

# Check ports are free
netstat -tulpn | grep -E '8000|5000|6379|5432|9090|3000'

# Check logs
docker-compose logs <service-name>

# Restart specific service
docker-compose restart <service-name>
```

### MLflow Connection Issues

```bash
# Check MLflow is running
docker-compose ps mlflow

# Check MLflow logs
docker-compose logs mlflow

# Test MLflow connection
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

### Redis Connection Issues

```bash
# Check Redis
docker-compose ps redis

# Test Redis connection
docker-compose exec redis redis-cli -a <password> PING

# Should return: PONG
```

### Celery Workers Not Processing

```bash
# Check worker status
docker-compose logs celery-worker

# Inspect active tasks
docker-compose exec celery-worker celery -A app.ml.inference inspect active

# Restart workers
docker-compose restart celery-worker
```

## 📊 Monitoring Checklist

Daily:
- [ ] Check Grafana dashboard
- [ ] Review model accuracy
- [ ] Check prediction latency
- [ ] Review error logs

Weekly:
- [ ] Check drift reports
- [ ] Review model performance trends
- [ ] Check cache hit rates
- [ ] Update baselines if needed

Monthly:
- [ ] Schedule model retraining
- [ ] Review A/B test results
- [ ] Optimize underperforming models
- [ ] Update documentation

## 🚨 Production Deployment Checklist

Before deploying to production:

- [ ] All tests passing (80%+ coverage)
- [ ] Model accuracy >80%
- [ ] Validation against current production passed
- [ ] Drift detection configured
- [ ] Monitoring alerts configured
- [ ] Rollback plan documented
- [ ] Environment variables set
- [ ] Secrets properly secured
- [ ] Backups configured
- [ ] Load testing completed
- [ ] Documentation updated

## 📚 Next Steps

1. Read the full [MLOPS_README.md](./MLOPS_README.md)
2. Explore the [API Documentation](http://localhost:8000/docs)
3. Review [test examples](./tests/test_ml_models.py)
4. Check out [pipeline examples](./app/ml/pipeline.py)
5. Configure your first [scheduled retraining](./MLOPS_README.md#automated-training-pipeline)

## 🆘 Getting Help

- Check logs: `docker-compose logs <service>`
- Check metrics: http://localhost:9090
- Check dashboards: http://localhost:3000
- Review full docs: [MLOPS_README.md](./MLOPS_README.md)

## 🎉 You're Ready!

Your MLOps infrastructure is now running. Start building production-ready ML models for trading!

Key URLs:
- **API**: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090
