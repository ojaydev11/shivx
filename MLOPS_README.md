# ShivX MLOps Infrastructure

## Overview

Production-ready ML operations infrastructure for the ShivX AI Trading System. This MLOps platform provides comprehensive model lifecycle management, monitoring, and optimization.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ShivX MLOps Platform                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   MLflow     │  │    Redis     │  │  PostgreSQL  │      │
│  │   Registry   │  │    Cache     │  │   Database   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Async      │  │    Model     │  │   Feature    │      │
│  │  Inference   │  │  Monitoring  │  │    Store     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Training    │  │   Serving    │  │     XAI      │      │
│  │  Pipeline    │  │ Optimization │  │  (LIME/SHAP) │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Prometheus   │  │   Grafana    │                         │
│  │  Metrics     │  │  Dashboard   │                         │
│  └──────────────┘  └──────────────┘                         │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Model Registry (MLflow)

**File**: `app/ml/registry.py`

Features:
- Semantic versioning (major.minor.patch)
- Model promotion workflow (dev → staging → production)
- Experiment tracking
- Model metadata and artifact management
- Model comparison and rollback

**Usage**:
```python
from app.ml.registry import ModelRegistry

registry = ModelRegistry(tracking_uri="http://mlflow:5000")

# Register model
version = registry.register_model(
    model=trained_model,
    model_name="rl_trading_ppo",
    model_type="rl",
    framework="pytorch",
    metadata={"author": "ml_team", "accuracy": 0.85}
)

# Promote to production
registry.promote_model(
    model_name="rl_trading_ppo",
    version=version,
    stage="Production"
)

# Load production model
model = registry.load_model("rl_trading_ppo", stage="Production")
```

**MLflow UI**: http://localhost:5000

### 2. Async ML Inference

**File**: `app/ml/inference.py`

Features:
- Non-blocking async predictions
- Batch inference for efficiency
- Redis caching (default TTL: 1 hour)
- Celery task queue
- Automatic timeout handling (5s default)

**Usage**:
```python
from app.ml.inference import MLInferenceService

service = MLInferenceService()
await service.initialize()

# Single prediction
result = await service.predict_async(
    model_id="rl_trading_ppo",
    features={"rsi": 65.2, "macd": 1.5},
    timeout=5.0
)

# Batch prediction
results = await service.predict_batch(
    model_id="rl_trading_ppo",
    features_list=[features1, features2, features3]
)
```

**Performance Targets**:
- Latency: <500ms (P95)
- Cache hit rate: >70%
- Throughput: 1000+ predictions/sec

### 3. Model Performance Monitoring

**File**: `app/ml/monitor.py`

Features:
- Real-time accuracy tracking
- Data drift detection (PSI, KS test)
- Prediction logging
- Latency monitoring
- Automated alerting

**Usage**:
```python
from app.ml.monitor import ModelMonitor, PredictionLog

monitor = ModelMonitor(drift_threshold=0.25)
await monitor.initialize()

# Log prediction
await monitor.log_prediction(PredictionLog(
    prediction_id="pred_123",
    model_id="rl_trading_ppo",
    model_version="1.0",
    features=features,
    prediction="BUY",
    confidence=0.82,
    latency_ms=45.2
))

# Detect drift
drift_report = await monitor.detect_data_drift(
    model_id="rl_trading_ppo",
    current_features=recent_features,
    method="psi"
)

# Get model health
health = await monitor.get_model_health("rl_trading_ppo")
```

**Drift Thresholds**:
- PSI < 0.1: No drift
- PSI 0.1-0.25: Small change
- PSI > 0.25: Significant drift (alert)

### 4. Automated Training Pipeline

**File**: `app/ml/training.py`

Features:
- Automated retraining workflows
- Model validation before deployment
- A/B testing framework
- Auto-promotion on improvement
- Rollback mechanism

**Usage**:
```python
from app.ml.training import TrainingPipeline, TrainingConfig

pipeline = TrainingPipeline()

# Configure training
config = TrainingConfig(
    model_name="rl_trading_ppo_v2",
    model_type="rl",
    dataset_path="/data/training.csv",
    hyperparameters={"learning_rate": 0.001},
    epochs=100,
    early_stopping=True
)

# Train model
result = await pipeline.train_model(config, X_train, y_train)

# Validate against current production
validation = await pipeline.validate_model(
    model_name="rl_trading_ppo",
    new_version=result.model_version,
    validation_data=X_val,
    validation_labels=y_val
)

# Auto-promote if better
if validation.promotion_approved:
    await pipeline.auto_promote_model(
        model_name="rl_trading_ppo",
        new_version=result.model_version,
        validation_result=validation,
        canary_percentage=0.1  # 10% canary deployment
    )
```

**Scheduling**:
```python
# Schedule daily retraining
await pipeline.schedule_retraining(
    model_name="rl_trading_ppo",
    schedule="daily",
    config=training_config
)
```

### 5. Model Serving Optimization

**File**: `app/ml/serving.py`

Features:
- ONNX conversion for faster inference
- INT8 quantization (4x size reduction)
- Model warming on startup
- GPU batch inference support
- Performance benchmarking

**Usage**:
```python
from app.ml.serving import ModelServingOptimizer

optimizer = ModelServingOptimizer(use_gpu=False, quantize=True)

# Convert to ONNX
onnx_path = optimizer.convert_to_onnx(
    model=pytorch_model,
    model_name="rl_trading_ppo",
    input_shape=(1, 10)
)

# Load ONNX model
session = optimizer.load_onnx_model(onnx_path, "rl_trading_ppo")

# Warm up model
optimizer.warm_up_model(session, input_shape=(1, 10))

# Fast inference
predictions = optimizer.predict_onnx(session, input_data)

# Benchmark
results = optimizer.benchmark_models(
    original_model=pytorch_model,
    onnx_session=session,
    input_shape=(1, 10),
    num_iterations=100
)
print(f"Speedup: {results['speedup']:.2f}x")
```

**Expected Improvements**:
- 2-5x faster inference (ONNX)
- 4x smaller model size (INT8)
- <100ms latency for most models

### 6. Feature Store

**File**: `app/ml/features.py`

Features:
- Centralized feature management
- Redis caching with TTL
- Feature versioning
- Schema validation
- Feature importance tracking

**Usage**:
```python
from app.ml.features import FeatureStore, FeatureSchema

store = FeatureStore(cache_ttl=3600)
await store.initialize()

# Register feature
schema = FeatureSchema(
    name="rsi",
    dtype="float",
    min_value=0.0,
    max_value=100.0,
    description="Relative Strength Index"
)

store.register_feature(schema, computation_fn=compute_rsi)

# Compute features
features = await store.compute_features(
    feature_names=["rsi", "macd", "volume_trend"],
    input_data=raw_market_data,
    use_cache=True
)

# Set feature importance
store.set_feature_importance({
    "rsi": 0.35,
    "macd": 0.28,
    "volume_trend": 0.22
})

# Get top features
top_features = store.get_top_features(n=10)
```

### 7. Model Explainability (XAI)

**File**: `app/ml/explainability.py`

Features:
- LIME for local explanations
- SHAP for global explanations
- Counterfactual generation
- Confidence intervals
- Feature importance visualization

**Usage**:
```python
from app.ml.explainability import XAISystem

xai = XAISystem(enable_lime=True, enable_shap=True)

# Explain prediction
explanation = xai.explain_prediction(
    model=trained_model,
    model_id="rl_trading_ppo",
    prediction_id="pred_123",
    features=input_features,
    feature_names=["rsi", "macd", "volume_trend"],
    prediction="BUY",
    confidence=0.82,
    method="lime"
)

print(explanation.explanation_text)

# Global feature importance
global_exp = xai.explain_global(
    model=trained_model,
    model_id="rl_trading_ppo",
    training_data=X_train,
    feature_names=feature_names,
    method="shap"
)
```

### 8. ML Pipeline Orchestration

**File**: `app/ml/pipeline.py`

Features:
- End-to-end ML pipeline management
- DAG-based execution
- Dependency management
- Failure recovery with retries
- Pipeline versioning

**Usage**:
```python
from app.ml.pipeline import MLPipeline, PipelineStage

pipeline = MLPipeline(
    pipeline_name="trading_model_pipeline",
    version="1.0"
)

# Add pipeline steps
pipeline.add_step(
    name="data_collection",
    stage=PipelineStage.DATA_COLLECTION,
    function=collect_data_fn
)

pipeline.add_step(
    name="feature_engineering",
    stage=PipelineStage.FEATURE_ENGINEERING,
    function=engineer_features_fn,
    dependencies=["data_collection"]
)

pipeline.add_step(
    name="model_training",
    stage=PipelineStage.MODEL_TRAINING,
    function=train_model_fn,
    dependencies=["feature_engineering"]
)

# Run pipeline
run = await pipeline.run(config={"model_type": "rl"})

print(f"Pipeline status: {run.status}")
print(f"Duration: {(run.completed_at - run.started_at).total_seconds()}s")
```

## Deployment

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Start services
docker-compose up -d

# Initialize databases
alembic upgrade head

# Run tests
pytest tests/test_ml_models.py -v --cov=app/ml
```

### Production Deployment

```bash
# Set production environment variables
export ENVIRONMENT=production
export POSTGRES_PASSWORD=<secure_password>
export REDIS_PASSWORD=<secure_password>
export GRAFANA_PASSWORD=<secure_password>

# Start all services
docker-compose -f docker-compose.yml up -d

# Check service health
docker-compose ps
curl http://localhost:8000/health
curl http://localhost:5000/health  # MLflow
curl http://localhost:9090/-/healthy  # Prometheus
curl http://localhost:3000/api/health  # Grafana
```

### Service Ports

| Service | Port | Description |
|---------|------|-------------|
| FastAPI | 8000 | Main API |
| MLflow | 5000 | Model registry UI |
| PostgreSQL | 5432 | Database |
| Redis | 6379 | Cache & queue |
| Prometheus | 9090 | Metrics |
| Grafana | 3000 | Dashboards |

## Monitoring & Alerts

### Grafana Dashboard

Access: http://localhost:3000 (admin/admin)

**ML Metrics Dashboard includes**:
- Prediction rate (predictions/sec)
- Prediction latency (P50, P95, P99)
- Model accuracy over time
- Data drift scores
- Cache hit rates
- Model health status

### Prometheus Alerts

**Configured alerts**:
1. **HighPredictionLatency**: P95 > 500ms for 5 minutes
2. **ModelAccuracyDegradation**: Accuracy < 75% for 10 minutes
3. **DataDriftDetected**: Drift score > 0.25 for 5 minutes
4. **LowPredictionRate**: Rate < 1/s for 10 minutes

### Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Track custom metrics
prediction_counter.labels(
    model_id="rl_trading_ppo",
    prediction_type="buy"
).inc()

prediction_latency.labels(
    model_id="rl_trading_ppo"
).observe(0.045)  # 45ms

model_accuracy_gauge.labels(
    model_id="rl_trading_ppo",
    metric_type="accuracy"
).set(0.85)
```

## Testing

### Run All ML Tests

```bash
# Run with coverage
pytest tests/test_ml_models.py -v --cov=app/ml --cov-report=html

# Run specific test class
pytest tests/test_ml_models.py::TestModelRegistry -v

# Run integration tests
pytest tests/test_ml_models.py::TestMLIntegration -v
```

### Test Coverage Goals

- **Target**: 80%+ coverage
- **Critical paths**: 100% coverage
  - Model loading/inference
  - Rollback procedures
  - Drift detection
  - Feature validation

## Best Practices

### Model Deployment Checklist

✅ **Before Deployment**:
1. Model trained with >80% accuracy
2. Validation passed against current production
3. A/B test completed successfully
4. Drift detection configured
5. Rollback plan documented
6. Monitoring alerts configured

✅ **Deployment Process**:
1. Deploy to staging first
2. Run canary deployment (10% traffic)
3. Monitor for 1 hour
4. Gradually increase to 100%
5. Monitor for 24 hours
6. Document deployment in MLflow

✅ **After Deployment**:
1. Monitor accuracy daily
2. Check drift scores weekly
3. Review alerts
4. Plan retraining schedule

### Model Versioning

```
Format: MAJOR.MINOR.PATCH

MAJOR: Breaking changes (new architecture)
MINOR: Backwards-compatible improvements
PATCH: Bug fixes, hyperparameter tuning

Examples:
- 1.0.0: Initial production model
- 1.1.0: Added new features
- 1.1.1: Fixed training bug
- 2.0.0: New model architecture
```

### Emergency Rollback

```python
from app.ml.registry import ModelRegistry

registry = ModelRegistry()

# Rollback to previous version
registry.rollback_model(
    model_name="rl_trading_ppo",
    target_version="1.0"  # or None for previous
)
```

**Rollback SLA**: <5 minutes

## Troubleshooting

### Common Issues

**1. MLflow connection error**
```bash
# Check MLflow is running
docker-compose ps mlflow

# Check logs
docker-compose logs mlflow

# Restart MLflow
docker-compose restart mlflow
```

**2. High prediction latency**
```bash
# Check Redis cache
redis-cli -a <password> INFO stats

# Check Celery workers
celery -A app.ml.inference inspect active

# Check model cache
ls -lh /tmp/model_cache/
```

**3. Drift detection not working**
```bash
# Check baseline is set
redis-cli -a <password> KEYS "baseline:*"

# Check prediction logs
redis-cli -a <password> LRANGE "predictions:model_id" 0 10
```

## Performance Optimization Tips

1. **Enable Redis caching**: 70%+ cache hit rate expected
2. **Use batch inference**: 3-5x throughput improvement
3. **Convert to ONNX**: 2-5x latency reduction
4. **Enable quantization**: 4x model size reduction
5. **Use GPU for batch**: 10-50x speedup (if available)
6. **Warm up models**: Eliminate cold start latency
7. **Monitor and tune**: Use Grafana dashboard

## Contributing

When adding new ML features:

1. Add tests to `tests/test_ml_models.py`
2. Update this documentation
3. Add Prometheus metrics
4. Update Grafana dashboard
5. Document in MLflow

## Support

For MLOps issues:
- Check logs: `docker-compose logs <service>`
- Check metrics: http://localhost:9090
- Check dashboards: http://localhost:3000
- Check MLflow: http://localhost:5000

## License

Part of ShivX AI Trading System - Proprietary
