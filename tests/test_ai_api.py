"""
AI/ML API Integration Tests
Coverage Target: 100% of app/routers/ai.py

Tests all AI/ML endpoints including models, predictions, training, and explainability
"""

import pytest
from fastapi import status


@pytest.mark.unit
class TestListModels:
    """Test GET /api/ai/models endpoint"""

    def test_list_models_default(self, client, test_token):
        """Test listing all ML models"""
        response = client.get(
            "/api/ai/models",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0

        # Verify model structure
        model = data[0]
        assert "model_id" in model
        assert "name" in model
        assert "version" in model
        assert "type" in model
        assert "status" in model

    def test_list_models_filter_by_status(self, client, test_token):
        """Test filtering models by status"""
        statuses = ["training", "ready", "deployed", "archived"]

        for status_filter in statuses:
            response = client.get(
                f"/api/ai/models?status={status_filter}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            # If any models returned, verify they match filter
            for model in data:
                assert model["status"] == status_filter

    def test_list_models_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/ai/models")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_list_models_readonly_permission(self, client, readonly_token):
        """Test with READ permission"""
        response = client.get(
            "/api/ai/models",
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestGetModel:
    """Test GET /api/ai/models/{model_id} endpoint"""

    def test_get_model_details(self, client, test_token):
        """Test getting specific model details"""
        response = client.get(
            "/api/ai/models/rl_ppo_v1",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["model_id"] == "rl_ppo_v1"
        assert "name" in data
        assert "version" in data
        assert "type" in data
        assert "status" in data
        assert "performance_metrics" in data

    def test_get_model_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/ai/models/test_model")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_model_different_ids(self, client, test_token):
        """Test with different model IDs"""
        model_ids = ["rl_ppo_v1", "lstm_price_v2", "test_model"]

        for model_id in model_ids:
            response = client.get(
                f"/api/ai/models/{model_id}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK


@pytest.mark.unit
class TestPredict:
    """Test POST /api/ai/predict endpoint"""

    def test_predict_basic(self, client, test_token):
        """Test making a prediction"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {
                "rsi": 65.5,
                "macd": 1.25,
                "volume_trend": 0.8
            },
            "explain": False
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "prediction_id" in data
        assert "model_id" in data
        assert data["model_id"] == "rl_ppo_v1"
        assert "prediction" in data
        assert "confidence" in data
        assert "generated_at" in data

    def test_predict_with_explainability(self, client, test_token):
        """Test prediction with explainability enabled"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {
                "rsi": 65.5,
                "macd": 1.25
            },
            "explain": True
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "explanation" in data
        assert data["explanation"] is not None
        assert "important_features" in data["explanation"]

    def test_predict_without_auth(self, client):
        """Test without authentication"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        response = client.post("/api/ai/predict", json=prediction_request)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_predict_insufficient_permission(self, client, readonly_token):
        """Test with READ permission - should fail (needs EXECUTE)"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {readonly_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_predict_invalid_data(self, client, test_token):
        """Test with missing required fields"""
        prediction_request = {
            "model_id": "rl_ppo_v1"
            # Missing features
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_predict_confidence_range(self, client, test_token):
        """Test that confidence is in valid range (0-1)"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert 0 <= data["confidence"] <= 1


@pytest.mark.unit
class TestTrainingJobs:
    """Test GET /api/ai/training-jobs endpoint"""

    def test_list_training_jobs(self, client, test_token):
        """Test listing training jobs"""
        response = client.get(
            "/api/ai/training-jobs",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)

    def test_list_training_jobs_filter_by_status(self, client, test_token):
        """Test filtering training jobs by status"""
        statuses = ["queued", "running", "completed", "failed"]

        for status_filter in statuses:
            response = client.get(
                f"/api/ai/training-jobs?status={status_filter}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code == status.HTTP_200_OK

    def test_list_training_jobs_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/ai/training-jobs")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestStartTraining:
    """Test POST /api/ai/train endpoint"""

    def test_start_training_basic(self, client, admin_token):
        """Test starting a training job"""
        training_config = {
            "model_name": "Test Model v1",
            "model_type": "supervised",
            "dataset_id": "dataset_123",
            "hyperparameters": {
                "hidden_layers": 3,
                "dropout": 0.2
            },
            "epochs": 100,
            "batch_size": 32,
            "learning_rate": 0.001
        }

        response = client.post(
            "/api/ai/train",
            json=training_config,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "job_id" in data
        assert data["model_name"] == "Test Model v1"
        assert data["status"] == "queued"
        assert data["epochs_total"] == 100

    def test_start_training_without_auth(self, client):
        """Test without authentication"""
        training_config = {
            "model_name": "Test",
            "model_type": "rl",
            "dataset_id": "test"
        }

        response = client.post("/api/ai/train", json=training_config)
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_start_training_insufficient_permission(self, client, test_token):
        """Test without ADMIN permission"""
        training_config = {
            "model_name": "Test",
            "model_type": "rl",
            "dataset_id": "test"
        }

        response = client.post(
            "/api/ai/train",
            json=training_config,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN

    def test_start_training_rl_disabled(self, client, admin_token, test_settings):
        """Test starting RL training when feature is disabled"""
        # Disable RL training
        test_settings.feature_rl_trading = False

        training_config = {
            "model_name": "RL Test",
            "model_type": "rl",
            "dataset_id": "test"
        }

        response = client.post(
            "/api/ai/train",
            json=training_config,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert "disabled" in response.json()["detail"].lower()

        # Re-enable
        test_settings.feature_rl_trading = True

    def test_start_training_invalid_data(self, client, admin_token):
        """Test with invalid training config"""
        training_config = {
            "model_name": "Test"
            # Missing required fields
        }

        response = client.post(
            "/api/ai/train",
            json=training_config,
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


@pytest.mark.unit
class TestGetTrainingJob:
    """Test GET /api/ai/training-jobs/{job_id} endpoint"""

    def test_get_training_job_details(self, client, test_token):
        """Test getting training job details"""
        response = client.get(
            "/api/ai/training-jobs/job_123",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "job_id" in data
        assert "model_name" in data
        assert "status" in data
        assert "progress" in data
        assert "epochs_completed" in data
        assert "epochs_total" in data

    def test_get_training_job_progress_range(self, client, test_token):
        """Test that progress is in valid range (0-1)"""
        response = client.get(
            "/api/ai/training-jobs/job_123",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        assert 0 <= data["progress"] <= 1

    def test_get_training_job_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/ai/training-jobs/job_123")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestDeployModel:
    """Test POST /api/ai/models/{model_id}/deploy endpoint"""

    def test_deploy_model(self, client, admin_token):
        """Test deploying a model"""
        response = client.post(
            "/api/ai/models/rl_ppo_v1/deploy",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["model_id"] == "rl_ppo_v1"
        assert data["status"] == "deployed"
        assert "deployed_at" in data

    def test_deploy_model_without_auth(self, client):
        """Test without authentication"""
        response = client.post("/api/ai/models/test/deploy")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_deploy_model_insufficient_permission(self, client, test_token):
        """Test without ADMIN permission"""
        response = client.post(
            "/api/ai/models/test/deploy",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.unit
class TestArchiveModel:
    """Test POST /api/ai/models/{model_id}/archive endpoint"""

    def test_archive_model(self, client, admin_token):
        """Test archiving a model"""
        response = client.post(
            "/api/ai/models/rl_ppo_v1/archive",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert data["model_id"] == "rl_ppo_v1"
        assert data["status"] == "archived"
        assert "archived_at" in data

    def test_archive_model_without_auth(self, client):
        """Test without authentication"""
        response = client.post("/api/ai/models/test/archive")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_archive_model_insufficient_permission(self, client, test_token):
        """Test without ADMIN permission"""
        response = client.post(
            "/api/ai/models/test/archive",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_403_FORBIDDEN


@pytest.mark.unit
class TestExplainability:
    """Test GET /api/ai/explainability/{prediction_id} endpoint"""

    def test_get_explainability(self, client, test_token):
        """Test getting explainability for a prediction"""
        response = client.get(
            "/api/ai/explainability/pred_123",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "prediction_id" in data
        assert "method" in data
        assert "feature_importance" in data
        assert "counterfactual" in data
        assert "confidence_breakdown" in data

    def test_explainability_feature_importance_structure(self, client, test_token):
        """Test feature importance structure"""
        response = client.get(
            "/api/ai/explainability/pred_123",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        data = response.json()
        features = data["feature_importance"]

        assert isinstance(features, list)
        if len(features) > 0:
            feature = features[0]
            assert "feature" in feature
            assert "importance" in feature
            assert "direction" in feature

    def test_explainability_without_auth(self, client):
        """Test without authentication"""
        response = client.get("/api/ai/explainability/pred_123")
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


@pytest.mark.unit
class TestAICapabilities:
    """Test GET /api/ai/capabilities endpoint"""

    def test_get_ai_capabilities(self, client):
        """Test getting AI capabilities - public endpoint"""
        response = client.get("/api/ai/capabilities")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()

        assert "rl_trading" in data
        assert "sentiment_analysis" in data
        assert "dex_arbitrage" in data
        assert "metacognition" in data
        assert "explainability" in data
        assert "available_models" in data

    def test_ai_capabilities_with_auth(self, client, test_token):
        """Test that endpoint works with auth too"""
        response = client.get(
            "/api/ai/capabilities",
            headers={"Authorization": f"Bearer {test_token}"}
        )

        assert response.status_code == status.HTTP_200_OK

    def test_ai_capabilities_available_models_is_list(self, client):
        """Test available models is a list"""
        response = client.get("/api/ai/capabilities")
        data = response.json()

        assert isinstance(data["available_models"], list)


@pytest.mark.integration
class TestAIWorkflows:
    """Integration tests for AI workflows"""

    def test_complete_ml_workflow(self, client, admin_token, test_token):
        """
        Test complete ML workflow:
        1. List models
        2. Make prediction
        3. Get explainability
        4. Start training
        5. Check training status
        """
        # 1. List models
        models_response = client.get(
            "/api/ai/models",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert models_response.status_code == status.HTTP_200_OK

        # 2. Make prediction
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5},
            "explain": True
        }
        pred_response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert pred_response.status_code == status.HTTP_200_OK
        prediction_id = pred_response.json()["prediction_id"]

        # 3. Get explainability
        explain_response = client.get(
            f"/api/ai/explainability/{prediction_id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert explain_response.status_code == status.HTTP_200_OK

        # 4. Start training
        training_config = {
            "model_name": "New Model",
            "model_type": "supervised",
            "dataset_id": "test",
            "epochs": 10
        }
        train_response = client.post(
            "/api/ai/train",
            json=training_config,
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert train_response.status_code == status.HTTP_200_OK
        job_id = train_response.json()["job_id"]

        # 5. Check training status
        job_response = client.get(
            f"/api/ai/training-jobs/{job_id}",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert job_response.status_code == status.HTTP_200_OK

    def test_model_lifecycle(self, client, admin_token):
        """Test model deployment and archival"""
        model_id = "test_model_v1"

        # Deploy
        deploy_response = client.post(
            f"/api/ai/models/{model_id}/deploy",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert deploy_response.status_code == status.HTTP_200_OK

        # Archive
        archive_response = client.post(
            f"/api/ai/models/{model_id}/archive",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert archive_response.status_code == status.HTTP_200_OK


@pytest.mark.security
class TestAISecurity:
    """Security tests for AI API"""

    def test_injection_in_model_id(self, client, test_token):
        """Test injection attempts in model_id"""
        malicious_ids = [
            "'; DROP TABLE--",
            "../../../etc/passwd",
            "<script>alert('xss')</script>"
        ]

        for malicious_id in malicious_ids:
            response = client.get(
                f"/api/ai/models/{malicious_id}",
                headers={"Authorization": f"Bearer {test_token}"}
            )

            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_malicious_features(self, client, test_token):
        """Test with malicious feature values"""
        prediction_request = {
            "model_id": "test",
            "features": {
                "malicious": "'; DROP TABLE--",
                "xss": "<script>alert('xss')</script>"
            }
        }

        response = client.post(
            "/api/ai/predict",
            json=prediction_request,
            headers={"Authorization": f"Bearer {test_token}"}
        )

        # Should handle safely
        assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


@pytest.mark.performance
class TestAIPerformance:
    """Performance tests for AI API"""

    def test_prediction_latency(self, client, test_token, benchmark):
        """Benchmark prediction latency"""
        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        def make_prediction():
            response = client.post(
                "/api/ai/predict",
                json=prediction_request,
                headers={"Authorization": f"Bearer {test_token}"}
            )
            assert response.status_code == status.HTTP_200_OK
            return response

        result = benchmark(make_prediction)
        assert result.status_code == status.HTTP_200_OK

    @pytest.mark.slow
    def test_concurrent_predictions(self, client, test_token):
        """Test concurrent prediction requests"""
        import concurrent.futures

        prediction_request = {
            "model_id": "rl_ppo_v1",
            "features": {"rsi": 65.5}
        }

        def make_request(_):
            return client.post(
                "/api/ai/predict",
                json=prediction_request,
                headers={"Authorization": f"Bearer {test_token}"}
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(r.status_code == status.HTTP_200_OK for r in results)
