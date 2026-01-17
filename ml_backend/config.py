"""
UIDAI ML Backend - Unified Configuration

Combines:
- Biometric Re-enrollment Risk Predictor
- ML Fraud Detection Backend

PRIVACY NOTICE:
- This application uses ONLY aggregated, anonymized data from public government APIs
- No individual Aadhaar numbers or personally identifiable information is processed
- All data is at state/district/age-group level only
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from functools import lru_cache

load_dotenv()

# data.gov.in API Configuration
DATA_GOV_API_KEY = os.getenv("DATA_GOV_API_KEY", "579b464db66ec23bdd00000107b605eca2de45d37099f6ba865c6cea")
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys
    data_gov_api_key: str = "579b464db66ec23bdd00000107b605eca2de45d37099f6ba865c6cea"
    gemini_api_key_1: str = ""
    
    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Data.gov.in API Config
    data_gov_base_url: str = "https://api.data.gov.in/resource"
    enrolment_resource_id: str = "ecd49b12-3084-4521-8f7e-ca8bf72069ba"
    demographic_resource_id: str = "19eac040-0b94-49fa-b239-4f2fd8677d53"
    biometric_resource_id: str = "65454dab-1517-40a3-ac1d-47d4dfe6891c"
    
    # ML Config
    isolation_forest_contamination: float = 0.1
    autoencoder_epochs: int = 50
    autoencoder_latent_dim: int = 16
    hdbscan_min_cluster_size: int = 5
    ensemble_weights: dict = {"isolation_forest": 0.4, "autoencoder": 0.35, "hdbscan": 0.25}
    
    # Thresholds
    high_risk_threshold: float = 0.8
    medium_risk_threshold: float = 0.5
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Dataset Resource IDs
DATASETS = {
    "enrolment": {
        "id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "name": "Aadhaar Monthly Enrolment Data",
        "description": "New enrollments per month with age group distributions",
        "resource_id": "ecd49b12-3084-4521-8f7e-ca8bf72069ba",
        "fields": ["date", "state", "district", "pincode", "age_0_5", "age_5_17", "age_18_greater"]
    },
    "demographic": {
        "id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "name": "Aadhaar Demographic Monthly Update Data",
        "description": "Demographic updates with age group distributions",
        "resource_id": "19eac040-0b94-49fa-b239-4f2fd8677d53",
        "fields": ["date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"]
    },
    "biometric": {
        "id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "name": "Aadhaar Biometric Monthly Update Data",
        "description": "Biometric updates (fingerprint/iris) with age groups",
        "resource_id": "65454dab-1517-40a3-ac1d-47d4dfe6891c",
        "fields": ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
    }
}

# High-risk states (historically higher biometric failure rates)
HIGH_RISK_STATES = ["Jharkhand", "Bihar", "Chhattisgarh", "Madhya Pradesh", "Odisha"]

# Model Configuration
MODEL_CONFIG = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "kmeans": {
        "n_clusters": 4,  # Low, Medium, High, Critical risk
        "random_state": 42
    },
    "isolation_forest": {
        "contamination": 0.1,  # Expected proportion of outliers
        "random_state": 42
    }
}

# Risk thresholds
RISK_THRESHOLDS = {
    "low": 0.3,
    "medium": 0.6,
    "high": 0.8
}

# Age buckets for feature engineering
AGE_BUCKETS = {
    "child": (0, 18),
    "adult": (18, 60),
    "elderly": (60, 120)
}

# Output directories
OUTPUT_DIR = "outputs"
VISUALIZATION_DIR = f"{OUTPUT_DIR}/visualizations"
MODEL_DIR = f"{OUTPUT_DIR}/models"
REPORT_DIR = f"{OUTPUT_DIR}/reports"
