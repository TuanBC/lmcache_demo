"""
Test configuration and fixtures.
"""

import os

import pytest
from dotenv import load_dotenv


@pytest.fixture(autouse=True, scope="session")
def set_test_environment():
    """Load environment variables from .env file for tests.
    
    Uses the same dotenv loading pattern as src/config/settings.py
    to ensure consistency between application and tests.
    """
    # Load .env file (same as settings.py does)
    load_dotenv()
    
    # Fallback for VLLM_MODEL if not set in .env (needed for prompty templates)
    os.environ.setdefault("VLLM_MODEL", "test-model")
    yield


@pytest.fixture
def sample_query() -> str:
    """Sample user query for testing."""
    return "What is the daily ATM withdrawal limit for consumer accounts?"


@pytest.fixture
def sample_manual() -> str:
    """Short manual excerpt for testing."""
    return """# Internal Operations Manual

## Section 3: Transaction Processing

### 3.1 Transaction Limits

#### 3.1.1 Daily Limits

| Transaction Type | Consumer Account | Business Account |
|------------------|------------------|------------------|
| ATM Withdrawal | $500 | $1,000 |
| Point of Sale | $2,500 | $5,000 |
| Online Transfer | $5,000 | $25,000 |
"""
