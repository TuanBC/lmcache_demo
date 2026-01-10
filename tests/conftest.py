"""
Test configuration and fixtures.
"""

import pytest


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
