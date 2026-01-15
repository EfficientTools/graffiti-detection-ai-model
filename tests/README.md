# Graffiti Detection AI - Unit Tests

## Overview

Comprehensive unit tests for the graffiti detection system covering:
- Dataset loading and preprocessing
- Data augmentation
- Evaluation metrics
- Alert system
- Visualization utilities
- Incident logging
- System integration

## Running Tests

### Run All Tests
```bash
# Using unittest
python tests/run_tests.py

# Using pytest (if installed)
pytest tests/

# With coverage report
pytest --cov=src --cov=scripts tests/
```

### Run Specific Test Modules
```bash
# Test dataset functionality
python -m unittest tests.test_dataset

# Test augmentation
python -m unittest tests.test_augmentation

# Test metrics
python -m unittest tests.test_metrics

# Test alerts
python -m unittest tests.test_alerts

# Test visualization
python -m unittest tests.test_visualization

# Test incident logger
python -m unittest tests.test_incident_logger

# Integration tests
python -m unittest tests.test_integration
```

### Run Specific Test Classes or Methods
```bash
# Run specific test class
python -m unittest tests.test_dataset.TestGraffitiDataset

# Run specific test method
python -m unittest tests.test_dataset.TestGraffitiDataset.test_dataset_creation
```

## Test Coverage

### Current Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| `src.data.dataset` | 3 tests | Dataset loading, preprocessing |
| `src.data.augmentation` | 4 tests | Augmentation pipelines |
| `src.evaluation.metrics` | 5 tests | IoU, AP, mAP calculations |
| `src.utils.alerts` | 4 tests | Alert system components |
| `src.utils.visualization` | 4 tests | Drawing and visualization |
| `scripts.incident_logger` | 7 tests | Incident logging system |
| Integration | 5 tests | End-to-end workflows |

### Generate Coverage Report
```bash
# Install coverage
pip install coverage pytest-cov

# Run with coverage
coverage run -m pytest tests/

# Generate HTML report
coverage html

# View report
open htmlcov/index.html
```

## Test Structure

```
tests/
├── __init__.py                 # Test package init
├── test_dataset.py             # Dataset tests
├── test_augmentation.py        # Augmentation tests
├── test_metrics.py             # Metrics tests
├── test_alerts.py              # Alert system tests
├── test_visualization.py       # Visualization tests
├── test_incident_logger.py     # Incident logger tests
├── test_integration.py         # Integration tests
├── run_tests.py                # Test runner
└── README.md                   # This file
```

## Writing New Tests

### Test Template
```python
import unittest

class TestNewFeature(unittest.TestCase):
    """Test new feature description"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Initialize test data
        pass
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up resources
        pass
    
    def test_feature_basic(self):
        """Test basic functionality"""
        # Arrange
        # Act
        # Assert
        pass
    
    def test_feature_edge_case(self):
        """Test edge cases"""
        pass

if __name__ == '__main__':
    unittest.main()
```

### Best Practices

1. **Test Organization**
   - One test file per module
   - Group related tests in classes
   - Use descriptive test names

2. **Test Independence**
   - Tests should not depend on each other
   - Use setUp/tearDown for test fixtures
   - Clean up resources after tests

3. **Assertions**
   - Use specific assertions (assertEqual, assertIsNotNone, etc.)
   - Test both success and failure cases
   - Include edge cases

4. **Mocking**
   - Mock external dependencies (API calls, file I/O)
   - Use unittest.mock for isolation
   - Test error handling

5. **Documentation**
   - Document test purpose in docstrings
   - Explain complex test scenarios
   - Keep tests readable

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov=scripts
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure project root is in PYTHONPATH
   - Run tests from project root directory

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **GPU Tests**
   - GPU tests are marked with `@pytest.mark.gpu`
   - Skip GPU tests if no GPU available:
     ```bash
     pytest -m "not gpu"
     ```

4. **Temporary Files**
   - Tests use `tempfile` for file operations
   - Automatically cleaned up after tests

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure all tests pass
3. Maintain >80% code coverage
4. Update this README with new test info

## Resources

- [unittest documentation](https://docs.python.org/3/library/unittest.html)
- [pytest documentation](https://docs.pytest.org/)
- [coverage.py documentation](https://coverage.readthedocs.io/)
