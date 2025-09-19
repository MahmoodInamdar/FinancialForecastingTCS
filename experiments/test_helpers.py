"""
Test Helpers for Financial Forecasting Agent Experiments

This module provides common testing utilities and helper functions
used across all experimental notebooks for the TCS Financial Forecasting Agent.
"""

import os
import sys
import json
import time
import logging
import hashlib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import Mock, patch


def setup_test_environment() -> Dict[str, Any]:
    """
    Set up common test environment for all experiments.

    Returns:
        Dict containing environment configuration
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Project paths
    project_root = Path(__file__).parent.parent
    data_folder = project_root / "data"
    output_folder = project_root / "experiments" / "outputs"

    # Create output directory if it doesn't exist
    output_folder.mkdir(exist_ok=True)

    # Environment configuration
    env_config = {
        "project_root": project_root,
        "data_folder": data_folder,
        "output_folder": output_folder,
        "timestamp": datetime.now().isoformat(),
        "test_session_id": generate_session_id()
    }

    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    return env_config


def generate_session_id() -> str:
    """Generate unique session ID for test tracking."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(time.time()).encode()).hexdigest()[:6]
    return f"test_{timestamp}_{random_suffix}"


class TestTimer:
    """Context manager for timing test operations."""

    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting: {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.logger.info(f"Completed: {self.operation_name} in {self.duration:.2f}s")


class MockDataGenerator:
    """Generate mock data for testing purposes."""

    @staticmethod
    def create_mock_financial_data(
        company: str = "TCS",
        periods: int = 4,
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate mock financial data for testing.

        Args:
            company: Company name
            periods: Number of periods to generate
            metrics: List of financial metrics to include

        Returns:
            DataFrame with mock financial data
        """
        if metrics is None:
            metrics = [
                "Revenue", "Net Income", "Total Assets", "Total Equity",
                "Operating Income", "Cash Flow", "Debt", "EBITDA"
            ]

        # Generate realistic financial values
        base_values = {
            "Revenue": 50000,
            "Net Income": 8000,
            "Total Assets": 75000,
            "Total Equity": 45000,
            "Operating Income": 12000,
            "Cash Flow": 10000,
            "Debt": 15000,
            "EBITDA": 15000
        }

        data = []
        for i in range(periods):
            quarter = f"Q{i+1} 2024"
            growth_factor = 1 + (i * 0.05)  # 5% growth per quarter

            period_data = {"Company": company, "Period": quarter}
            for metric in metrics:
                base_value = base_values.get(metric, 1000)
                # Add some randomness
                variance = np.random.normal(1, 0.1)
                value = base_value * growth_factor * variance
                period_data[metric] = round(value, 2)

            data.append(period_data)

        return pd.DataFrame(data)

    @staticmethod
    def create_mock_document_metadata(
        document_count: int = 5,
        doc_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate mock document metadata for testing.

        Args:
            document_count: Number of documents to simulate
            doc_types: List of document types

        Returns:
            List of mock document metadata
        """
        if doc_types is None:
            doc_types = [
                "quarterly_report", "annual_report", "earnings_call",
                "investor_presentation", "sec_filing"
            ]

        documents = []
        for i in range(document_count):
            doc_type = doc_types[i % len(doc_types)]

            metadata = {
                "document_id": f"doc_{i+1:03d}",
                "document_type": doc_type,
                "company": "TCS",
                "period": f"Q{(i % 4) + 1} 2024",
                "file_name": f"tcs_{doc_type}_{i+1}.pdf",
                "file_size": np.random.randint(1000000, 10000000),  # 1-10MB
                "page_count": np.random.randint(10, 100),
                "created_date": (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat(),
                "contains_tables": np.random.choice([True, False], p=[0.8, 0.2]),
                "language": "english",
                "confidence_score": np.random.uniform(0.85, 0.99)
            }

            documents.append(metadata)

        return documents

    @staticmethod
    def create_mock_table_data(
        table_count: int = 3,
        rows_per_table: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Generate mock extracted table data.

        Args:
            table_count: Number of tables to generate
            rows_per_table: Number of rows per table

        Returns:
            List of mock table data
        """
        tables = []

        financial_headers = [
            ["Metric", "Q1 2024", "Q2 2024", "Q3 2024", "Q4 2024"],
            ["Item", "Current Year", "Previous Year", "Change %"],
            ["Category", "Amount", "Percentage", "Notes"]
        ]

        for i in range(table_count):
            headers = financial_headers[i % len(financial_headers)]
            rows = []

            # Generate table rows
            for j in range(rows_per_table):
                row = []
                for k, header in enumerate(headers):
                    if k == 0:  # First column - text
                        row.append(f"Item_{j+1}")
                    elif "%" in header or "Percentage" in header:
                        row.append(f"{np.random.uniform(5, 25):.1f}%")
                    elif "Notes" in header:
                        row.append("Sample note")
                    else:  # Numeric columns
                        row.append(f"{np.random.uniform(1000, 50000):.2f}")
                rows.append(row)

            table_data = {
                "table_id": f"table_{i+1}",
                "headers": headers,
                "rows": rows,
                "extraction_method": ["qwen", "layoutlm", "kosmos"][i % 3],
                "confidence_score": np.random.uniform(0.8, 0.95),
                "table_type": ["financial_metrics", "income_statement", "balance_sheet"][i % 3],
                "page_number": i + 1
            }

            tables.append(table_data)

        return tables


class TestDataValidator:
    """Validate test data and results."""

    @staticmethod
    def validate_financial_data(data: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate financial data structure and content.

        Args:
            data: Financial data DataFrame

        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        # Check required columns
        required_columns = ["Company", "Period"]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            validation_results["errors"].append(f"Missing required columns: {missing_columns}")
            validation_results["is_valid"] = False

        # Check for empty data
        if data.empty:
            validation_results["errors"].append("DataFrame is empty")
            validation_results["is_valid"] = False

        # Check for numeric columns
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_columns) == 0:
            validation_results["warnings"].append("No numeric columns found")

        # Calculate validation metrics
        validation_results["metrics"] = {
            "row_count": len(data),
            "column_count": len(data.columns),
            "numeric_columns": len(numeric_columns),
            "null_values": data.isnull().sum().sum(),
            "duplicate_rows": data.duplicated().sum()
        }

        return validation_results

    @staticmethod
    def validate_extracted_tables(tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate extracted table data.

        Args:
            tables: List of extracted table dictionaries

        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        if not tables:
            validation_results["errors"].append("No tables provided")
            validation_results["is_valid"] = False
            return validation_results

        # Check required fields
        required_fields = ["table_id", "headers", "rows"]
        for i, table in enumerate(tables):
            missing_fields = [field for field in required_fields if field not in table]
            if missing_fields:
                validation_results["errors"].append(
                    f"Table {i}: Missing required fields: {missing_fields}"
                )
                validation_results["is_valid"] = False

        # Calculate metrics
        total_rows = sum(len(table.get("rows", [])) for table in tables)
        avg_confidence = np.mean([
            table.get("confidence_score", 0) for table in tables
            if "confidence_score" in table
        ])

        validation_results["metrics"] = {
            "table_count": len(tables),
            "total_rows": total_rows,
            "average_confidence": avg_confidence,
            "tables_with_confidence": sum(1 for table in tables if "confidence_score" in table)
        }

        return validation_results


class TestResultCollector:
    """Collect and manage test results across experiments."""

    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        self.session_id = generate_session_id()
        self.results = {
            "experiment_name": experiment_name,
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "test_cases": {},
            "performance_metrics": {},
            "errors": [],
            "summary": {}
        }

    def add_test_case(self, test_name: str, result: Dict[str, Any]):
        """Add test case result."""
        self.results["test_cases"][test_name] = {
            "timestamp": datetime.now().isoformat(),
            "result": result
        }

    def add_performance_metric(self, metric_name: str, value: Union[float, int], unit: str = ""):
        """Add performance metric."""
        self.results["performance_metrics"][metric_name] = {
            "value": value,
            "unit": unit,
            "timestamp": datetime.now().isoformat()
        }

    def add_error(self, error: Exception, context: str = ""):
        """Add error information."""
        error_info = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.results["errors"].append(error_info)

    def finalize_results(self) -> Dict[str, Any]:
        """Finalize and return complete results."""
        self.results["end_time"] = datetime.now().isoformat()

        # Calculate summary statistics
        total_tests = len(self.results["test_cases"])
        passed_tests = sum(
            1 for test in self.results["test_cases"].values()
            if test["result"].get("status") == "passed"
        )

        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "total_errors": len(self.results["errors"]),
            "performance_metrics_count": len(self.results["performance_metrics"])
        }

        return self.results

    def save_results(self, output_folder: Path) -> Path:
        """Save results to file."""
        filename = f"{self.experiment_name}_{self.session_id}_results.json"
        filepath = output_folder / filename

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        return filepath


def create_test_assertion(
    condition: bool,
    error_message: str,
    test_name: str = ""
) -> Dict[str, Any]:
    """
    Create standardized test assertion result.

    Args:
        condition: Test condition result
        error_message: Error message if condition fails
        test_name: Name of the test

    Returns:
        Test result dictionary
    """
    return {
        "test_name": test_name,
        "status": "passed" if condition else "failed",
        "timestamp": datetime.now().isoformat(),
        "error_message": error_message if not condition else None
    }


def benchmark_function(
    func: Callable,
    *args,
    iterations: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """
    Benchmark function execution time and performance.

    Args:
        func: Function to benchmark
        *args: Function arguments
        iterations: Number of iterations to run
        **kwargs: Function keyword arguments

    Returns:
        Benchmark results
    """
    execution_times = []
    results = []
    errors = []

    for i in range(iterations):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            results.append(result)
        except Exception as e:
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            errors.append(str(e))

    return {
        "function_name": func.__name__,
        "iterations": iterations,
        "successful_runs": len(results),
        "failed_runs": len(errors),
        "avg_execution_time": np.mean(execution_times),
        "min_execution_time": np.min(execution_times),
        "max_execution_time": np.max(execution_times),
        "std_execution_time": np.std(execution_times),
        "total_execution_time": np.sum(execution_times),
        "errors": errors[:5],  # Keep only first 5 errors
        "last_successful_result": results[-1] if results else None
    }


def create_mock_api_response(
    status_code: int = 200,
    response_data: Optional[Dict] = None,
    delay: float = 0
) -> Mock:
    """
    Create mock API response for testing.

    Args:
        status_code: HTTP status code
        response_data: Response data
        delay: Simulated delay in seconds

    Returns:
        Mock response object
    """
    if response_data is None:
        response_data = {"status": "success", "data": {}}

    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)

    # Add delay if specified
    if delay > 0:
        original_json = mock_response.json
        def delayed_json():
            time.sleep(delay)
            return original_json()
        mock_response.json = delayed_json

    return mock_response


# Convenience functions for common test scenarios
def assert_dataframe_not_empty(df: pd.DataFrame, test_name: str = "") -> Dict[str, Any]:
    """Assert that DataFrame is not empty."""
    return create_test_assertion(
        condition=not df.empty,
        error_message=f"DataFrame is empty",
        test_name=test_name
    )


def assert_columns_exist(df: pd.DataFrame, required_columns: List[str], test_name: str = "") -> Dict[str, Any]:
    """Assert that required columns exist in DataFrame."""
    missing_columns = [col for col in required_columns if col not in df.columns]
    return create_test_assertion(
        condition=len(missing_columns) == 0,
        error_message=f"Missing required columns: {missing_columns}",
        test_name=test_name
    )


def assert_performance_threshold(
    actual_time: float,
    threshold: float,
    test_name: str = ""
) -> Dict[str, Any]:
    """Assert that performance is within threshold."""
    return create_test_assertion(
        condition=actual_time <= threshold,
        error_message=f"Performance {actual_time:.2f}s exceeds threshold {threshold}s",
        test_name=test_name
    )


# Global test configuration
DEFAULT_TEST_CONFIG = {
    "timeouts": {
        "api_call": 30,
        "model_inference": 60,
        "data_processing": 45,
        "file_operation": 15
    },
    "thresholds": {
        "accuracy": 0.85,
        "precision": 0.80,
        "recall": 0.75,
        "f1_score": 0.80
    },
    "mock_data": {
        "default_company": "TCS",
        "default_periods": 4,
        "default_documents": 5
    }
}


if __name__ == "__main__":
    # Example usage
    print("Test Helpers Module")
    print("==================")

    # Setup environment
    env = setup_test_environment()
    print(f"Project Root: {env['project_root']}")
    print(f"Session ID: {env['test_session_id']}")

    # Generate sample data
    mock_data = MockDataGenerator.create_mock_financial_data()
    print(f"\nMock Financial Data Shape: {mock_data.shape}")
    print(mock_data.head())

    # Validate data
    validation = TestDataValidator.validate_financial_data(mock_data)
    print(f"\nValidation Results: {validation['is_valid']}")
    print(f"Metrics: {validation['metrics']}")