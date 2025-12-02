"""Shared test fixtures and configuration"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def mock_file_api():
    """Mock Files API"""
    mock = Mock()
    mock.upload_file = AsyncMock()
    mock.get_file_content = AsyncMock()
    return mock


@pytest.fixture
def mock_benchmarks_api():
    """Mock Benchmarks API"""
    mock = Mock()
    mock.list_benchmarks = AsyncMock(return_value=[])
    mock.get_benchmark = AsyncMock()
    return mock


@pytest.fixture
def mock_safety_api():
    """Mock Safety API"""
    mock = Mock()
    mock.run_shield = AsyncMock()
    return mock


@pytest.fixture
def mock_shields_api():
    """Mock Shields API"""
    mock = Mock()
    mock.list_shields = AsyncMock(return_value=[])
    mock.get_shield = AsyncMock()
    return mock


@pytest.fixture
def mock_deps(mock_file_api, mock_benchmarks_api):
    """Standard mock dependencies"""
    from llama_stack_provider_trustyai_garak.compat import Api
    
    return {
        Api.files: mock_file_api,
        Api.benchmarks: mock_benchmarks_api
    }


@pytest.fixture
def mock_deps_with_safety(mock_deps, mock_safety_api, mock_shields_api):
    """Mock dependencies including safety and shields"""
    from llama_stack_provider_trustyai_garak.compat import Api
    
    mock_deps[Api.safety] = mock_safety_api
    mock_deps[Api.shields] = mock_shields_api
    return mock_deps


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()