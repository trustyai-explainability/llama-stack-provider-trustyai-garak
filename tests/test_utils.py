"""Tests for utility functions in utils.py"""

import pytest
import os
from pathlib import Path
from unittest.mock import patch


class TestXDGFunctions:
    """Test cases for XDG environment variable handling"""

    def test_ensure_xdg_vars_sets_defaults(self, monkeypatch):
        """Test that _ensure_xdg_vars sets default XDG variables when missing"""
        # Clear existing XDG variables
        for var in ['XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME']:
            monkeypatch.delenv(var, raising=False)
        
        from llama_stack_provider_trustyai_garak.utils import _ensure_xdg_vars
        
        # Call the function (now lazy, not called at import)
        _ensure_xdg_vars()
        
        # Check that XDG variables are set to /tmp defaults
        assert os.environ.get('XDG_CACHE_HOME') == '/tmp/.cache'
        assert os.environ.get('XDG_DATA_HOME') == '/tmp/.local/share'
        assert os.environ.get('XDG_CONFIG_HOME') == '/tmp/.config'

    def test_ensure_xdg_vars_preserves_existing(self, monkeypatch, tmp_path):
        """Test that _ensure_xdg_vars preserves existing XDG variables"""
        # Set custom XDG variables before loading
        custom_cache = tmp_path / "my_cache"
        custom_cache.mkdir(parents=True, exist_ok=True)
        
        monkeypatch.setenv('XDG_CACHE_HOME', str(custom_cache))
        monkeypatch.setenv('XDG_DATA_HOME', '/custom/data')
        monkeypatch.setenv('XDG_CONFIG_HOME', '/custom/config')
        
        # Reload utils
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        # Should preserve existing values
        assert os.environ.get('XDG_CACHE_HOME') == str(custom_cache)
        assert os.environ.get('XDG_DATA_HOME') == '/custom/data'
        assert os.environ.get('XDG_CONFIG_HOME') == '/custom/config'

    def test_ensure_xdg_vars_is_idempotent(self, monkeypatch):
        """Test that _ensure_xdg_vars can be called multiple times safely"""
        # Clear XDG vars
        for var in ['XDG_CACHE_HOME', 'XDG_DATA_HOME', 'XDG_CONFIG_HOME']:
            monkeypatch.delenv(var, raising=False)
        
        from llama_stack_provider_trustyai_garak.utils import _ensure_xdg_vars
        
        # Call it multiple times
        _ensure_xdg_vars()
        first_cache = os.environ.get('XDG_CACHE_HOME')
        
        _ensure_xdg_vars()
        second_cache = os.environ.get('XDG_CACHE_HOME')
        
        # Should be the same (idempotent)
        assert first_cache == second_cache == '/tmp/.cache'

    def test_get_scan_base_dir_with_garak_scan_dir(self, monkeypatch, tmp_path):
        """Test that get_scan_base_dir respects GARAK_SCAN_DIR environment variable"""
        custom_dir = tmp_path / "my_scans"
        custom_dir.mkdir(parents=True, exist_ok=True)
        
        monkeypatch.setenv("GARAK_SCAN_DIR", str(custom_dir))
        
        # Reload to pick up env var
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        scan_dir = utils.get_scan_base_dir()
        
        assert scan_dir == custom_dir
        assert str(scan_dir) == str(custom_dir)

    def test_get_scan_base_dir_uses_xdg_cache_home(self, monkeypatch, tmp_path):
        """Test that get_scan_base_dir falls back to XDG_CACHE_HOME when no override"""
        # Clear GARAK_SCAN_DIR
        monkeypatch.delenv("GARAK_SCAN_DIR", raising=False)
        
        # Set custom XDG_CACHE_HOME
        custom_cache = tmp_path / "cache"
        custom_cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
        
        # Reload to pick up env vars
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        scan_dir = utils.get_scan_base_dir()
        
        # Should be XDG_CACHE_HOME/llama_stack_garak_scans
        expected = custom_cache / "llama_stack_garak_scans"
        assert scan_dir == expected

    def test_get_scan_base_dir_default_tmp_cache(self, monkeypatch):
        """Test that get_scan_base_dir defaults to /tmp/.cache/llama_stack_garak_scans"""
        # Clear both GARAK_SCAN_DIR and XDG_CACHE_HOME
        monkeypatch.delenv("GARAK_SCAN_DIR", raising=False)
        monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
        
        # Reload to trigger default XDG setup
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        scan_dir = utils.get_scan_base_dir()
        
        # Should default to /tmp/.cache/llama_stack_garak_scans
        assert str(scan_dir) == "/tmp/.cache/llama_stack_garak_scans"


class TestHTTPClientWithTLS:
    """Test cases for HTTP client TLS verification"""

    def test_get_http_client_with_tls_true(self):
        """Test HTTP client with TLS verification enabled"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
            get_http_client_with_tls(True)
            mock_client.assert_called_once_with(verify=True)

    def test_get_http_client_with_tls_false(self):
        """Test HTTP client with TLS verification disabled"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
            get_http_client_with_tls(False)
            mock_client.assert_called_once_with(verify=False)

    def test_get_http_client_with_tls_string_true(self):
        """Test HTTP client with TLS string 'true'"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        for value in ["true", "True", "TRUE", "1", "yes", "on"]:
            with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
                get_http_client_with_tls(value)
                mock_client.assert_called_once_with(verify=True)

    def test_get_http_client_with_tls_string_false(self):
        """Test HTTP client with TLS string 'false'"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        for value in ["false", "False", "FALSE", "0", "no", "off"]:
            with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
                get_http_client_with_tls(value)
                mock_client.assert_called_once_with(verify=False)

    def test_get_http_client_with_tls_cert_path(self):
        """Test HTTP client with certificate path"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        cert_path = "/path/to/cert.pem"
        
        with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
            get_http_client_with_tls(cert_path)
            mock_client.assert_called_once_with(verify=cert_path)

    def test_get_http_client_with_tls_none(self):
        """Test HTTP client with None defaults to True"""
        from llama_stack_provider_trustyai_garak.utils import get_http_client_with_tls
        
        with patch('llama_stack_provider_trustyai_garak.utils.httpx.Client') as mock_client:
            get_http_client_with_tls(None)
            mock_client.assert_called_once_with(verify=True)

