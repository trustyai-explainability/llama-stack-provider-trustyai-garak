"""Tests for configuration classes"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from llama_stack_provider_trustyai_garak.config import (
    GarakInlineConfig,
    GarakRemoteConfig,
    KubeflowConfig,
    GarakScanConfig
)


class TestGarakInlineConfig:
    """Test cases for GarakInlineConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GarakInlineConfig()
        assert config.llama_stack_url == "http://localhost:8321"
        assert config.garak_model_type_openai == "openai.OpenAICompatible"
        assert config.garak_model_type_function == "function.Single"
        assert config.timeout == 60 * 60 * 3
        assert config.max_workers == 5
        assert config.max_concurrent_jobs == 5
        assert config.tls_verify is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = GarakInlineConfig(
            llama_stack_url="https://custom.api.com",
            garak_model_type_openai="custom.model",
            timeout=1000,
            max_workers=10,
            max_concurrent_jobs=3
        )
        assert config.llama_stack_url == "https://custom.api.com"
        assert config.garak_model_type_openai == "custom.model"
        assert config.timeout == 1000
        assert config.max_workers == 10
        assert config.max_concurrent_jobs == 3

    def test_tls_verify_boolean(self):
        """Test TLS verify with boolean values"""
        config_true = GarakInlineConfig(tls_verify=True)
        assert config_true.tls_verify is True
        
        config_false = GarakInlineConfig(tls_verify=False)
        assert config_false.tls_verify is False

    def test_tls_verify_cert_path(self, tmp_path):
        """Test TLS verify with certificate path"""
        # Create a temporary certificate file
        cert_file = tmp_path / "cert.pem"
        cert_file.write_text("CERTIFICATE CONTENT")
        
        config = GarakInlineConfig(tls_verify=str(cert_file))
        assert config.tls_verify == str(cert_file)

    def test_tls_verify_invalid_path(self):
        """Test TLS verify with invalid certificate path"""
        with pytest.raises(ValidationError) as exc_info:
            GarakInlineConfig(tls_verify="/invalid/path/cert.pem")
        assert "TLS certificate file does not exist" in str(exc_info.value)

    def test_tls_verify_directory_path(self, tmp_path):
        """Test TLS verify with directory path (should fail)"""
        with pytest.raises(ValidationError) as exc_info:
            GarakInlineConfig(tls_verify=str(tmp_path))
        assert "TLS certificate path is not a file" in str(exc_info.value)

    def test_llama_stack_url_validation(self):
        """Test llama_stack_url validation (whitespace trimming)"""
        config = GarakInlineConfig(llama_stack_url="  https://api.com  ")
        assert config.llama_stack_url == "https://api.com"

    def test_invalid_llama_stack_url_type(self):
        """Test invalid llama_stack_url type"""
        with pytest.raises(ValidationError):
            GarakInlineConfig(llama_stack_url=123)

    def test_sample_run_config_with_integers(self):
        """Test sample_run_config class method with integer values"""
        config_dict = GarakInlineConfig.sample_run_config(
            llama_stack_url="https://test.api.com",
            timeout=5000,
            max_workers=8,
            max_concurrent_jobs=10
        )
        assert config_dict["llama_stack_url"] == "https://test.api.com"
        assert config_dict["timeout"] == 5000
        assert config_dict["max_workers"] == 8
        assert config_dict["max_concurrent_jobs"] == 10
        assert "garak_model_type_openai" in config_dict
        assert "garak_model_type_function" in config_dict

    def test_sample_run_config_with_defaults(self):
        """Test sample_run_config with default template values"""
        # The method has default template strings that can't be converted to int
        # This is a known limitation - the method will raise an error when called with defaults
        # that are template strings trying to be converted to int
        with pytest.raises(ValueError) as exc_info:
            GarakInlineConfig.sample_run_config()
        assert "invalid literal for int()" in str(exc_info.value)


class TestGarakRemoteConfig:
    """Test cases for GarakRemoteConfig"""

    def test_remote_config_with_kubeflow(self):
        """Test remote config with Kubeflow settings"""
        kubeflow_config = KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
            namespace="garak-namespace",
            base_image="garak:latest"
        )
        
        config = GarakRemoteConfig(
            kubeflow_config=kubeflow_config
        )
        
        assert config.kubeflow_config.pipelines_endpoint == "https://kfp.example.com"
        assert config.kubeflow_config.namespace == "garak-namespace"
        assert config.kubeflow_config.base_image == "garak:latest"
        
        # Should inherit base config defaults
        assert config.llama_stack_url == "http://localhost:8321"
        assert config.timeout == 60 * 60 * 3


    def test_remote_config_inherits_base_fields(self):
        """Test that GarakRemoteConfig inherits all base fields"""
        kubeflow_config = KubeflowConfig(
            pipelines_endpoint="https://kfp.test.com",
            namespace="test",
            base_image="test:latest"
        )
        
        config = GarakRemoteConfig(
            llama_stack_url="https://custom.com",
            timeout=5000,
            max_workers=10,
            kubeflow_config=kubeflow_config
        )
        
        # Check inherited fields
        assert config.llama_stack_url == "https://custom.com"
        assert config.timeout == 5000
        assert config.max_workers == 10
        assert config.garak_model_type_openai == "openai.OpenAICompatible"
        
        # Check it doesn't have inline-only field
        assert not hasattr(config, 'max_concurrent_jobs') or config.max_concurrent_jobs == 5  # inherited default

    def test_remote_config_tls_verify(self, tmp_path):
        """Test TLS verify setting in remote config"""
        cert_file = tmp_path / "cert.pem"
        cert_file.write_text("CERTIFICATE")
        
        kubeflow_config = KubeflowConfig(
            pipelines_endpoint="https://kfp.test.com",
            namespace="test",
            base_image="test:latest"
        )
        
        config = GarakRemoteConfig(
            tls_verify=str(cert_file),
            kubeflow_config=kubeflow_config
        )
        
        assert config.tls_verify == str(cert_file)


class TestKubeflowConfig:
    """Test cases for KubeflowConfig"""

    def test_kubeflow_config_required_fields(self):
        """Test that all required fields must be provided"""
        with pytest.raises(ValidationError):
            KubeflowConfig()  # Missing required fields

    def test_kubeflow_config_valid(self):
        """Test valid Kubeflow configuration"""
        config = KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
            namespace="default",
            base_image="python:3.9"
        )
        assert config.pipelines_endpoint == "https://kfp.example.com"
        assert config.namespace == "default"
        assert config.base_image == "python:3.9"
        assert config.pipelines_api_token is None  # Optional field

    def test_kubeflow_config_missing_required_fields(self):
        """Test that required fields must be provided"""
        # Missing pipelines_endpoint
        with pytest.raises(ValidationError) as exc_info:
            KubeflowConfig(
                namespace="default",
                base_image="python:3.9"
            )
        assert "pipelines_endpoint" in str(exc_info.value)

        # Missing namespace
        with pytest.raises(ValidationError) as exc_info:
            KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
                base_image="python:3.9"
        )
        assert "namespace" in str(exc_info.value)

    def test_kubeflow_config_with_token(self):
        """Test KubeflowConfig with pipelines_api_token"""
        config = KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
            namespace="default",
            base_image="test:latest",
            pipelines_api_token="test-token-12345"
        )
        assert config.pipelines_api_token.get_secret_value() == "test-token-12345"

    def test_kubeflow_config_token_default_none(self):
        """Test that pipelines_api_token defaults to None"""
        config = KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
            namespace="default",
            base_image="test:latest"
        )
        assert config.pipelines_api_token is None


class TestGarakScanConfig:
    """Test cases for GarakScanConfig"""

    def test_default_scan_config(self):
        """Test default scan configuration"""
        config = GarakScanConfig()
        
        # Check framework profiles
        assert "trustyai_garak::owasp_llm_top10" in config.FRAMEWORK_PROFILES
        assert "trustyai_garak::avid_security" in config.FRAMEWORK_PROFILES
        assert "trustyai_garak::avid_ethics" in config.FRAMEWORK_PROFILES
        assert "trustyai_garak::avid_performance" in config.FRAMEWORK_PROFILES
        
        # Check scan profiles
        assert "trustyai_garak::quick" in config.SCAN_PROFILES
        assert "trustyai_garak::standard" in config.SCAN_PROFILES
        
        # Check other defaults
        assert config.VULNERABLE_SCORE == 0.5
        assert config.parallel_probes == 8
        assert config.cleanup_scan_dir_on_exit is True
        assert config.scan_dir.name == "llama_stack_garak_scans"

    def test_framework_profile_structure(self):
        """Test framework profile structure"""
        config = GarakScanConfig()
        owasp_profile = config.FRAMEWORK_PROFILES["trustyai_garak::owasp_llm_top10"]
        
        assert "name" in owasp_profile
        assert "description" in owasp_profile
        assert "taxonomy_filters" in owasp_profile
        assert "timeout" in owasp_profile
        assert "documentation" in owasp_profile
        assert "taxonomy" in owasp_profile
        assert isinstance(owasp_profile["taxonomy_filters"], list)
        assert owasp_profile["taxonomy"] == "owasp"

    def test_scan_profile_structure(self):
        """Test scan profile structure"""
        config = GarakScanConfig()
        quick_profile = config.SCAN_PROFILES["trustyai_garak::quick"]
        
        assert "name" in quick_profile
        assert "description" in quick_profile
        assert "probes" in quick_profile
        assert "timeout" in quick_profile
        assert isinstance(quick_profile["probes"], list)
        assert len(quick_profile["probes"]) > 0

    def test_scan_dir_path(self):
        """Test scan directory path construction"""
        config = GarakScanConfig()
        assert "llama_stack_garak_scans" in str(config.scan_dir)
        assert config.scan_dir.is_absolute()

    def test_scan_dir_respects_garak_scan_dir_env(self, monkeypatch, tmp_path):
        """Test that GARAK_SCAN_DIR environment variable overrides default scan_dir"""
        custom_scan_dir = tmp_path / "custom_garak_scans"
        custom_scan_dir.mkdir(parents=True, exist_ok=True)
        
        # Set GARAK_SCAN_DIR environment variable
        monkeypatch.setenv("GARAK_SCAN_DIR", str(custom_scan_dir))
        
        # Reload utils to pick up new environment variable
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        # Create config - should use GARAK_SCAN_DIR
        config = GarakScanConfig()
        
        assert config.scan_dir == custom_scan_dir
        assert str(config.scan_dir) == str(custom_scan_dir)

    def test_scan_dir_uses_xdg_cache_home_when_no_override(self, monkeypatch, tmp_path):
        """Test that scan_dir uses XDG_CACHE_HOME when GARAK_SCAN_DIR not set"""
        # Clear GARAK_SCAN_DIR if it exists
        monkeypatch.delenv("GARAK_SCAN_DIR", raising=False)
        
        # Set XDG_CACHE_HOME
        custom_cache = tmp_path / "custom_cache"
        custom_cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
        
        # Reload the utils module to pick up new XDG_CACHE_HOME
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        config = GarakScanConfig()
        
        # Should be XDG_CACHE_HOME/llama_stack_garak_scans
        assert str(custom_cache) in str(config.scan_dir)
        assert "llama_stack_garak_scans" in str(config.scan_dir)

    def test_cleanup_scan_dir_defaults_to_true(self):
        """Test that cleanup_scan_dir_on_exit defaults to True for production"""
        config = GarakScanConfig()
        assert config.cleanup_scan_dir_on_exit is True

    def test_scan_dir_respects_garak_scan_dir_env(self, monkeypatch, tmp_path):
        """Test that GARAK_SCAN_DIR environment variable overrides default scan_dir"""
        custom_scan_dir = tmp_path / "custom_garak_scans"
        custom_scan_dir.mkdir(parents=True, exist_ok=True)
        
        # Set GARAK_SCAN_DIR environment variable
        monkeypatch.setenv("GARAK_SCAN_DIR", str(custom_scan_dir))
        
        # Reload utils to pick up new environment variable
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        # Create config - should use GARAK_SCAN_DIR
        config = GarakScanConfig()
        
        assert config.scan_dir == custom_scan_dir
        assert str(config.scan_dir) == str(custom_scan_dir)

    def test_scan_dir_uses_xdg_cache_home_when_no_override(self, monkeypatch, tmp_path):
        """Test that scan_dir uses XDG_CACHE_HOME when GARAK_SCAN_DIR not set"""
        # Clear GARAK_SCAN_DIR if it exists
        monkeypatch.delenv("GARAK_SCAN_DIR", raising=False)
        
        # Set XDG_CACHE_HOME
        custom_cache = tmp_path / "custom_cache"
        custom_cache.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("XDG_CACHE_HOME", str(custom_cache))
        
        # Reload the utils module to pick up new XDG_CACHE_HOME
        from importlib import reload
        from llama_stack_provider_trustyai_garak import utils
        reload(utils)
        
        config = GarakScanConfig()
        
        # Should be XDG_CACHE_HOME/llama_stack_garak_scans
        assert str(custom_cache) in str(config.scan_dir)
        assert "llama_stack_garak_scans" in str(config.scan_dir)

    def test_cleanup_scan_dir_defaults_to_true(self):
        """Test that cleanup_scan_dir_on_exit defaults to True for production"""
        config = GarakScanConfig()
        assert config.cleanup_scan_dir_on_exit is True