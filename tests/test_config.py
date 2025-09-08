"""Tests for configuration classes"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from llama_stack_provider_trustyai_garak.config import (
    GarakEvalProviderConfig,
    GarakRemoteConfig,
    KubeflowConfig,
    GarakScanConfig
)


class TestGarakEvalProviderConfig:
    """Test cases for GarakEvalProviderConfig"""

    def test_default_config(self):
        """Test default configuration values"""
        config = GarakEvalProviderConfig()
        assert config.base_url == "http://localhost:8321/v1"
        assert config.garak_model_type_openai == "openai.OpenAICompatible"
        assert config.garak_model_type_function == "function.Single"
        assert config.timeout == 60 * 60 * 3
        assert config.max_workers == 5
        assert config.max_concurrent_jobs == 5
        assert config.tls_verify is True

    def test_custom_config(self):
        """Test custom configuration values"""
        config = GarakEvalProviderConfig(
            base_url="https://custom.api.com/v1",
            garak_model_type_openai="custom.model",
            timeout=1000,
            max_workers=10,
            max_concurrent_jobs=3
        )
        assert config.base_url == "https://custom.api.com/v1"
        assert config.garak_model_type_openai == "custom.model"
        assert config.timeout == 1000
        assert config.max_workers == 10
        assert config.max_concurrent_jobs == 3

    def test_tls_verify_boolean(self):
        """Test TLS verify with boolean values"""
        config_true = GarakEvalProviderConfig(tls_verify=True)
        assert config_true.tls_verify is True
        
        config_false = GarakEvalProviderConfig(tls_verify=False)
        assert config_false.tls_verify is False

    def test_tls_verify_cert_path(self, tmp_path):
        """Test TLS verify with certificate path"""
        # Create a temporary certificate file
        cert_file = tmp_path / "cert.pem"
        cert_file.write_text("CERTIFICATE CONTENT")
        
        config = GarakEvalProviderConfig(tls_verify=str(cert_file))
        assert config.tls_verify == str(cert_file)

    def test_tls_verify_invalid_path(self):
        """Test TLS verify with invalid certificate path"""
        with pytest.raises(ValidationError) as exc_info:
            GarakEvalProviderConfig(tls_verify="/invalid/path/cert.pem")
        assert "TLS certificate file does not exist" in str(exc_info.value)

    def test_tls_verify_directory_path(self, tmp_path):
        """Test TLS verify with directory path (should fail)"""
        with pytest.raises(ValidationError) as exc_info:
            GarakEvalProviderConfig(tls_verify=str(tmp_path))
        assert "TLS certificate path is not a file" in str(exc_info.value)

    def test_base_url_validation(self):
        """Test base URL validation (whitespace trimming)"""
        config = GarakEvalProviderConfig(base_url="  https://api.com/v1  ")
        assert config.base_url == "https://api.com/v1"

    def test_invalid_base_url_type(self):
        """Test invalid base URL type"""
        with pytest.raises(ValidationError):
            GarakEvalProviderConfig(base_url=123)

    def test_sample_run_config_with_integers(self):
        """Test sample_run_config class method with integer values"""
        config_dict = GarakEvalProviderConfig.sample_run_config(
            base_url="https://test.api.com",
            timeout=5000,
            max_workers=8,
            max_concurrent_jobs=10
        )
        assert config_dict["base_url"] == "https://test.api.com"
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
            GarakEvalProviderConfig.sample_run_config()
        assert "invalid literal for int()" in str(exc_info.value)


class TestGarakRemoteConfig:
    """Test cases for GarakRemoteConfig"""

    def test_remote_config_with_kubeflow(self):
        """Test remote config with Kubeflow settings"""
        kubeflow_config = KubeflowConfig(
            pipelines_endpoint="https://kfp.example.com",
            namespace="garak-namespace",
            experiment_name="garak-experiment",
            base_image="garak:latest"
        )
        
        config = GarakRemoteConfig(
            kubeflow_config=kubeflow_config
        )
        
        assert config.kubeflow_config.pipelines_endpoint == "https://kfp.example.com"
        assert config.kubeflow_config.namespace == "garak-namespace"
        assert config.kubeflow_config.experiment_name == "garak-experiment"
        assert config.kubeflow_config.base_image == "garak:latest"
        
        # Should inherit base config defaults
        assert config.base_url == "http://localhost:8321/v1"
        assert config.timeout == 60 * 60 * 3


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
            experiment_name="test-exp",
            base_image="python:3.9"
        )
        assert config.pipelines_endpoint == "https://kfp.example.com"
        assert config.namespace == "default"
        assert config.experiment_name == "test-exp"
        assert config.base_image == "python:3.9"


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
        assert config.cleanup_scan_dir_on_exit is False
        assert config.scan_dir.name == "_scan_files"

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
        assert config.scan_dir == config.base_dir / "_scan_files"
        assert config.base_dir == Path(__file__).parent.parent / "src" / "llama_stack_provider_trustyai_garak"