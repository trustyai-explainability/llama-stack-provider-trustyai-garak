"""Tests for shield scanning functionality"""

import pytest
import httpx
from unittest.mock import Mock, patch, PropertyMock
import os

from llama_stack_provider_trustyai_garak.shield_scan import (
    SimpleShieldOrchestrator,
    simple_shield_orchestrator,
    CANNED_RESPONSE_TEXT
)
from llama_stack.apis.safety import RunShieldResponse


class TestSimpleShieldOrchestrator:
    """Test cases for SimpleShieldOrchestrator"""

    @pytest.fixture
    def orchestrator(self):
        """Create a fresh orchestrator instance"""
        return SimpleShieldOrchestrator()

    @pytest.fixture
    def mock_client(self):
        """Create a mock HTTP client"""
        client = Mock(spec=httpx.Client)
        return client

    def test_get_tls_verify_setting_true(self, orchestrator):
        """Test TLS verify setting with true values"""
        test_cases = ["True", "true", "1", "yes", "on", "TRUE"]
        for value in test_cases:
            with patch.dict(os.environ, {"GARAK_TLS_VERIFY": value}):
                assert orchestrator._get_tls_verify_setting() is True

    def test_get_tls_verify_setting_false(self, orchestrator):
        """Test TLS verify setting with false values"""
        test_cases = ["False", "false", "0", "no", "off", "FALSE"]
        for value in test_cases:
            with patch.dict(os.environ, {"GARAK_TLS_VERIFY": value}):
                assert orchestrator._get_tls_verify_setting() is False

    def test_get_tls_verify_setting_cert_path(self, orchestrator):
        """Test TLS verify setting with certificate path"""
        cert_path = "/path/to/cert.pem"
        with patch.dict(os.environ, {"GARAK_TLS_VERIFY": cert_path}):
            assert orchestrator._get_tls_verify_setting() == cert_path

    def test_get_tls_verify_setting_default(self, orchestrator):
        """Test TLS verify setting default value"""
        with patch.dict(os.environ, {}, clear=True):
            # Remove GARAK_TLS_VERIFY if it exists
            os.environ.pop("GARAK_TLS_VERIFY", None)
            assert orchestrator._get_tls_verify_setting() is True

    @patch('llama_stack_provider_trustyai_garak.shield_scan._process_clients', {})
    def test_client_property_creates_new_client(self, orchestrator):
        """Test that client property creates a new client when needed"""
        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            
            client = orchestrator.client
            
            mock_client_class.assert_called_once()
            assert client == mock_client_instance

    @patch('llama_stack_provider_trustyai_garak.shield_scan._process_clients', {})
    def test_client_property_reuses_existing_client(self, orchestrator):
        """Test that client property reuses existing client for same process"""
        with patch('httpx.Client') as mock_client_class:
            mock_client_instance = Mock()
            mock_client_class.return_value = mock_client_instance
            
            # First access
            client1 = orchestrator.client
            # Second access  
            client2 = orchestrator.client
            
            # Should only create one client
            mock_client_class.assert_called_once()
            assert client1 == client2

    def test_get_shield_response_success(self, orchestrator):
        """Test successful shield response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "violation": None
        }
        mock_client.post.return_value = mock_response
        
        # Use PropertyMock to mock the property
        with patch.object(type(orchestrator), 'client', new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client
            result = orchestrator._get_shield_response(
                shield_id="test_shield",
                prompt="test prompt",
                base_url="http://test.com"
            )
            
            assert isinstance(result, RunShieldResponse)
            mock_client.post.assert_called_once()

    def test_get_shield_response_with_violation(self, orchestrator):
        """Test shield response with violation"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "violation": {
                "user_message": "Violation detected",
                "violation_level": "error",
                "metadata": {}
            }
        }
        mock_client.post.return_value = mock_response
        
        with patch.object(type(orchestrator), 'client', new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client
            result = orchestrator._get_shield_response(
                shield_id="test_shield",
                prompt="harmful prompt",
                base_url="http://test.com"
            )
            
            assert isinstance(result, RunShieldResponse)
            assert result.violation is not None

    def test_run_shields_with_early_exit_no_violation(self, orchestrator):
        """Test running shields with no violations"""
        mock_response = RunShieldResponse(violation=None)
        
        with patch.object(orchestrator, '_get_shield_response', return_value=mock_response):
            result = orchestrator._run_shields_with_early_exit(
                shield_ids=["shield1", "shield2"],
                prompt="test prompt",
                base_url="http://test.com"
            )
            
            assert result is False

    def test_get_llm_response_success(self, orchestrator):
        """Test successful LLM response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "LLM response text"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 12,
                "total_tokens": 22
            }
        }
        mock_client.post.return_value = mock_response
        
        with patch.object(type(orchestrator), 'client', new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client
            result = orchestrator._get_LLM_response(
                model="test-model",
                prompt="test prompt",
                base_url="http://test.com"
            )
            
            assert result.choices[0].message.content == "LLM response text"

    def test_call_with_input_shield_violation(self, orchestrator):
        """Test orchestrator call with input shield violation"""
        with patch.object(orchestrator, '_run_shields_with_early_exit', return_value=True):
            result = orchestrator(
                prompt="harmful prompt",
                llm_io_shield_mapping={"input": ["shield1"], "output": []},
                base_url="http://test.com",
                model="test-model"
            )
            
            assert result == [CANNED_RESPONSE_TEXT]

    def test_call_with_output_shield_violation(self, orchestrator):
        """Test orchestrator call with output shield violation"""
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content="harmful response"))]
        
        with patch.object(orchestrator, '_run_shields_with_early_exit') as mock_shields:
            mock_shields.side_effect = [False, True]  # No input violation, output violation
            with patch.object(orchestrator, '_get_LLM_response', return_value=mock_llm_response):
                result = orchestrator(
                    prompt="test prompt",
                    llm_io_shield_mapping={"input": ["shield1"], "output": ["shield2"]},
                    base_url="http://test.com",
                    model="test-model"
                )
                
                assert result == [CANNED_RESPONSE_TEXT]

    def test_call_no_shields(self, orchestrator):
        """Test orchestrator call with no shields"""
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock(message=Mock(content="normal response"))]
        
        with patch.object(orchestrator, '_get_LLM_response', return_value=mock_llm_response):
            result = orchestrator(
                prompt="test prompt",
                llm_io_shield_mapping={"input": [], "output": []},
                base_url="http://test.com",
                model="test-model"
            )
            
            assert result == ["normal response"]

    def test_close_client(self, orchestrator):
        """Test closing client"""
        process_id = os.getpid()
        mock_client = Mock()
        
        with patch('llama_stack_provider_trustyai_garak.shield_scan._process_clients', {process_id: mock_client}):
            orchestrator.close()
            mock_client.close.assert_called_once()


class TestSimpleShieldOrchestratorSingleton:
    """Test the singleton instance"""

    def test_simple_shield_orchestrator_exists(self):
        """Test that simple_shield_orchestrator is available"""
        assert simple_shield_orchestrator is not None
        assert isinstance(simple_shield_orchestrator, SimpleShieldOrchestrator)