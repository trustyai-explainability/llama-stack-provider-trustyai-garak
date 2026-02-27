"""Tests for utility functions in utils.py and result_utils.py"""

import pytest
import os
import json
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


class TestResultUtils:
    """Test cases for result_utils functions"""

    def test_parse_jsonl_empty_lines(self):
        """Test parse_jsonl function handles empty lines correctly"""
        from llama_stack_provider_trustyai_garak.result_utils import parse_jsonl

        test_content = '{"entry_type": "init"}\n\n{"entry_type": "completion"}\n  \n'

        parsed = parse_jsonl(test_content)

        # Should parse valid JSON and skip empty lines
        assert len(parsed) == 2
        assert parsed[0]["entry_type"] == "init"
        assert parsed[1]["entry_type"] == "completion"

    def test_derive_template_vars_with_digest_entry(self):
        """Test derive_template_vars function with a digest entry present"""
        # Load test data
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()

        from llama_stack_provider_trustyai_garak.result_utils import parse_jsonl, derive_template_vars

        # Parse the JSONL content
        raw_report = parse_jsonl(test_content)

        # Get template variables
        template_vars = derive_template_vars(raw_report)

        # Verify the returned dictionary has expected keys and values
        assert "raw_report" in template_vars
        assert "report_name" in template_vars
        assert template_vars["raw_report"] == raw_report

    def test_derive_template_vars_without_digest_entry(self):
        """Test derive_template_vars function when no digest entry is present"""
        from llama_stack_provider_trustyai_garak.result_utils import derive_template_vars

        # Test data without digest entry
        test_report = [
            {"entry_type": "start_run setup", "some_key": "value"},
            {"entry_type": "init", "garak_version": "0.14.0.pre1"}
        ]

        template_vars = derive_template_vars(test_report)

        # Should return unknown report name when no digest entry
        assert template_vars["raw_report"] == test_report
        assert template_vars["report_name"] == "unknown"

    def test_generate_art_report(self):
        """Test generate_art_report function renders template correctly"""
        # Load test data
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()

        from llama_stack_provider_trustyai_garak.result_utils import generate_art_report

        # Generate the report
        rendered_html = generate_art_report(test_content)

        # Verify the HTML contains expected content
        assert "<!DOCTYPE html>" in rendered_html
        assert "<title>Garak - Automated Red Teaming report</title>" in rendered_html

        # Verify the expected JavaScript libraries are included
        assert "vega.min.js" in rendered_html
        assert "vega-lite.min.js" in rendered_html
        assert "vega-embed.min.js" in rendered_html

        # Open in browser for manual inspection
        try:
            import os
            import tempfile
            import webbrowser

            # Create a temporary HTML file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
                temp_file.write(rendered_html)
                temp_file.flush()

                # Open in browser
                webbrowser.open(f'file://{temp_file.name}')
                print(f"Report opened in browser: {temp_file.name}")
        except Exception as e:
            # If anything goes wrong with opening the browser, just continue the test
            print(f"Could not open browser for manual inspection: {e}")
            
    def test_result_parsing_with_art_result(self):
        # Load test data
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()
        
        from llama_stack_provider_trustyai_garak.result_utils import parse_generations_from_report_content
        
        generations, score_rows_by_probe, raw_entries_by_probe = parse_generations_from_report_content(test_content, eval_threshold=0.5)
        ## we only look at probes and not harnesses
        assert len(generations) == 73
        assert set(score_rows_by_probe.keys()) == {'base.IntentProbe', 'spo.SPOIntent', 'spo.SPOIntentUserAugmented',
                                                   'spo.SPOIntentSystemAugmented', 'spo.SPOIntentBothAugmented'}
        assert set(raw_entries_by_probe.keys()) == set(score_rows_by_probe.keys())


class TestIntentsAggregation:
    """Tests for intents prompt-level aggregation aligned with ART HTML report."""

    def test_intents_aggregates_match_high_level_stats(self):
        """Verify calculate_intents_aggregates produces the same metrics as
        vega_data + high_level_stats (the ART HTML report pipeline)."""
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()

        from llama_stack_provider_trustyai_garak.result_utils import (
            parse_jsonl, vega_data, high_level_stats,
            parse_generations_from_report_content, calculate_intents_aggregates,
        )

        raw_report = parse_jsonl(test_content)
        art_data = vega_data(raw_report)
        art_stats = high_level_stats(art_data)
        art_dict = {s["label"]: s["value"] for s in art_stats}

        _, _, raw_entries_by_probe = parse_generations_from_report_content(test_content, 0.5)
        all_raw = [e for entries in raw_entries_by_probe.values() for e in entries]
        intents_metrics = calculate_intents_aggregates(all_raw)

        assert intents_metrics["total_attacks"] == art_dict["Total attacks"]
        assert intents_metrics["successful_attacks"] == art_dict["Successful attacks"]
        assert intents_metrics["safe_prompts"] == art_dict["Safe prompts"]
        expected_rate = art_dict["Attack success rate"].replace("%", "")
        assert format(intents_metrics["attack_success_rate"], '.0f') == expected_rate

    def test_intents_aggregates_per_probe(self):
        """Per-probe intents aggregates should sum to overall totals
        for total_attacks and successful_attacks."""
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()

        from llama_stack_provider_trustyai_garak.result_utils import (
            parse_generations_from_report_content, calculate_intents_aggregates,
        )

        _, _, raw_entries_by_probe = parse_generations_from_report_content(test_content, 0.5)

        sum_attacks = 0
        sum_successful = 0
        for probe_entries in raw_entries_by_probe.values():
            metrics = calculate_intents_aggregates(probe_entries)
            sum_attacks += metrics["total_attacks"]
            sum_successful += metrics["successful_attacks"]
            assert metrics["total_prompts"] >= metrics["safe_prompts"]
            assert metrics["attack_success_rate"] >= 0

        all_raw = [e for entries in raw_entries_by_probe.values() for e in entries]
        overall = calculate_intents_aggregates(all_raw)
        assert overall["total_attacks"] == sum_attacks
        assert overall["successful_attacks"] == sum_successful

    def test_intents_aggregates_empty_input(self):
        from llama_stack_provider_trustyai_garak.result_utils import calculate_intents_aggregates
        result = calculate_intents_aggregates([])
        assert result["total_attacks"] == 0
        assert result["successful_attacks"] == 0
        assert result["attack_success_rate"] == 0

    def test_combine_parsed_results_uses_intents_path(self):
        """combine_parsed_results with art_intents=True should produce
        intents-style metrics, not attempt-level ones."""
        test_data_path = Path(__file__).parent / "_resources/garak_earlystop_run.jsonl"
        with open(test_data_path, 'r') as f:
            test_content = f.read()

        from llama_stack_provider_trustyai_garak.result_utils import (
            parse_generations_from_report_content,
            parse_aggregated_from_avid_content,
            parse_digest_from_report_content,
            combine_parsed_results,
        )

        generations, score_rows_by_probe, raw_entries_by_probe = \
            parse_generations_from_report_content(test_content, 0.5)
        digest = parse_digest_from_report_content(test_content)

        result_intents = combine_parsed_results(
            generations, score_rows_by_probe, {},
            0.5, digest,
            art_intents=True,
            raw_entries_by_probe=raw_entries_by_probe,
        )
        result_native = combine_parsed_results(
            generations, score_rows_by_probe, {},
            0.5, digest,
            art_intents=False,
        )

        overall_intents = result_intents["scores"]["_overall"]["aggregated_results"]
        overall_native = result_native["scores"]["_overall"]["aggregated_results"]

        # Intents path should have prompt-level fields
        assert "total_attacks" in overall_intents
        assert "safe_prompts" in overall_intents
        assert "total_attempts" not in overall_intents

        # Native path should have attempt-level fields
        assert "total_attempts" in overall_native
        assert "vulnerable_responses" in overall_native
        assert "total_attacks" not in overall_native
        
