"""Tests for configuration classes"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from llama_stack_provider_trustyai_garak.config import (
    GarakScanConfig,
)
from llama_stack_provider_trustyai_garak.core.config_resolution import deep_merge_dicts


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
        assert "trustyai_garak::avid" in config.FRAMEWORK_PROFILES
        assert "trustyai_garak::quality" in config.FRAMEWORK_PROFILES
        assert "trustyai_garak::cwe" in config.FRAMEWORK_PROFILES

        # Check scan profiles
        assert "trustyai_garak::quick" in config.SCAN_PROFILES

        # Check other defaults
        assert config.VULNERABLE_SCORE == 0.5
        assert config.parallel_probes == 8
        assert config.cleanup_scan_dir_on_exit is True
        assert config.scan_dir.name == "trustyai_garak_scans"

    def test_framework_profile_structure(self):
        """Test framework profile structure with new garak_config format"""
        config = GarakScanConfig()
        owasp_profile = config.FRAMEWORK_PROFILES["trustyai_garak::owasp_llm_top10"]

        assert "name" in owasp_profile
        assert "description" in owasp_profile
        assert "garak_config" in owasp_profile  # NEW: garak_config instead of individual keys
        assert "timeout" in owasp_profile
        assert "documentation" in owasp_profile

        # Check garak_config structure
        garak_config = owasp_profile["garak_config"]
        assert "run" in garak_config
        assert "reporting" in garak_config
        assert garak_config["run"]["probe_tags"] == "owasp:llm"  # NEW: probe_tags instead of taxonomy_filters
        assert garak_config["reporting"]["taxonomy"] == "owasp"

    def test_scan_profile_structure(self):
        """Test scan profile structure with new garak_config format"""
        config = GarakScanConfig()
        quick_profile = config.SCAN_PROFILES["trustyai_garak::quick"]

        assert "name" in quick_profile
        assert "description" in quick_profile
        assert "garak_config" in quick_profile  # NEW: garak_config instead of top-level keys
        assert "timeout" in quick_profile

        # Check garak_config structure
        garak_config = quick_profile["garak_config"]
        assert "plugins" in garak_config
        assert "probe_spec" in garak_config["plugins"]  # NEW: probe_spec instead of probes
        assert garak_config["plugins"]["probe_spec"] is not None
        assert len(garak_config["plugins"]["probe_spec"]) > 0

    def test_scan_dir_path(self):
        """Test scan directory path construction"""
        config = GarakScanConfig()
        assert "trustyai_garak_scans" in str(config.scan_dir)
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

        # Should be XDG_CACHE_HOME/trustyai_garak_scans
        assert str(custom_cache) in str(config.scan_dir)
        assert "trustyai_garak_scans" in str(config.scan_dir)

    def test_cleanup_scan_dir_defaults_to_true(self):
        """Test that cleanup_scan_dir_on_exit defaults to True for production"""
        config = GarakScanConfig()
        assert config.cleanup_scan_dir_on_exit is True

    @pytest.mark.parametrize(
        "profile_key, expected_probe_tag, expected_taxonomy",
        [
            ("trustyai_garak::owasp_llm_top10", "owasp:llm", "owasp"),
            ("trustyai_garak::avid", "avid-effect", "avid-effect"),
            ("trustyai_garak::avid_security", "avid-effect:security", "avid-effect"),
            ("trustyai_garak::avid_ethics", "avid-effect:ethics", "avid-effect"),
            ("trustyai_garak::avid_performance", "avid-effect:performance", "avid-effect"),
            ("trustyai_garak::quality", "quality", "quality"),
            ("trustyai_garak::cwe", "cwe", "cwe"),
        ],
    )
    def test_framework_profiles_parametrized(self, profile_key, expected_probe_tag, expected_taxonomy):
        """Test framework profile structure with new garak_config format"""
        config = GarakScanConfig()
        profile = config.FRAMEWORK_PROFILES[profile_key]

        # Basic structure checks shared across all framework profiles
        assert "name" in profile
        assert "description" in profile
        assert "garak_config" in profile  # NEW: garak_config instead of individual keys
        assert "timeout" in profile

        garak_config = profile["garak_config"]

        # Ensure expected garak_config sections exist
        assert "run" in garak_config
        assert "reporting" in garak_config
        assert "probe_tags" in garak_config["run"]

        # (1) Validate that the expected probe tag matches exactly
        assert garak_config["run"]["probe_tags"] == expected_probe_tag

        # (2) Validate taxonomy wiring
        assert garak_config["reporting"]["taxonomy"] == expected_taxonomy


class TestDeepMergeDicts:
    """Verify deep_merge_dicts honours leaf-level overrides without clobbering siblings."""

    def test_override_single_leaf_preserves_siblings(self):
        base = {
            "plugins": {
                "probes": {
                    "tap": {
                        "TAPIntent": {
                            "depth": 10,
                            "width": 10,
                            "branching_factor": 4,
                            "attack_model_config": {"uri": "", "max_tokens": 500},
                        }
                    }
                }
            }
        }
        override = {
            "plugins": {
                "probes": {
                    "tap": {
                        "TAPIntent": {
                            "depth": 20,
                        }
                    }
                }
            }
        }
        result = deep_merge_dicts(base, override)
        tap = result["plugins"]["probes"]["tap"]["TAPIntent"]
        assert tap["depth"] == 20
        assert tap["width"] == 10
        assert tap["branching_factor"] == 4
        assert tap["attack_model_config"]["max_tokens"] == 500

    def test_override_nested_dict_leaf_preserves_sibling_keys(self):
        base = {
            "plugins": {
                "detectors": {
                    "judge": {
                        "detector_model_config": {"uri": "http://old", "api_key": "k1", "max_tokens": 200},
                        "MulticlassJudge": {
                            "system_prompt": "Default prompt",
                            "score_key": "complied",
                            "confidence_cutoff": 70,
                        },
                    }
                }
            }
        }
        override = {
            "plugins": {
                "detectors": {
                    "judge": {
                        "MulticlassJudge": {
                            "system_prompt": "New custom prompt",
                        }
                    }
                }
            }
        }
        result = deep_merge_dicts(base, override)
        judge = result["plugins"]["detectors"]["judge"]
        assert judge["detector_model_config"]["uri"] == "http://old"
        assert judge["detector_model_config"]["max_tokens"] == 200
        mcj = judge["MulticlassJudge"]
        assert mcj["system_prompt"] == "New custom prompt"
        assert mcj["score_key"] == "complied"
        assert mcj["confidence_cutoff"] == 70

    def test_override_does_not_mutate_base(self):
        base = {"a": {"b": 1, "c": 2}}
        override = {"a": {"b": 99}}
        result = deep_merge_dicts(base, override)
        assert result["a"]["b"] == 99
        assert result["a"]["c"] == 2
        assert base["a"]["b"] == 1, "Original base must not be mutated"

    def test_adding_new_key_at_deep_level(self):
        base = {"plugins": {"detectors": {"judge": {"detector_model_name": "m1"}}}}
        override = {"plugins": {"detectors": {"judge": {"MulticlassJudge": {"system_prompt": "Added later"}}}}}
        result = deep_merge_dicts(base, override)
        judge = result["plugins"]["detectors"]["judge"]
        assert judge["detector_model_name"] == "m1"
        assert judge["MulticlassJudge"]["system_prompt"] == "Added later"

    def test_dict_override_merges_preserving_siblings(self):
        base = {"run": {"generations": 5, "eval_threshold": 0.5}}
        override = {"run": {"generations": 100}}
        result = deep_merge_dicts(base, override)
        assert result["run"]["generations"] == 100
        assert result["run"]["eval_threshold"] == 0.5

    def test_non_dict_override_replaces_entirely(self):
        base = {"run": {"generations": 5, "eval_threshold": 0.5}}
        override = {"run": "disabled"}
        result = deep_merge_dicts(base, override)
        assert result["run"] == "disabled"
