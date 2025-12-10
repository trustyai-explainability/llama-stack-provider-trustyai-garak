"""Tests for version management utilities"""

from unittest.mock import patch, MagicMock, mock_open


class TestGetGarakVersion:
    """Test cases for get_garak_version()"""

    def test_get_version_from_metadata_with_exact_version(self):
        """Test getting version from importlib.metadata with exact version constraint"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        # Mock the distribution to return garak with exact version
        mock_dist = MagicMock()
        mock_dist.requires = ["garak==0.12.0", "llama-stack==0.3.0", "httpx>=0.28.1"]
        
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            version = get_garak_version()
            assert version == "garak==0.12.0"

    def test_get_version_from_metadata_with_min_version(self):
        """Test getting version with minimum version constraint"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        mock_dist = MagicMock()
        mock_dist.requires = ["garak>=0.12.0", "other-package==1.0.0"]
        
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            version = get_garak_version()
            assert version == "garak>=0.12.0"

    def test_get_version_from_metadata_with_environment_markers(self):
        """Test version extraction when dependency has environment markers"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        mock_dist = MagicMock()
        mock_dist.requires = [
            "garak==0.12.0; python_version >= '3.12'",
            "other-package==1.0.0"
        ]
        
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            version = get_garak_version()
            # Should strip environment markers
            assert version == "garak==0.12.0"
            assert "python_version" not in version

    def test_get_version_from_metadata_no_requires(self):
        """Test fallback when distribution has no requires"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        mock_dist = MagicMock()
        mock_dist.requires = None
        
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            # Should fall back to reading pyproject.toml or return "garak"
            version = get_garak_version()
            assert version.startswith("garak")

    def test_get_version_from_metadata_garak_not_in_requires(self):
        """Test fallback when garak is not in requires list"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        mock_dist = MagicMock()
        mock_dist.requires = ["llama-stack==0.3.0", "httpx>=0.28.1"]
        
        with patch("importlib.metadata.distribution", return_value=mock_dist):
            # Should fall back since garak is not found
            version = get_garak_version()
            assert version.startswith("garak")

    def test_get_version_metadata_import_error_fallback_to_toml(self):
        """Test fallback to pyproject.toml when importlib.metadata fails"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        # Create a temporary pyproject.toml
        toml_content = b"""
[project]
name = "test-package"
dependencies = [
    "garak==0.12.0",
    "other-package==1.0.0"
]
"""
        # Mock the distribution to raise an exception
        with patch("importlib.metadata.distribution", side_effect=Exception("Package not found")):
            # Mock the pyproject.toml file reading
            with patch("builtins.open", mock_open(read_data=toml_content)):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": ["garak==0.12.0", "other-package==1.0.0"]
                            }
                        }
                        version = get_garak_version()
                        assert version == "garak==0.12.0"

    def test_get_version_toml_with_environment_markers(self):
        """Test version extraction from toml with environment markers"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": [
                                    "garak==0.12.0; python_version >= '3.12'",
                                    "other-package"
                                ]
                            }
                        }
                        version = get_garak_version()
                        # Should strip environment markers
                        assert version == "garak==0.12.0"

    def test_get_version_toml_file_not_found(self):
        """Test final fallback when pyproject.toml doesn't exist"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("pathlib.Path.exists", return_value=False):
                version = get_garak_version()
                # Should return unversioned fallback
                assert version == "garak"

    def test_get_version_toml_parse_error(self):
        """Test final fallback when toml parsing fails"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load", side_effect=Exception("Parse error")):
                        version = get_garak_version()
                        # Should return unversioned fallback
                        assert version == "garak"

    def test_get_version_toml_no_dependencies_key(self):
        """Test fallback when pyproject.toml has no dependencies key"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        mock_toml.return_value = {"project": {}}
                        version = get_garak_version()
                        # Should return unversioned fallback
                        assert version == "garak"

    def test_get_version_toml_garak_not_in_dependencies(self):
        """Test fallback when garak is not in pyproject.toml dependencies"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": ["other-package==1.0.0"]
                            }
                        }
                        version = get_garak_version()
                        # Should return unversioned fallback
                        assert version == "garak"

    def test_get_version_from_inline_extra_in_toml(self):
        """Test that get_garak_version reads from optional-dependencies.inline in pyproject.toml"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        # Mock importlib.metadata to fail, forcing toml fallback
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        # Simulate new pyproject.toml structure with garak in inline extra
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": [
                                    "llama-stack>=0.3.0",
                                    "kfp>=2.14.6",
                                    # No garak here anymore!
                                ],
                                "optional-dependencies": {
                                    "inline": [
                                        "langchain==0.3.27",
                                        "garak==0.12.0",  # Should be read from here
                                    ],
                                    "dev": ["pytest"]
                                }
                            }
                        }
                        version = get_garak_version()
                        # Should get version from inline extra (new behavior)
                        assert version == "garak==0.12.0"

    def test_get_version_from_inline_extra_prefers_over_dependencies(self):
        """Test that optional-dependencies.inline is preferred over dependencies"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        # Both have garak, but inline should win
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": [
                                    "garak==0.99.9",  # Old version in dependencies
                                ],
                                "optional-dependencies": {
                                    "inline": [
                                        "garak==0.12.0",  # Should use this one
                                    ]
                                }
                            }
                        }
                        version = get_garak_version()
                        # Should prefer inline extra version
                        assert version == "garak==0.12.0"

    def test_get_version_fallback_to_dependencies_when_no_inline_extra(self):
        """Test that it falls back to dependencies when inline extra doesn't have garak"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with patch("importlib.metadata.distribution", side_effect=Exception("Not found")):
            with patch("builtins.open", mock_open()):
                with patch("pathlib.Path.exists", return_value=True):
                    with patch("tomllib.load") as mock_toml:
                        # Inline extra exists but doesn't have garak
                        mock_toml.return_value = {
                            "project": {
                                "dependencies": [
                                    "garak==0.11.5",  # Should fall back to this
                                ],
                                "optional-dependencies": {
                                    "inline": [
                                        "langchain==0.3.27",
                                        # No garak here
                                    ]
                                }
                            }
                        }
                        version = get_garak_version()
                        # Should fall back to dependencies (backwards compatibility)
                        assert version == "garak==0.11.5"

    def test_get_version_real_package_metadata(self):
        """Test with real package metadata (integration test)"""
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        # This will try to get the real version from the installed package
        # It should not raise an exception
        version = get_garak_version()
        
        assert isinstance(version, str)
        assert version.startswith("garak")
        # If running in development, it might be just "garak"
        # If installed, it should have a version specifier
        assert len(version) >= len("garak")

    def test_get_version_logging_on_metadata_error(self, caplog):
        """Test that errors are logged when metadata extraction fails"""
        import logging
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with caplog.at_level(logging.DEBUG):
            with patch("importlib.metadata.distribution", side_effect=Exception("Test error")):
                with patch("pathlib.Path.exists", return_value=False):
                    version = get_garak_version()
                    
                    # Should log specific error message
                    assert "Error getting garak version from importlib.metadata" in caplog.text
                    assert version == "garak"

    def test_get_version_logging_on_toml_error(self, caplog):
        """Test that errors are logged when toml reading fails"""
        import logging
        from llama_stack_provider_trustyai_garak import get_garak_version
        
        with caplog.at_level(logging.DEBUG):
            with patch("importlib.metadata.distribution", side_effect=Exception("Metadata error")):
                with patch("builtins.open", mock_open()):
                    with patch("pathlib.Path.exists", return_value=True):
                        with patch("tomllib.load", side_effect=Exception("TOML error")):
                            version = get_garak_version()
                            
                            # Should log both metadata and toml errors
                            assert "Error getting garak version from importlib.metadata" in caplog.text
                            assert "Error getting garak version from pyproject.toml" in caplog.text
                            assert version == "garak"


class TestProviderSpecsUseGarakVersion:
    """Test that provider specs correctly use get_garak_version()"""

    def test_inline_provider_uses_garak_version(self):
        """Test that inline provider spec uses get_garak_version()"""
        from llama_stack_provider_trustyai_garak.inline.provider import get_provider_spec
        
        spec = get_provider_spec()
        
        # Check that pip_packages contains garak
        assert len(spec.pip_packages) > 0
        garak_package = next((pkg for pkg in spec.pip_packages if pkg.startswith("garak")), None)
        assert garak_package is not None
        assert "garak" in garak_package

    def test_remote_provider_uses_garak_version(self):
        """Test that remote provider spec uses get_garak_version()"""
        from llama_stack_provider_trustyai_garak.remote.provider import get_provider_spec
        
        spec = get_provider_spec()
        
        # Check that pip_packages contains garak
        assert len(spec.pip_packages) > 0
        garak_package = next((pkg for pkg in spec.pip_packages if pkg.startswith("garak")), None)
        assert garak_package is not None
        assert "garak" in garak_package

    def test_inline_and_remote_use_same_garak_version(self):
        """Test that inline and remote providers use the same garak version"""
        from llama_stack_provider_trustyai_garak.inline.provider import get_provider_spec as inline_spec
        from llama_stack_provider_trustyai_garak.remote.provider import get_provider_spec as remote_spec
        
        inline = inline_spec()
        remote = remote_spec()
        
        inline_garak = next((pkg for pkg in inline.pip_packages if pkg.startswith("garak")), None)
        remote_garak = next((pkg for pkg in remote.pip_packages if pkg.startswith("garak")), None)
        
        # Both should have garak and they should be the same version
        assert inline_garak is not None
        assert remote_garak is not None
        assert inline_garak == remote_garak

    def test_provider_spec_garak_version_is_valid_format(self):
        """Test that the garak version in provider specs is a valid format"""
        from llama_stack_provider_trustyai_garak.inline.provider import get_provider_spec
        import re
        
        spec = get_provider_spec()
        garak_package = next((pkg for pkg in spec.pip_packages if pkg.startswith("garak")), None)
        
        # Should match patterns like:
        # - "garak"
        # - "garak==0.12.0"
        # - "garak>=0.12.0"
        # - "garak~=0.12.0"
        version_pattern = r"^garak(==|>=|<=|~=|!=)?\d*\.?\d*\.?\d*$"
        assert re.match(version_pattern, garak_package), f"Invalid garak version format: {garak_package}"

