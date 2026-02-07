"""
Unit tests for configuration module.
"""

import pytest
from cross_dim_transfer.src.configs.default import (
    Config,
    get_config,
    MODEL_CONFIGS,
    DATASET_CONFIGS,
    TRAINING_CONFIGS
)


class TestConfig:
    """Tests for Config class."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = Config()

        assert config.ct_embedding_dim == 768
        assert config.pathology_embedding_dim == 768
        assert config.shared_embedding_dim == 512
        assert config.num_attention_heads == 8

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = Config(
            ct_embedding_dim=512,
            num_attention_heads=4
        )

        assert config.ct_embedding_dim == 512
        assert config.num_attention_heads == 4
        # Unchanged values should remain default
        assert config.shared_embedding_dim == 512

    def test_medgemma_config(self):
        """Test MedGemma configuration."""
        config = Config()

        assert config.medgemma_model_name == "google/medgemma-1.5-4b-it"
        assert config.medgemma_load_in_4bit == True
        assert config.medgemma_feature_dim == 768


class TestGetConfig:
    """Tests for get_config function."""

    def test_small_model(self):
        """Test small model preset."""
        config = get_config(model_size='small')

        assert config.ct_embedding_dim == 512
        assert config.pathology_embedding_dim == 512
        assert config.shared_embedding_dim == 256

    def test_medium_model(self):
        """Test medium model preset."""
        config = get_config(model_size='medium')

        assert config.ct_embedding_dim == 768
        assert config.pathology_embedding_dim == 768
        assert config.shared_embedding_dim == 512

    def test_large_model(self):
        """Test large model preset."""
        config = get_config(model_size='large')

        assert config.ct_embedding_dim == 1024
        assert config.pathology_embedding_dim == 1024
        assert config.shared_embedding_dim == 768

    def test_dataset_lite(self):
        """Test lite dataset preset."""
        config = get_config(dataset_size='lite')

        assert config.num_ct_slices == 32
        assert config.target_ct_size == (32, 32, 32)

    def test_dataset_debug(self):
        """Test debug dataset preset."""
        config = get_config(dataset_size='debug')

        assert config.num_ct_slices == 16
        assert config.target_ct_size == (16, 16, 16)

    def test_training_fast(self):
        """Test fast training preset."""
        config = get_config(training_mode='fast')

        assert config.num_epochs == 20
        assert config.batch_size == 4

    def test_training_thorough(self):
        """Test thorough training preset."""
        config = get_config(training_mode='thorough')

        assert config.num_epochs == 200
        assert config.batch_size == 2
        assert config.learning_rate == 5e-5

    def test_override_preset(self):
        """Test that kwargs override preset values."""
        config = get_config(
            model_size='small',
            ct_embedding_dim=1024  # Override preset
        )

        assert config.ct_embedding_dim == 1024


class TestPresetConfigs:
    """Tests for preset configuration dictionaries."""

    def test_model_configs_exist(self):
        """Test that model presets exist."""
        assert 'small' in MODEL_CONFIGS
        assert 'medium' in MODEL_CONFIGS
        assert 'large' in MODEL_CONFIGS

    def test_dataset_configs_exist(self):
        """Test that dataset presets exist."""
        assert 'full' in DATASET_CONFIGS
        assert 'lite' in DATASET_CONFIGS
        assert 'debug' in DATASET_CONFIGS

    def test_training_configs_exist(self):
        """Test that training presets exist."""
        assert 'fast' in TRAINING_CONFIGS
        assert 'standard' in TRAINING_CONFIGS
        assert 'thorough' in TRAINING_CONFIGS

    def test_preset_structure(self):
        """Test that presets have expected keys."""
        assert 'ct_embedding_dim' in MODEL_CONFIGS['medium']
        assert 'shared_embedding_dim' in MODEL_CONFIGS['medium']

        assert 'num_ct_slices' in DATASET_CONFIGS['full']
        assert 'target_ct_size' in DATASET_CONFIGS['full']

        assert 'num_epochs' in TRAINING_CONFIGS['standard']
        assert 'batch_size' in TRAINING_CONFIGS['standard']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
