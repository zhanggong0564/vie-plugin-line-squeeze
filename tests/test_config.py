import pytest
from pydantic import ValidationError

from vie_plugin_line_squeeze.config import LineSqueezeConfig


def test_config_reads_prefixed_environment(monkeypatch):
    monkeypatch.setenv("LINE_SQUEEZE_DET_CONF_THRESHOLD", "0.65")
    assert LineSqueezeConfig().det_conf_threshold == 0.65


def test_config_rejects_invalid_class_count(monkeypatch):
    monkeypatch.setenv("LINE_SQUEEZE_DET_NC", "0")
    with pytest.raises(ValidationError):
        LineSqueezeConfig()
