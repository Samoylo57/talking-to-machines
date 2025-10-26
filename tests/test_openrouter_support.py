from unittest.mock import MagicMock

import openai

from talkingtomachines.config import DevelopmentConfig
from talkingtomachines.generative.synthetic_subject import SyntheticSubject


def test_synthetic_subject_uses_configured_openrouter_base_url(monkeypatch):
    monkeypatch.setattr(DevelopmentConfig, "OPENAI_API_KEY", "test-key", raising=False)
    monkeypatch.setattr(
        DevelopmentConfig,
        "OPENAI_BASE_URL",
        "https://openrouter.ai/api/v1",
        raising=False,
    )

    captured_kwargs = {}

    def fake_openai_client(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(openai, "OpenAI", fake_openai_client)

    SyntheticSubject(
        experiment_id="exp-1",
        experiment_context="context",
        session_id="session",
        profile_info={"Name": "Tester"},
        model_info="gpt-5",
        temperature=0.5,
        include_backstories=False,
        hf_inference_endpoint="",
    )

    assert captured_kwargs["api_key"] == "test-key"
    assert captured_kwargs["base_url"] == "https://openrouter.ai/api/v1"
    assert set(captured_kwargs) <= {"api_key", "base_url"}
