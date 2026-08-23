from src.config import Settings


def test_settings_defaults_are_sane():
    s = Settings()
    assert s.test_size == 0.2
    assert s.n_risk_clusters >= 2
    assert s.model_alias == "champion"


def test_settings_can_be_overridden_by_env(monkeypatch):
    monkeypatch.setenv("MODEL_ALIAS", "shadow")
    monkeypatch.setenv("RANDOM_STATE", "7")
    s = Settings()
    assert s.model_alias == "shadow"
    assert s.random_state == 7
