from agent import config


def test_openai_config_is_three_module_globals():
    assert isinstance(config.BASE_URL, str)
    assert isinstance(config.MODEL, str)
    assert isinstance(config.API_KEY, str)
