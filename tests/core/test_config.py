from dezero.core.config import Config, no_grad, using_config


class TestConfig:
    def test_config(self):
        assert Config.enable_backprop is True

    def test_using_config(self):
        with using_config("enable_backprop", False):
            assert Config.enable_backprop is False
        assert Config.enable_backprop is True

    def test_no_grad(self):
        with no_grad():
            assert Config.enable_backprop is False
        assert Config.enable_backprop is True
