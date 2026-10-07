import pytest


@pytest.fixture(autouse=True, scope="function")
def run_before_each_test(tmp_path):
    """Lets each test run on a fresh, isolated item registry."""
    from ezmm.common import item_registry
    item_registry.close()
    item_registry.clear_cache()
    item_registry.set_path(tmp_path / "registry")
    yield
    item_registry.close()
    item_registry.clear_cache()
