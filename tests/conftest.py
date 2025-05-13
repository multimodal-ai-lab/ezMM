import pytest


@pytest.fixture(autouse=True, scope="function")
def run_before_each_test():
    from ezmm.common.registry import item_registry
    item_registry.reset()
