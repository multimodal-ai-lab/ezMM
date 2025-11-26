import pytest


@pytest.fixture(autouse=True, scope="function")
def run_before_each_test():
    from ezmm import reset_ezmm
    reset_ezmm()
