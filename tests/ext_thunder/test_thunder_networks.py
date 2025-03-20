"""Run thunder tests as part of LitGPT CI"""
from litgpt.utils import _THUNDER_AVAILABLE

if _THUNDER_AVAILABLE:
    from thunder.tests.test_networks import *  # noqa: F403
