"""Fail a plugin invocation that executes no passing test calls."""
import pytest


def pytest_configure(config) -> None:
    config.pluginmanager.register(RequireExecutedTests(), 'require_executed_plugin_tests')


class RequireExecutedTests:
    def __init__(self) -> None:
        self.passed = 0

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == 'call' and report.passed:
            self.passed += 1

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus) -> None:
        if not self.passed and not session.exitstatus:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
