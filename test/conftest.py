import pytest

_TERMINAL_LOG: list[str] = []

@pytest.fixture
def logger():
    def write(message: str) -> None:
        _TERMINAL_LOG.append(message)

    return write


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _TERMINAL_LOG:
        return

    terminalreporter.section("human-readable log for checking normalization")

    for message in _TERMINAL_LOG:
        terminalreporter.write_line(f"- {message}")
