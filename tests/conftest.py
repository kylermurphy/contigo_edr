"""Shared pytest configuration for the contigo test suite.

Tests marked ``network``, ``orekit`` or ``slow`` are skipped unless the matching
command-line flag is given (or ``--run-all``). Unmarked tests form the fast tier
that CI runs: they must not touch the network or start the JVM.

added: 09/11/2026 Claude Code
"""

import pytest

TIERS = ("network", "orekit", "slow")


def pytest_addoption(parser: pytest.Parser) -> None:
    for tier in TIERS:
        parser.addoption(
            f"--run-{tier}",
            action="store_true",
            default=False,
            help=f"run tests marked @pytest.mark.{tier}",
        )
    parser.addoption(
        "--run-all",
        action="store_true",
        default=False,
        help="run every test tier",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--run-all"):
        return
    for tier in TIERS:
        if config.getoption(f"--run-{tier}"):
            continue
        skip = pytest.mark.skip(reason=f"{tier} test: use --run-{tier} to run")
        for item in items:
            # check real markers only; item.keywords also holds node names
            if item.get_closest_marker(tier) is not None:
                item.add_marker(skip)
