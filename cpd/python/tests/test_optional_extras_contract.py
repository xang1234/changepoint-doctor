from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = REPO_ROOT / "cpd" / "python" / "pyproject.toml"
EXPECTED_EXTRAS = {"plot", "notebooks", "parity", "dev"}


def _optional_dependencies() -> dict[str, list[str]]:
    pyproject = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project_table = pyproject["project"]
    optional = project_table["optional-dependencies"]
    assert isinstance(optional, dict)
    return optional


def test_expected_extras_groups_are_defined() -> None:
    optional = _optional_dependencies()
    assert set(optional.keys()) == EXPECTED_EXTRAS


def test_each_extra_group_has_at_least_one_requirement() -> None:
    optional = _optional_dependencies()
    for extra_name, requirements in optional.items():
        assert requirements, f"extra '{extra_name}' must declare at least one requirement"


def test_dev_extra_superset_of_workflow_extras() -> None:
    optional = _optional_dependencies()
    dev = set(optional["dev"])

    for extra_name in ("plot", "notebooks", "parity"):
        for requirement in optional[extra_name]:
            assert (
                requirement in dev
            ), f"dev extra should include '{requirement}' from '{extra_name}'"
