from __future__ import annotations

from pathlib import Path

import yaml


def _workflow_trigger(config):
    trigger = config.get("on", config.get(True))
    if trigger is None:
        raise AssertionError("Workflow trigger section is missing")
    return trigger


def test_deploy_workflow_is_wired_for_pages():
    workflow_path = Path(".github/workflows/deploy.yml")
    config = yaml.safe_load(workflow_path.read_text())

    trigger = _workflow_trigger(config)
    assert "push" in trigger
    assert trigger["push"]["branches"] == ["main"]
    assert "workflow_dispatch" in trigger

    assert config["permissions"]["contents"] == "write"
    assert config["jobs"]["deploy"]["runs-on"] == "ubuntu-latest"

    steps = config["jobs"]["deploy"]["steps"]
    step_uses = [step.get("uses") for step in steps if isinstance(step, dict)]
    assert "actions/checkout@v4" in step_uses
    assert "peaceiris/actions-gh-pages@v4" in step_uses

    deploy_step = next(step for step in steps if step.get("uses") == "peaceiris/actions-gh-pages@v4")
    assert deploy_step["with"]["publish_branch"] == "gh-pages"
    assert deploy_step["with"]["publish_dir"] == "./_site"

