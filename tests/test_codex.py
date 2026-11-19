# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CodexAgent subprocess lifecycle tests (no real Codex CLI required)."""
from __future__ import annotations

import time
from pathlib import Path

import dream_rsi.agents.codex as codex_module
from dream_rsi.agents import base
from dream_rsi.agents.codex import CodexAgent


def test_codex_timeout_kills_the_process_group_and_untracks_it(tmp_path: Path, monkeypatch):
    # Fake "codex" CLI: answers --help instantly (for binary verification),
    # otherwise sleeps well past the agent's timeout.
    fake_bin = tmp_path / "codex"
    fake_bin.write_text(
        '#!/bin/sh\n[ "$1" = exec ] && [ "$2" = --help ] && exit 0\nsleep 5\n',
        encoding="utf-8",
    )
    fake_bin.chmod(0o755)

    captured: list = []
    real_register = codex_module.register_proc

    def spying_register(proc):
        captured.append(proc)
        return real_register(proc)

    monkeypatch.setattr(codex_module, "register_proc", spying_register)

    agent = CodexAgent(codex_bin=str(fake_bin), timeout_sec=0.3, plain_exec=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = agent.run(
        prompt="hi",
        workspace=workspace,
        node_dir=workspace,
        log_dir=tmp_path / "log",
        guide={},
    )

    assert result.status == "failed"
    assert "timeout" in result.message
    assert len(captured) == 1

    proc = captured[0]
    for _ in range(20):
        if proc.poll() is not None:
            break
        time.sleep(0.1)
    assert proc.poll() is not None, "codex subprocess was left running after a timeout"
    assert proc not in base._ACTIVE, "codex subprocess was never unregistered"
