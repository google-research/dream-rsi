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

"""OpenAI Codex CLI coding agent."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from dream_rsi.agents.base import (
    AgentResult,
    kill_proc_group,
    register_proc,
    save_prompt,
    save_run_logs,
    unregister_proc,
)

_verified_codex: set[str] = set()


def _assert_openai_codex(codex_path: str) -> None:
    if codex_path in _verified_codex:
        return
    cp = subprocess.run(
        [codex_path, "exec", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
        env=os.environ.copy(),
    )
    blob = ((cp.stdout or "") + (cp.stderr or "")).lower()
    if any(m in blob for m in ("granian", "django", ":9810", "listening at: http")):
        raise RuntimeError(f"Wrong `codex` binary at {codex_path} (not OpenAI Codex CLI)")
    _verified_codex.add(codex_path)


def _resolve_codex_bin(codex_bin: str) -> str:
    if codex_bin == "codex":
        found = shutil.which("codex")
        if not found:
            raise FileNotFoundError("Cannot find `codex` in PATH")
        path = found
    else:
        path = str(Path(codex_bin).resolve())
    _assert_openai_codex(path)
    return path


class CodexAgent:
    """
    OpenAI Codex CLI coding agent.

    Env: CODEX_BIN, CODEX_MODEL, CODEX_EXEC_TIMEOUT_SEC, CODEX_EXTRA_ARGS, CODEX_PLAIN_EXEC
    """

    name = "codex"

    def __init__(
        self,
        *,
        codex_bin: str = "codex",
        model: str | None = None,
        timeout_sec: float | None = 600.0,
        extra_args: list[str] | None = None,
        plain_exec: bool = False,
    ) -> None:
        self.codex_bin = codex_bin
        self.model = model
        self.timeout_sec = timeout_sec
        self.extra_args = extra_args or []
        self.plain_exec = plain_exec

    def _build_command(self, workspace: Path, prompt: str, last_message: Path) -> list[str]:
        codex_path = _resolve_codex_bin(self.codex_bin)
        cmd: list[str] = [codex_path, *self.extra_args, "exec", "--skip-git-repo-check", "-C", str(workspace)]
        if not self.plain_exec:
            cmd.extend(["--json", "--full-auto"])
        if self.model:
            cmd.extend(["-m", self.model])
        cmd.extend(["-o", str(last_message), prompt])
        return cmd

    def run(
        self,
        *,
        prompt: str,
        workspace: Path,
        node_dir: Path,
        log_dir: Path,
        guide: dict[str, Any],
        run_cwd: Path | None = None,
        stage: str | None = None,
    ) -> AgentResult:
        del stage  # Gemini alone currently has stage-specific tool allowlists.
        save_prompt(log_dir, prompt)
        last_message = log_dir / "codex_last_message.txt"
        cwd = run_cwd or workspace

        try:
            cmd = self._build_command(cwd, prompt, last_message)
        except (FileNotFoundError, RuntimeError) as exc:
            return AgentResult(status="failed", message=str(exc))

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(cwd),
                env=os.environ.copy(),
                start_new_session=True,
            )
        except OSError as exc:
            return AgentResult(status="failed", message=str(exc))

        # Own process group + registration, like every other agent/evaluator
        # subprocess, so a timeout or shutdown_agents() actually kills `codex`
        # instead of leaving it running in the background.
        register_proc(proc)
        try:
            try:
                stdout, _ = proc.communicate(timeout=self.timeout_sec)
            except subprocess.TimeoutExpired:
                kill_proc_group(proc)
                proc.communicate()
                return AgentResult(status="failed", message=f"codex timeout after {self.timeout_sec}s")
        finally:
            unregister_proc(proc)

        save_run_logs(log_dir, stdout=stdout or "", command=cmd)
        if proc.returncode != 0:
            return AgentResult(status="failed", message=f"codex exit {proc.returncode}")

        return AgentResult(message="codex finished")


def from_env() -> CodexAgent:
    timeout_raw = os.environ.get("CODEX_EXEC_TIMEOUT_SEC", "600").strip()
    return CodexAgent(
        codex_bin=os.environ.get("CODEX_BIN", "codex").strip() or "codex",
        model=os.environ.get("CODEX_MODEL") or None,
        timeout_sec=float(timeout_raw) if timeout_raw else None,
        extra_args=shlex.split(os.environ.get("CODEX_EXTRA_ARGS", "").strip()),
        plain_exec=os.environ.get("CODEX_PLAIN_EXEC", "").lower() in ("1", "true", "yes"),
    )
