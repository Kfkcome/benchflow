"""BenchFlow-native verifier runner."""

from __future__ import annotations

import json
import logging
import shlex

from pydantic import BaseModel

from benchflow.env import resolve_env_vars
from benchflow.paths import EnvironmentPaths, TrialPaths
from benchflow.task import Task


class VerifierResult(BaseModel):
    rewards: dict[str, float | int | str | list | dict] | None = None


class AddTestsDirError(Exception):
    pass


class VerifierOutputParseError(Exception):
    pass


class DownloadVerifierDirError(Exception):
    pass


class RewardFileNotFoundError(FileNotFoundError):
    pass


class RewardFileEmptyError(Exception):
    pass


class Verifier:
    """Upload tests, run ``test.sh``, and parse verifier reward artifacts."""

    def __init__(
        self,
        task: Task,
        trial_paths: TrialPaths,
        environment,
        extra_env: dict[str, str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self._task = task
        self._trial_paths = trial_paths
        self._environment = environment
        self._extra_env = dict(extra_env or {})
        self._logger = (logger or logging.getLogger(__name__)).getChild("verifier")

    def _parse_reward_text(self) -> dict[str, float]:
        if self._trial_paths.reward_text_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                f"Reward file is empty at {self._trial_paths.reward_text_path}"
            )
        text = self._trial_paths.reward_text_path.read_text().strip()
        first_line = text.splitlines()[0] if text else ""
        try:
            return {"reward": float(first_line)}
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from text file {self._trial_paths.reward_text_path}"
            ) from e

    def _parse_reward_json(self) -> dict:
        if self._trial_paths.reward_json_path.stat().st_size == 0:
            raise RewardFileEmptyError(
                f"Reward file is empty at {self._trial_paths.reward_json_path}"
            )
        try:
            parsed = json.loads(self._trial_paths.reward_json_path.read_text())
        except (ValueError, TypeError) as e:
            raise VerifierOutputParseError(
                f"Failed to parse rewards from JSON file {self._trial_paths.reward_json_path}"
            ) from e
        if not isinstance(parsed, dict):
            raise VerifierOutputParseError(
                f"Reward JSON must be an object: {self._trial_paths.reward_json_path}"
            )
        return parsed

    def _merge_evaluation_details(self, rewards: dict) -> dict:
        """Lift common LLM-judge details into the public reward payload."""
        details_path = self._trial_paths.verifier_dir / "evaluation_details.json"
        if not details_path.exists():
            return rewards
        try:
            details = json.loads(details_path.read_text())
        except json.JSONDecodeError:
            return rewards
        if not isinstance(details, dict):
            return rewards
        enriched = dict(rewards)
        enriched.setdefault("details", details)
        results = details.get("results")
        if isinstance(results, list) and "rubric" not in enriched:
            rubric = []
            for item in results:
                if not isinstance(item, dict):
                    continue
                verdict = str(item.get("verdict", "")).lower()
                score = 1.0 if verdict == "pass" else 0.0
                rubric.append(
                    {
                        "name": item.get("id") or item.get("title") or "criterion",
                        "score": score,
                        "title": item.get("title"),
                        "verdict": item.get("verdict"),
                        "reasoning": item.get("reasoning"),
                    }
                )
            if rubric:
                enriched["rubric"] = rubric
        return enriched

    async def verify(self) -> VerifierResult:
        try:
            await self._environment.upload_dir(
                source_dir=self._task.paths.tests_dir,
                target_dir=str(EnvironmentPaths.tests_dir),
            )
        except Exception as e:
            raise AddTestsDirError("Failed to add tests directory to environment.") from e

        self._trial_paths.test_stdout_path.touch()

        base_env = self._task.config.verifier.env or {}
        verifier_env = {**base_env, **self._extra_env}
        env = None
        if verifier_env:
            for key in verifier_env:
                if "api_key" in key.lower():
                    self._logger.info(
                        "The verifier.env contains an API key; LLM judge calls may incur costs."
                    )
            env = resolve_env_vars(verifier_env)

        test_script_path = shlex.quote(
            str(
                EnvironmentPaths.tests_dir
                / self._task.paths.test_path.relative_to(
                    self._task.paths.tests_dir
                ).as_posix()
            )
        )
        test_stdout_path = shlex.quote(
            str(
                EnvironmentPaths.verifier_dir
                / self._trial_paths.test_stdout_path.relative_to(
                    self._trial_paths.verifier_dir
                ).as_posix()
            )
        )
        await self._environment.exec(f"chmod +x {test_script_path}", user="root")
        await self._environment.exec(
            command=f"{test_script_path} > {test_stdout_path} 2>&1",
            env=env,
            user=self._task.config.verifier.user,
        )

        if not getattr(self._environment, "is_mounted", False):
            try:
                await self._environment.download_dir(
                    source_dir=str(EnvironmentPaths.verifier_dir),
                    target_dir=self._trial_paths.verifier_dir,
                )
            except Exception as e:
                raise DownloadVerifierDirError(
                    "Failed to download verifier directory from environment"
                ) from e

        if self._trial_paths.reward_json_path.exists():
            rewards = self._parse_reward_json()
        elif self._trial_paths.reward_text_path.exists():
            rewards = self._parse_reward_text()
        else:
            raise RewardFileNotFoundError(
                f"No reward file found at {self._trial_paths.reward_text_path} "
                f"or {self._trial_paths.reward_json_path}"
            )

        return VerifierResult(rewards=self._merge_evaluation_details(rewards))
