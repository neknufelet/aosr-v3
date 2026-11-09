"""命令列考卷的同步分段：各工人只做正在讀的段落，上游收據完成後才啟動下游。"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import cast

import pytest

from tests.engine import test_material_gradient_cli as cases
from tests.engine._scheme_cache import shared_json
from tests.engine._split_probe_cache import Record, Sections

START_SURROGATES = {
    "unvalidated": "unvalidated", "zero_identity": "zero-identity", "outside_family": "good",
    "five_presets": "good", "moved_speaker": "good", "moved_seat": "good",
    "other_density": "other-density", "other_speed": "other-speed",
    "narrow-axis": "narrow-axis", "missing-point": "missing-point", "under_searches": "good",
}
RESUME_REJECTIONS = ("resume_other_identity", "resume_scheme_changed", "resume_project_source_points",
                     "resume_project_receiver_points", "resume_narrow_axis")


def _prepare(directory: Path) -> str:
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    cases._inputs(directory)
    return "{}"


class _Runs:
    def __init__(self, directory: Path, request: pytest.FixtureRequest, factory: pytest.TempPathFactory,
                 worker: str) -> None:
        self.directory, self.request = directory, request
        self.sections = Sections(factory, worker, "material-gradient-cli", directory, self.produce, self.prepare)

    def surrogate(self, name: str) -> None:
        group = "base" if name in ("good", "unvalidated", "zero-identity") else name
        self.sections.load(f"surrogate-{group}")

    def folder(self) -> Path:
        concurrent = cast(Record, self.sections.load("concurrent")["concurrent"])
        return Path(str(concurrent["folder"]))

    def prepare(self, group: str) -> None:
        """在拿本段的鎖與計時之前完成上游；下游行程不負責等檔。"""
        if group.startswith("surrogate-"):
            return
        if group.startswith("start-"):
            self.surrogate(START_SURROGATES[group.removeprefix("start-")])
            return
        if group.startswith("probe-"):
            if group == "probe-reference":
                self.surrogate("air-other")
            else:
                self.sections.load("probe-reference")
            self.folder()
            return
        self.surrogate("good")
        if group not in ("concurrent", "interrupted"):
            self.folder()
        if group == "resume-rejections":
            self.surrogate("narrow-axis")

    def produce(self, group: str) -> Record:
        directory = self.directory
        if group.startswith("surrogate-"):
            cases._surrogates(directory, self.request, group.removeprefix("surrogate-"))
            return {"ready": True}
        if group.startswith("start-"):
            name = group.removeprefix("start-")
            return {name: cases._start_rejection(directory, name)}
        if group.startswith("probe-"):
            if group == 'probe-reference':
                return cases._probe(directory, self.folder(), "all")
            part = group.removeprefix('probe-')
            isolated = cases._probe(directory, self.folder(), 'rebuild-'+part)
            shared = self.sections.load('probe-reference')
            combined = [*cast(list[Record], shared[part]), *cast(list[Record], isolated[part])]
            return {part: sorted(combined, key=lambda row: cast(int, row['at' if part == 'stops' else 'keep']))}
        if group == "concurrent":
            return {group: cases._concurrent_start(directory)}
        if group == "interrupted":
            return {group: cases._interrupted(directory)}
        return self.after_start(group)

    def after_start(self, group: str) -> Record:
        directory, folder = self.directory, self.folder()
        if group == "report":
            return {group: cases._run(directory, cases._command("report", "--folder", str(folder)))}
        if group == "other_settings":
            return {group: cases._other_settings(directory)}
        if group == "cli_stop":
            return {group: cases._cli_stop(directory, folder)}
        if group == "resume-rejections":
            return cases._rejections(directory, folder)
        raise ValueError(group)

    def output(self) -> Record:
        output = self.sections.record({key: key for key in (
            "concurrent", "report", "other_settings", "interrupted", "cli_stop")})
        output["rejections"] = self.sections.record({
            **{name: f"start-{name}" for name in START_SURROGATES},
            **{name: "resume-rejections" for name in RESUME_REJECTIONS}})
        output["probe"] = self.sections.record({
            **{key: "probe-reference" for key in ("reference", "phases", "pre_loop_rows", "other_header", "air", "control")},
            "stops": "probe-stops", "cuts": "probe-cuts"})
        return output


def split_runs(request: pytest.FixtureRequest, factory: pytest.TempPathFactory, worker: str) -> tuple[Path, Record]:
    directory = (factory.mktemp("material-gradient-cli") if worker == "master"
                 else factory.getbasetemp().parent / "material-gradient-cli-shared")
    shared_json(factory, worker, "material-gradient-cli-split-v1-inputs", lambda: _prepare(directory))
    return directory, _Runs(directory, request, factory, worker).output()
