"""#1082 段 2-8 子行程：小房間真的建替身（真的完整解、真的留出驗證），但把固定材料候選自己的訓練與留出組拿掉，
讓替身真的沒學過它；這間小房間沒學過也只差約 0.0006 dB，所以候選核對（只有這一道）在這個子行程裡用考卷自己的窄尺
``TIGHT_DB``（產品登記簿的兩把尺不動，留出驗證照登記簿判），開跑那一道固定材料核對一定先沒過；
看建替身會不會在裝進格子之前就核、補訓練、過了才裝。

``argv``：工作資料夾、模式。``stall``：預算夠，補訓練的樣本殘差卻已經在停止值內、補不進點（這間小房間只有一片交給搜尋，
跟核對那一組不同的只剩「其他片也換隨機家族」那幾組）——要停、不裝、照實講；``cap``：建完就把這一次的預算收到剛好用完，
沒有餘額補訓練，要不裝、講加大預算；``pass``：建完把停止殘差收到 ``PASS_STOP``（只這一次建模、考卷自己的值），
補訓練真的解了候選那幾組、核對過了才做留出驗證、裝進格子、留紀錄，預覽才說可以直接開搜尋。命令列參數由網頁那一層（``SurrogateBuilds.start``，假的開服務層）照產品組出來，
這裡在同一個行程叫 ``surrogate_build.build``（才換得了訓練組）；結果寫 ``probe.json``。

#1142：``pass`` 模式裝好之後，再由網頁那一層照產品組出開跑的命令列參數，在同一個行程叫命令列開跑那一段
（``cli._start``，只把「建資料夾、起子行程」換成記下 ``project.json`` 那一份），跑兩次：一次照產品（核對紀錄在、
身分相同 → 沿用），一次先把核對紀錄移開（→ 照舊重核，真的完整解）；兩次的固定材料核對與輸出都記進 ``starts``。
這兩次之前先把上面換掉的函式換回產品原樣（窄尺、記錄判決都不套到開跑那一段）。建替身那一段另記「量程式身分」與
「判候選」的先後（``events``；身分照產品原樣算），考建模之前量一次、寫紀錄之前再量一次。
"""
from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, cast

from aosr import runtime

if TYPE_CHECKING:
    from collections.abc import Callable

    from aosr.gui.gradient_jobs import GradientJobs
    from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances
    from aosr.gui.gradient_plan import GradientRequest

DIRECTORY, MODE = Path(sys.argv[1]), sys.argv[2]
FIXED = "panel_absorbing"
# 只給候選核對用的考卷窄尺（逐頻 dB）：沒學過候選的替身差約 0.0006 dB，補訓練後要壓到它以下。
TIGHT_DB = 0.0002
# ``pass`` 模式補訓練用的停止殘差（考卷自己的值）：讓補的那幾組真的被解進快照。
PASS_STOP = 1e-9
NAME = "zoned"


def _arguments() -> tuple[GradientJobs, GradientRequest, str, list[str]]:
    """網頁那一層（假的開服務層）照產品組出建替身的命令列參數；回工作管理器、預覽選項、建模代號、參數。"""
    from aosr.config.paths import config_path
    from aosr.gui.gradient_jobs import GradientJobs
    from aosr.gui.gradient_plan import GradientRequest
    from aosr.gui.material_editor import user_library_path
    from aosr.gui.surrogate_builds import SurrogateBuilds, SurrogateRequest
    from aosr.gui.zone_design import DesignSettings, ZoneDesign, write_design
    from aosr.materials import library
    from pydantic import TypeAdapter
    from tests.engine._material_gradient_zones_case import scheme
    from tests.engine.test_gui_gradient_jobs import FakeLauncher

    data_dir = DIRECTORY / "data"
    (data_dir / "schemes").mkdir(parents=True)
    (data_dir / "designs").mkdir()
    base = scheme()
    (data_dir / "schemes" / f"{NAME}.json").write_text(base.model_dump_json(), encoding="utf-8")
    zones = {"x0": {"right": ZoneDesign(mode="search", allowed=("glass_wool_48k", FIXED))}}
    write_design(data_dir, DesignSettings(scheme_id=NAME, zones=zones), base)
    entry: library.LibraryEntry = TypeAdapter(library.LibraryEntry).validate_python({
        "kind": "fixed", "id": FIXED, "label": "吸音板（考卷）", "provenance": "model", "source": "考卷",
        "material": {"kind": "constant_impedance", "impedance_pa_s_m": 450.0}})
    user_library_path(data_dir).write_text(library.dump_user_library((cast(library.FixedEntry, entry),)), encoding="utf-8")
    launcher = FakeLauncher()
    jobs = GradientJobs(data_dir, launcher=launcher, engine_commit="c" * 40, capabilities=config_path("capabilities.toml"),
                        scheme_lock=threading.RLock())
    search = GradientRequest(device="cpu")
    view = SurrogateBuilds(jobs).start(NAME, SurrogateRequest(workers=1, dense_threads=1, stop_residual=1e-4, search=search))
    (_, argv, _, _), = launcher.launched
    return jobs, search, str(view["build_id"]), argv[argv.index("aosr.gradient.surrogate_build") + 1:]


def _patch(seen: dict[str, list[object]]) -> dict[str, object]:
    """候選的訓練組拿掉、候選核對用窄尺、記下每次判決與補了哪些組；``cap`` 模式建完就把預算收到剛好用完、
    ``pass`` 模式把停止殘差收到 ``PASS_STOP``。換函式一律用 ``setattr``（換上去的是考卷自己的版本，型別不必跟原本一樣）。"""
    from aosr.gradient import fixed_candidates, run, surrogate_build
    from aosr.gradient.fixed_candidates import CandidateVerdict
    from aosr.gradient.surrogate_training import MaterialSet, TrainingSets
    from aosr.physics.fem_reduced_build import ReducedModelBuilder

    # 替身沒學過這個候選：訓練與留出都拿掉它自己的那幾組（其他照產品）。
    training_sets: Callable[[TrainingSets], list[MaterialSet]] = TrainingSets.training_sets
    holdout_sets: Callable[[TrainingSets], list[MaterialSet]] = TrainingSets.holdout_sets

    def without_candidate(sets: list[MaterialSet]) -> list[MaterialSet]:
        return [item for item in sets if not item.label.startswith("固定 ")]

    setattr(TrainingSets, "training_sets", lambda self: without_candidate(training_sets(self)))
    setattr(TrainingSets, "holdout_sets", lambda self: without_candidate(holdout_sets(self)))
    setattr(TrainingSets, "require_every_family", lambda self, training, holdout: None)
    judged, added, generated, events = seen["judged"], seen["added"], seen["generated"], seen["events"]
    configurations: dict[str, object] = {}
    judge: Callable[..., dict[str, CandidateVerdict]] = fixed_candidates.judge_candidates
    identity: Callable[..., str] = fixed_candidates.check_code_identity

    def measured(**kwargs: object) -> str:
        events.append("identity")  # #1142：建模之前量一次、寫紀錄之前再量一次（產品原樣算，只記先後）
        return identity(**kwargs)

    setattr(fixed_candidates, "check_code_identity", measured)
    rulers = run.holdout_rulers

    def tight() -> object:
        found = rulers()
        return dataclasses.replace(found, frequency=dataclasses.replace(found.frequency, value=TIGHT_DB))

    def recording(model: object, operators: object, found: dict[str, object], allowed: object,
                  **kwargs: object) -> dict[str, CandidateVerdict]:
        configurations.update(found)
        setattr(run, "holdout_rulers", tight)
        try:
            verdicts = judge(model, operators, found, allowed, **kwargs)
        finally:
            setattr(run, "holdout_rulers", rulers)
        events.append("judge")
        judged.append({target: {"frequency_db": verdict.frequency_db, "passed": verdict.passed,
                                "full_solves": verdict.report.full_solves,
                                "progress_given": kwargs.get("progress") is not None}
                       for target, verdict in verdicts.items()})
        return verdicts

    setattr(fixed_candidates, "judge_candidates", recording)
    checked: Callable[..., surrogate_build._Extension] = surrogate_build._fixed_checked
    extension_sets: Callable[..., list[MaterialSet]] = TrainingSets.extension_sets

    def counting(self: TrainingSets, item: object, **kwargs: object) -> list[MaterialSet]:
        found = extension_sets(self, item, **kwargs)
        generated.append(len(found))
        return found

    setattr(TrainingSets, "extension_sets", counting)

    def capped(job: surrogate_build._Run, builder: ReducedModelBuilder, model: object,
               prepared: surrogate_build._Prepared) -> surrogate_build._Extension:
        if MODE == "cap":
            job.settings = job.settings.model_copy(update={"max_full_solves": builder.full_solves})
            builder.settings = job.settings
        elif MODE == "pass":
            job.settings = job.settings.model_copy(update={"stop_residual": PASS_STOP})
            builder.settings = job.settings
        seen["before"].append({"full_solves": builder.full_solves, "batch": job.settings.batch,
                               "cap": job.settings.max_full_solves})
        extension = checked(job, builder, model, prepared)
        added.extend(prepared.sets.impedances(item) for item in extension.added)
        return extension

    setattr(surrogate_build, "_fixed_checked", capped)
    return configurations


def _starts(jobs: GradientJobs, search: GradientRequest, slot: Path, restore: Callable[[], None]) -> list[object]:
    """網頁照產品組出開跑參數、在這個行程叫開跑那一段兩次（沿用、移開紀錄重核）；回兩次的核對與輸出。"""
    from aosr.gradient import cli
    from aosr.gradient.run import GradientProject
    from aosr.gradient.surrogate_fixed_checks import FIXED_CHECKS_FILE
    from aosr.gradient.surrogate_files import MODEL_FILE

    restore()
    launcher = getattr(jobs, "launcher")
    launcher.active.clear()  # 建替身那一個服務（假的）做完了
    jobs.start(NAME, search)
    argv = list(launcher.launched[-1][1])
    arguments = argv[argv.index("aosr.gradient.cli") + 1:]
    captured: list[GradientProject] = []

    def keep(root: Path, project: GradientProject, snapshot: bytes | None = None) -> int:
        captured.append(project)
        return 0

    setattr(cli, "_create_and_run", keep)
    found: list[object] = []
    record = slot / FIXED_CHECKS_FILE
    for label in ("reuse", "recheck"):
        if label == "recheck":
            record.rename(slot / "moved-fixed-checks.json")
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            code = cli._start(cli._parser().parse_args(arguments))
        project, = captured[-1:]
        found.append({"label": label, "exit": code, "stderr": stderr.getvalue(),
                      "surrogate_is_slot": Path(arguments[arguments.index("--surrogate") + 1]) == slot / MODEL_FILE,
                      "checks": [check.model_dump(mode="json") for check in project.fixed_candidate_checks]})
    return found


def main() -> None:
    runtime.configure_calculation_jax()

    from aosr.gradient import fixed_candidates, run, surrogate_build
    from aosr.gradient.surrogate_training import TrainingSets

    jobs, search, build_id, arguments = _arguments()
    seen: dict[str, list[object]] = {"judged": [], "added": [], "generated": [], "before": [], "events": []}
    originals = {name: getattr(TrainingSets, name)
                 for name in ("training_sets", "holdout_sets", "require_every_family", "extension_sets")}
    judge, rulers, checked = fixed_candidates.judge_candidates, run.holdout_rulers, surrogate_build._fixed_checked
    identity = fixed_candidates.check_code_identity

    def restore() -> None:
        for name, value in originals.items():
            setattr(TrainingSets, name, value)
        setattr(fixed_candidates, "judge_candidates", judge)
        setattr(fixed_candidates, "check_code_identity", identity)
        setattr(run, "holdout_rulers", rulers)
        setattr(surrogate_build, "_fixed_checked", checked)

    configurations = _patch(seen)
    judged, added, generated = seen["judged"], seen["added"], seen["generated"]
    code = surrogate_build.build(surrogate_build._parser().parse_args(arguments))
    work = Path(arguments[arguments.index("--work") + 1])
    slot = Path(arguments[arguments.index("--slot") + 1])
    held_out = [any(surrogate_build.same_configuration(cast("FrequencyBoundaryImpedances", item),
                                                       cast("FrequencyBoundaryImpedances", held))
                    for held in configurations.values()) for item in added]
    training = json.loads((slot / "training.json").read_text(encoding="utf-8")) if slot.is_dir() else None
    record = json.loads((slot / "fixed_checks.json").read_text(encoding="utf-8")) if slot.is_dir() else None
    surrogate = None
    if slot.is_dir():
        surrogate = jobs.preview(NAME, search)["surrogate"]
    starts = _starts(jobs, search, slot, restore) if MODE == "pass" and code == 0 else None
    (DIRECTORY / "probe.json").write_text(json.dumps({
        "exit": code, "build_id": build_id, "arguments": arguments, "judged": judged, "before": seen["before"],
        "events": seen["events"],
        "added": len(added), "generated": generated, "added_equal_to_check": held_out,
        "result": json.loads((work / "result.json").read_text(encoding="utf-8")), "slot": slot.is_dir(),
        "training_labels": None if training is None else [item["label"] for item in training["training"]],
        "record": record, "preview": surrogate, "starts": starts,
        "rounds": [json.loads(line) for line in (work / "fixed_check_rounds.jsonl").read_text(encoding="utf-8").splitlines()]
        if (work / "fixed_check_rounds.jsonl").is_file() else None}, ensure_ascii=False, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
