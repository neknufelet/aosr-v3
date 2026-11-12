"""#700 網頁開第一、二階段擺位搜尋的後端：表單補設定與檢查、開跑的鏈、停止的兩條路、接續、整台一次一件、放進結果清單。

開服務那一層換成假的（只記下被叫了什麼、服務在不在由考卷控制），搜尋命令列不真的跑；狀態靠考卷自己寫的
代號檔、狀態檔與離開碼檔推。真正跑子行程的那一層（``search_exec``）用真的子行程考：接不接下一步、轉一次訊號。
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Literal, cast

import httpx
import pytest

from aosr.gui import search_jobs as search_jobs_module
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_exec import StepExit, write_exit
from aosr.gui.gradient_jobs import ActionRefused
from aosr.gui.search_exec import Chain, ChainStep
from aosr.gui.search_jobs import SearchBusy, SearchJobs, SettingsRefused
from aosr.reporting.evaluation import purpose_settings
from aosr.reporting.scheme import Scheme
from aosr.config.paths import config_path
from aosr.runtime import child_process_env
from aosr.search import cli
from aosr.search.outer_status import OUTER_MESSAGES, OuterStatus, snapshot_of
from aosr.search.run import RefineStatus, SearchStatus
from aosr.search.select import SelectOutcome
from aosr.search.settings import SearchSettings
from aosr.search.store import SearchIdentity, SearchStore
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine._search_speaker_setup_cases import flow_project
from tests.engine._search_store_cases import settings_document
from tests.engine.test_gui_gradient_jobs import FakeLauncher

NAME = "room"
CAPABILITIES = config_path("capabilities.toml")
Record = dict[str, object]


def _body() -> Record:
    """表單送來的樣子：沒有用途、沒有喇叭高與箱體（方案有喇叭設定，照方案補）。"""
    document = settings_document()
    document.pop("purpose")
    layout = cast(Record, document["layout"])
    layout.pop("speaker_height_m")
    layout.pop("cabinet")
    return document


def _jobs(tmp_path: Path, launcher: FakeLauncher | None = None, others: str | None = None) -> SearchJobs:
    (tmp_path / "schemes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "schemes" / f"{NAME}.json").write_text(flow_project("stand").model_dump_json(), encoding="utf-8")
    return SearchJobs(tmp_path, launcher=launcher or FakeLauncher(), engine_commit="c" * 40, capabilities=CAPABILITIES,
                      scheme_lock=threading.Lock(), others=lambda: others, python="py")


def _folder(jobs: SearchJobs, job_id: str, status: SearchStatus | None = SearchStatus()) -> Path:
    """照命令列建一個真的搜尋資料夾（快照都在，SearchStore.open 讀得回），寫代號檔；status 給了就寫狀態檔。"""
    project = flow_project("stand")
    settings = SearchSettings.model_validate(json.loads((jobs.root / job_id / "settings.json").read_text(encoding="utf-8")))
    identity = SearchIdentity("phys-test", "calc-test",
                              purpose_settings(config_path("quality_targets.toml"), project.purpose))
    store = SearchStore.create(jobs.searches, project=project, settings=settings, identity=identity,
                               versions={"python": "t", "optuna": "t", "numpy": "t"})
    (jobs.root / job_id / "search.id").write_text(store.search_id + "\n", encoding="utf-8")
    if status is not None:
        (store.path / "status.json").write_text(status.model_dump_json(), encoding="utf-8")
    return store.path


def _finish(jobs: SearchJobs, job_id: str, index: int, *, start: int | None, auto: int | None,
            signals: tuple[int, ...] = ()) -> None:
    """照 search_exec 的樣子寫每一小步與整條的離開碼檔（整條＝最後跑的那一步）。"""
    steps = jobs.root / job_id / "steps"
    last = None
    for name, code in (("start", start), ("auto", auto)):
        if code is not None:
            last = StepExit(exit_code=code, signals=signals, finished_at=time.time())
            write_exit(steps / f"{index}.{name}.exit.json", last)
    assert last is not None
    write_exit(steps / f"{index}.exit.json", last)


def _done_status(best: int | Literal["baseline"] | None = 7) -> SearchStatus:
    refine = RefineStatus(state="stopped", stop_reason="stable", best=best)
    status = SearchStatus(state="converged", refine=refine)
    return status.model_copy(update={"outer": OuterStatus(conclusion="complete", message=OUTER_MESSAGES["complete"],
                                                          snapshot=snapshot_of(status))})


# ---- 表單 → 設定 ----

def test_preview_takes_purpose_and_speaker_fields_from_the_scheme(tmp_path: Path) -> None:
    """抓表單漏寫的用途、喇叭高、箱體沒照方案補（命令列同一支），或預覽漏了方案事實與忙不忙。"""
    jobs = _jobs(tmp_path)
    scheme = flow_project("stand")
    assert scheme.speaker_setup is not None
    answer = jobs.preview(NAME, _body())
    settings = cast(Record, answer["settings"])
    layout = cast(Record, settings["layout"])
    assert settings["purpose"] == scheme.purpose
    assert layout["speaker_height_m"] == scheme.speakers["left"].z
    assert layout["cabinet"] == scheme.speaker_setup.cabinet.model_dump(mode="json")
    facts = cast(Record, answer["facts"])
    assert facts["has_speaker_setup"] is True and "書架喇叭" in cast(str, facts["speaker_setup"])
    assert facts["front_wall_guess"] == "x0" and answer["busy"] is None
    # 表單硬寫別的用途也照方案：用途不是表單能選的。
    assert cast(Record, jobs.preview(NAME, _body() | {"purpose": "other"})["settings"])["purpose"] == scheme.purpose


@pytest.mark.parametrize("change,words", [
    ({"budget": 0}, "budget"),
    ({"layout": {"front_wall": "x0"}}, "front_distance_m"),
    ({"extra": 1}, "extra"),
])
def test_preview_refuses_settings_with_the_field_named(tmp_path: Path, change: Record, words: str) -> None:
    """抓過不了設定檢查卻被放行，或拒收原因沒點出哪一格。"""
    jobs = _jobs(tmp_path)
    with pytest.raises(SettingsRefused, match=words):
        jobs.preview(NAME, _body() | change)


def test_preview_runs_the_folder_checks_without_writing(tmp_path: Path) -> None:
    """抓預覽沒跑建資料夾前那一道（鎖座位時寫了聆聽距離要拒收），或預覽就建了搜尋資料夾。"""
    jobs = _jobs(tmp_path)
    body = _body()
    cast(Record, body["layout"])["seat_locked"] = True
    with pytest.raises(SettingsRefused, match="座位鎖定時不准寫聆聽距離範圍"):
        jobs.preview(NAME, body)
    assert not any(jobs.searches.iterdir()) if jobs.searches.exists() else True


def test_cli_start_writes_the_new_folder_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """抓命令列建好資料夾卻沒寫代號檔（網頁就認不得自己開的是哪一場）。"""
    project = flow_project("stand")
    project_path, settings_path = tmp_path / "project.json", tmp_path / "settings.json"
    project_path.write_text(project.model_dump_json(), encoding="utf-8")
    settings_path.write_text(json.dumps(_body() | {"purpose": project.purpose}), encoding="utf-8")
    identity = SearchIdentity("phys-test", "calc-test", purpose_settings(config_path("quality_targets.toml"), project.purpose))
    monkeypatch.setattr(cli, "_identity", lambda purpose, capabilities: identity)
    store = cli._create(argparse.Namespace(project=project_path, settings=settings_path, root=tmp_path / "searches",
                                           capabilities=CAPABILITIES, id_file=tmp_path / "search.id"))
    assert (tmp_path / "search.id").read_text(encoding="utf-8") == store.search_id + "\n"


# ---- 開跑 ----

def test_start_launches_one_chain_of_start_then_auto(tmp_path: Path) -> None:
    """抓開跑沒照「start 建資料夾寫代號、auto 補在後面」接、搜尋資料夾不在網頁的搜尋區、快照不是預覽那一份，或服務名不對。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    view = jobs.start(NAME, _body())
    job_id = cast(str, view["job_id"])
    assert view["state"] == "starting" and cast(Record, view["actions"])["stop"] is True
    unit, argv, env, _ = launcher.launched[0]
    assert unit == f"aosr-search-{job_id}-0" and env["OMP_NUM_THREADS"] == "1"
    assert argv[:3] == ["py", "-m", "aosr.gui.search_exec"]
    chain = Chain.model_validate_json(Path(argv[3]).read_text(encoding="utf-8"))
    directory = tmp_path / "search-jobs" / job_id
    start, auto = chain.steps
    assert start.argv[:4] == ("py", "-m", "aosr.search.cli", "start") and not start.append_folder
    assert start.argv[start.argv.index("--root") + 1] == str(tmp_path / "searches")
    assert start.argv[start.argv.index("--id-file") + 1] == str(directory / "search.id") == chain.id_file
    assert start.argv[start.argv.index("--engine-commit") + 1] == "c" * 40
    assert auto.argv[3] == "auto" and auto.append_folder
    assert auto.argv[auto.argv.index("--modal-cache-dir") + 1] == str(tmp_path / "modal-cache")
    assert json.loads((directory / "settings.json").read_text(encoding="utf-8")) == jobs.preview(NAME, _body())["settings"]
    assert (directory / "scheme.json").read_text(encoding="utf-8") == (tmp_path / "schemes" / f"{NAME}.json").read_text(
        encoding="utf-8")


def test_only_one_heavy_job_on_the_machine(tmp_path: Path) -> None:
    """抓第二件被放行：自己這邊還在跑、別處的重計算服務、網頁裡第三階段在算、命令列開的搜尋拿著資料夾，四種都要擋。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    first = jobs.start(NAME, _body())
    with pytest.raises(SearchBusy, match=cast(str, first["job_id"])):
        jobs.start(NAME, _body())
    assert len(launcher.launched) == 1
    other = FakeLauncher()
    other.active = {"aosr-gradient-x-0"}
    with pytest.raises(SearchBusy, match="aosr-gradient-x-0"):
        _jobs(tmp_path / "b", other).start(NAME, _body())
    with pytest.raises(SearchBusy, match="第三階段"):
        _jobs(tmp_path / "c", others="第三階段 abc").start(NAME, _body())
    held = _jobs(tmp_path / "d")
    held.searches.mkdir()
    folder = held.searches / ("e" * 32)
    folder.mkdir()
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(SearchBusy, match="e" * 32):
            held.start(NAME, _body())
    finally:
        os.close(descriptor)
    assert held.start(NAME, _body())["state"] == "starting"


# ---- 狀態 ----

def test_states_follow_the_id_file_status_and_exit_records(tmp_path: Path) -> None:
    """抓狀態判錯：資料夾建好前後、start 做完進 auto、auto 做完、單題上限暫停、中斷、命令列回中斷。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id)
    view = jobs.get(job_id)
    assert view["state"] == "searching" and view["search_url"] == f"/searches/{folder.name}"
    assert "細算：未開始" in cast(str, view["summary"])  # 進度頁清單同一支的那一行
    write_exit(jobs.root / job_id / "steps" / "0.start.exit.json", StepExit(exit_code=0, finished_at=time.time()))
    assert jobs.get(job_id)["state"] == "finishing"
    launcher.active.clear()
    assert jobs.get(job_id)["state"] == "interrupted"
    assert cast(Record, jobs.get(job_id)["actions"])["resume"] is True
    (folder / "status.json").write_text(_done_status().model_dump_json(), encoding="utf-8")
    _finish(jobs, job_id, 0, start=0, auto=0)
    view = jobs.get(job_id)
    assert view["state"] == "done" and view["actions"] == {"stop": False, "resume": False, "select": True}
    assert view["refine_best"] == 7 and view["outer"] == OUTER_MESSAGES["complete"]
    _finish(jobs, job_id, 0, start=4, auto=None)
    assert jobs.get(job_id)["state"] == "paused_time_limit"
    _finish(jobs, job_id, 0, start=0, auto=3)
    assert jobs.get(job_id)["state"] == "interrupted"
    (jobs.root / job_id / "steps" / "0.auto.exit.json").unlink()
    _finish(jobs, job_id, 0, start=0, auto=None)
    assert jobs.get(job_id)["state"] == "stopped"  # start 正常結束卻沒接 auto：有人放了停止記號


def test_start_that_never_made_a_folder_is_failed_without_resume(tmp_path: Path) -> None:
    """抓命令列一開始就拒收（沒有資料夾）還給接續。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    launcher.active.clear()
    _finish(jobs, job_id, 0, start=1, auto=None)
    view = jobs.get(job_id)
    assert view["state"] == "failed" and view["actions"] == {"stop": False, "resume": False, "select": False}
    with pytest.raises(ActionRefused, match="不能接續"):
        jobs.resume(job_id)


# ---- 停止與接續 ----

def test_stop_puts_both_markers_while_the_outer_loop_is_open(tmp_path: Path) -> None:
    """抓停止沒放搜尋與細算兩個記號（批界才停）、或去停服務。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id)
    view = jobs.stop(job_id)
    assert (folder / "stop").is_file() and (folder / "refine-stop").is_file() and launcher.stopped == []
    assert view["state"] == "stopping"
    launcher.active.clear()
    assert jobs.get(job_id)["state"] == "stopped"
    with pytest.raises(ActionRefused, match="沒有在算"):
        jobs.stop(job_id)


@pytest.mark.parametrize("concluded", [False, True])
def test_stop_stops_the_service_before_the_folder_or_after_the_conclusion(tmp_path: Path, concluded: bool) -> None:
    """抓資料夾還沒建好、或外圈已有結論（正在補附件）時只放記號——那時記號停不到。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id, _done_status()) if concluded else None
    jobs.stop(job_id)
    assert launcher.stopped == [f"aosr-search-{job_id}-0"]
    assert folder is None or not (folder / "stop").exists()


def test_resume_runs_auto_alone_on_the_known_folder(tmp_path: Path) -> None:
    """抓接續又跑 start、沒把資料夾直接寫進命令列、或整台忙時照樣開。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id)
    launcher.active.clear()
    launcher.active.add("aosr-gradient-y-0")
    with pytest.raises(SearchBusy):
        jobs.resume(job_id)
    launcher.active.clear()
    view = jobs.resume(job_id)
    assert view["state"] == "finishing"
    unit, argv, _, _ = launcher.launched[-1]
    assert unit == f"aosr-search-{job_id}-1"
    chain = Chain.model_validate_json(Path(argv[3]).read_text(encoding="utf-8"))
    (step,) = chain.steps
    assert step.name == "auto" and step.argv[-1] == str(folder) and not step.append_folder
    assert chain.folder == str(folder)


# ---- 放進結果清單 ----

@pytest.mark.parametrize("best,trial", [(7, 7), ("baseline", None)])
def test_select_puts_the_refined_first_into_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                    best: int | Literal["baseline"], trial: int | None) -> None:
    """抓放進結果清單時拿錯候選（原方案要傳 None）、放到別的資料夾，或還沒做完就准放。"""
    launcher = FakeLauncher()
    jobs = _jobs(tmp_path, launcher)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id, _done_status(best))
    with pytest.raises(ActionRefused):
        jobs.select(job_id)
    calls: list[tuple[Path, int | None, Path]] = []

    def fake(store: SearchStore, trial_number: int | None, data_dir: Path) -> SelectOutcome:
        calls.append((store.path, trial_number, data_dir))
        return SelectOutcome(run_id="r" * 32, result_path=data_dir / "results" / "x.json", is_new=True)

    monkeypatch.setattr(search_jobs_module, "select_refined", fake)
    launcher.active.clear()
    _finish(jobs, job_id, 0, start=0, auto=0)
    answer = jobs.select(job_id)
    assert calls == [(folder, trial, tmp_path.resolve())]
    assert answer["url"] == f"/results/{'r' * 32}"


def test_previous_lists_each_search_settings_without_purpose(tmp_path: Path) -> None:
    """抓「從上一場帶入」帶了用途（用途照方案）、或漏了某一場。"""
    jobs = _jobs(tmp_path)
    job_id = cast(str, jobs.start(NAME, _body())["job_id"])
    folder = _folder(jobs, job_id)
    (jobs.searches / ("f" * 32)).mkdir()
    answer = jobs.previous()
    (item,) = cast(list[Record], answer["searches"])
    assert item["search_id"] == folder.name and item["scheme_id"] == flow_project("stand").scheme_id
    assert "purpose" not in cast(Record, item["settings"]) and item["stage_one"] is False
    assert len(cast(list[str], answer["skipped"])) == 1


# ---- 網址 ----

def test_routes_answer_listing_preview_errors_and_busy(tmp_path: Path) -> None:
    """抓網址沒接上、壞 JSON 不回 400、忙的時候不回 409。"""
    data = tmp_path / "data"
    (data / "schemes").mkdir(parents=True)
    (data / "schemes" / f"{NAME}.json").write_text(flow_project("stand").model_dump_json(), encoding="utf-8")
    launcher = FakeLauncher()
    app = create_app(GuiSettings(engine_commit="c" * 40, data_dir=data, gradient_launcher=launcher,
                                 startup_fingerprint="f", startup_physics_identity="p"))

    async def calls() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as client:
            return [await client.get("/api/search-jobs"),
                    await client.post(f"/api/search-jobs/schemes/{NAME}/preview", content=b"{",
                                        headers={"content-type": "application/json"}),
                    await client.post(f"/api/search-jobs/schemes/{NAME}/preview", json=_body()),
                    await client.post(f"/api/search-jobs/schemes/{NAME}/start", json=_body()),
                    await client.post(f"/api/search-jobs/schemes/{NAME}/start", json=_body()),
                    await client.get(f"/api/search-jobs/schemes/{NAME}")]

    listing, broken, preview, started, busy, facts = asyncio.run(calls())
    assert listing.status_code == 200 and listing.json()["jobs"] == []
    assert broken.status_code == 400 and "JSON" in broken.json()["error"]
    assert preview.status_code == 200 and preview.json()["settings"]["budget"] == 11
    assert started.status_code == 200 and started.json()["state"] == "starting"
    assert busy.status_code == 409 and busy.json()["busy"] == started.json()["job_id"]
    assert facts.status_code == 200 and facts.json()["scheme_id"] == flow_project("stand").scheme_id


# ---- 服務裡那一層 ----

def _script(text: str) -> tuple[str, ...]:
    return (sys.executable, "-c", text)


def _run_chain(tmp_path: Path, start: str, auto: str, *, signal_after: Path | None = None) -> tuple[StepExit, Path]:
    root = tmp_path / "searches"
    root.mkdir()
    chain = Chain(log=str(tmp_path / "log"), done=str(tmp_path / "done.json"), search_root=str(root),
                  id_file=str(tmp_path / "search.id"),
                  steps=(ChainStep(name="start", argv=_script(start), record=str(tmp_path / "start.json")),
                         ChainStep(name="auto", argv=_script(auto), record=str(tmp_path / "auto.json"), append_folder=True)))
    (tmp_path / "chain.json").write_text(chain.model_dump_json(), encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-m", "aosr.gui.search_exec", str(tmp_path / "chain.json")],
                               env=forward_cpu_pins(child_process_env()), cwd=tmp_path)
    if signal_after is not None:
        deadline = time.monotonic() + 60
        while not signal_after.exists():
            assert time.monotonic() < deadline and process.poll() is None
            time.sleep(0.05)
        process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=60) == 0
    return StepExit.model_validate_json((tmp_path / "done.json").read_text(encoding="utf-8")), root


MAKE_FOLDER = ("import os,sys; d=os.path.join({root!r}, 'a'*32); os.makedirs(d); {extra}"
               "open({id_file!r},'w').write('a'*32+'\\n'); sys.exit({code})")


def _start(tmp_path: Path, *, code: int = 0, extra: str = "") -> str:
    return MAKE_FOLDER.format(root=str(tmp_path / "searches"), id_file=str(tmp_path / "search.id"), code=code, extra=extra)


def test_chain_appends_the_new_folder_to_auto_after_start_succeeds(tmp_path: Path) -> None:
    """抓 auto 沒拿到 start 剛建的資料夾，或整條的離開碼不是最後一步的。"""
    seen = tmp_path / "auto-argv.json"
    done, root = _run_chain(tmp_path, _start(tmp_path),
                            f"import json,sys; json.dump(sys.argv[1:], open({str(seen)!r},'w')); sys.exit(5)")
    assert json.loads(seen.read_text(encoding="utf-8")) == [str(root / ("a" * 32))]
    assert done.exit_code == 5 and done.signals == ()
    assert StepExit.model_validate_json((tmp_path / "start.json").read_text(encoding="utf-8")).exit_code == 0


@pytest.mark.parametrize("code,extra", [(1, ""), (0, "open(os.path.join(d,'stop'),'w'); "),
                                        (0, "open(os.path.join(d,'refine-stop'),'w'); ")])
def test_chain_does_not_open_auto_after_failure_or_a_stop_marker(tmp_path: Path, code: int, extra: str) -> None:
    """抓 start 失敗、或停止記號已放了，照樣接 auto。"""
    done, _ = _run_chain(tmp_path, _start(tmp_path, code=code, extra=extra), "import sys; sys.exit(9)")
    assert done.exit_code == code and not (tmp_path / "auto.json").exists()


def test_chain_forwards_one_sigterm_and_stops_there(tmp_path: Path) -> None:
    """抓停服務時這一層先走、訊號沒轉給子行程、轉兩次，或收過訊號還接 auto。"""
    ready, count = tmp_path / "ready", tmp_path / "count"
    start = (f"import signal,sys,time\nseen=[]\nsignal.signal(signal.SIGTERM, lambda s,f: seen.append(s))\n"
             f"open({str(ready)!r},'w').close()\nend=time.time()+3\n"
             f"while time.time()<end: time.sleep(0.05)\nopen({str(count)!r},'w').write(str(len(seen)))\nsys.exit(3)")
    done, _ = _run_chain(tmp_path, start, "import sys; sys.exit(0)", signal_after=ready)
    assert done.exit_code == 3 and done.signals == (signal.SIGTERM,)
    assert count.read_text(encoding="utf-8") == "1" and not (tmp_path / "auto.json").exists()


def test_scheme_without_speaker_setup_needs_height_and_cabinet(tmp_path: Path) -> None:
    """抓方案沒有喇叭設定時悄悄補出喇叭高或箱體（那兩格要表單自己寫）。"""
    jobs = _jobs(tmp_path)
    document = flow_project("stand").model_dump(mode="json")
    document.pop("speaker_setup")
    document["furniture"] = None
    (tmp_path / "schemes" / "plain.json").write_text(json.dumps(document), encoding="utf-8")
    assert Scheme.model_validate(document).speaker_setup is None
    with pytest.raises(SettingsRefused, match="speaker_height_m"):
        jobs.preview("plain", _body())
    full = settings_document()
    full.pop("purpose")
    assert cast(Record, jobs.preview("plain", full)["settings"])["layout"] is not None
