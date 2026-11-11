"""#1106（#1082 段 1-6b）第三階段搜尋頁的伺服器補件：進度細節、終點方案存成新方案並開一般計算、頁面與靜態檔。

開服務那一層與一般計算都換成假的（假的開服務層記下被叫了什麼；一般計算的執行程式是「什麼都不做」的 Python）。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import cast

import httpx
import pytest

from aosr.gradient import folder_files
from aosr.gradient.cli import EXIT_STOPPED
from aosr.gui.app import GuiSettings, create_app
from aosr.gui.gradient_jobs import LaunchFailed
from aosr.config.paths import config_path
import aosr.gui.gradient_follow as follow_module
from aosr.gui.gradient_follow import ADOPTED_FILE, RIGID_BASELINE_WARNING, GradientFollow, adopted_name, descent_steps
from aosr.gui.jobs import JobManager
from aosr.gui.gradient_plan import GradientRequest
from aosr.gui.zone_design import read_design, write_design
from aosr.reporting.scheme import Scheme
from aosr.runtime import child_process_env
from tests.engine._cpu_pins import forward_cpu_pins
from tests.engine.test_gui_gradient_jobs import NAME, FakeLauncher, Record, _data, _finish, _folder, _jobs, _scheme

DESCENT = folder_files.LEDGER_FILE


def _ledger(folder: Path, descent_rows: int, *, extra: tuple[str, ...] = ()) -> None:
    """假帳本：標頭一列、descent 若干列（起點那一列也是 descent）、再接幾列別的階段。"""
    lines = [json.dumps({"ledger_version": "v6"})]
    lines.extend(json.dumps({"step": index, "phase": "descent"}) for index in range(descent_rows))
    lines.extend(json.dumps({"step": descent_rows + index, "phase": phase}) for index, phase in enumerate(extra))
    (folder / DESCENT).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_descent_steps_count_rows_after_the_starting_row(tmp_path: Path) -> None:
    """抓把起點那一列算成一步（60 步走滿會印 61／60）、把別的階段算進往下走、或讀不懂的列讓整頁壞掉。"""
    assert descent_steps(tmp_path) is None
    _ledger(tmp_path, 61, extra=("snap", "retune", "candidate_jump", "round_decision"))
    assert descent_steps(tmp_path) == 60
    with (tmp_path / DESCENT).open("a", encoding="utf-8") as stream:
        stream.write("不是 JSON\n")
    assert descent_steps(tmp_path) == 60
    _ledger(tmp_path, 1)
    assert descent_steps(tmp_path) == 0
    (tmp_path / DESCENT).write_text(json.dumps({"ledger_version": "v6"}) + "\n", encoding="utf-8")
    assert descent_steps(tmp_path) == 0


def _serve(data_dir: Path, launcher: FakeLauncher, scenario: Callable[[httpx.AsyncClient], Awaitable[None]]) -> None:
    async def go() -> None:
        app = create_app(GuiSettings(engine_commit="a" * 40, data_dir=data_dir, gradient_launcher=launcher,
                                     runner=(sys.executable, "-c", "pass")))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
            await scenario(client)
    asyncio.run(go())


def _log(data_dir: Path, view: Record, index: int, text: str) -> None:
    path = data_dir / "gradient" / str(view["job_id"]) / "steps" / f"{index}.log"
    path.write_text(text, encoding="utf-8")


def test_detail_shows_steps_notes_and_the_full_failure_line(tmp_path: Path) -> None:
    """抓步數上限沒照開跑時指定的值、注意行（空腔不調）沒撈出來、失敗原因只給最後 8 行而把真正那一行截掉，
    或還在跑、做完的步驟也被當成失敗撈原因。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest(max_steps=45))
    job_id = str(view["job_id"])
    seen: list[Record] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        empty = (await client.get(f"/api/gradient/runs/{job_id}/detail")).json()
        assert empty["descent_steps"] is None and empty["max_steps"] == 45 and empty["failures"] == []
        folder = _folder(data_dir, view, None)
        _ledger(folder, 11)
        _log(data_dir, view, 0, "注意：材料 my_wool 的空腔不調（固定 0.05 m）\n開始搜尋\n")
        running = (await client.get(f"/api/gradient/runs/{job_id}/detail")).json()
        assert running["descent_steps"] == 10 and running["notes"] == ["注意：材料 my_wool 的空腔不調（固定 0.05 m）"]
        assert running["failures"] == []
        _finish(data_dir, launcher, view, EXIT_STOPPED)
        seen.append((await client.get(f"/api/gradient/runs/{job_id}/detail")).json())
        resumed = jobs.resume(job_id)
        failure = "梯度搜尋接續失敗：MaterialsRefused：使用者材料庫快照被改過（SHA-256 不同），拒收"
        _log(data_dir, resumed, 1, "\n".join([failure, *(f"收尾第 {index} 行" for index in range(20))]) + "\n")
        _finish(data_dir, launcher, resumed, 1)
        seen.append((await client.get(f"/api/gradient/runs/{job_id}/detail")).json())

    _serve(data_dir, launcher, scenario)
    stopped, failed = seen
    assert stopped["failures"] == [] and stopped["state"] == "stopped"
    assert failed["state"] == "failed" and failed["failures"] == [
        "梯度搜尋接續失敗：MaterialsRefused：使用者材料庫快照被改過（SHA-256 不同），拒收"]
    assert failed["notes"] == ["注意：材料 my_wool 的空腔不調（固定 0.05 m）"]


def _done_job(data_dir: Path, launcher: FakeLauncher, exported: Record) -> str:
    """一跑走到終點做完：資料夾、終點檔、匯出的方案都在。"""
    jobs = _jobs(data_dir, launcher)
    view = jobs.start(NAME, GradientRequest())
    folder = _folder(data_dir, view, None)
    _finish(data_dir, launcher, view, 0)
    view = jobs.endpoint(str(view["job_id"]))
    _finish(data_dir, launcher, view, 0)
    (folder / folder_files.ENDPOINT_FILE).write_text("{}", encoding="utf-8")
    (folder / folder_files.EXPORTED_SCHEME_FILE).write_text(json.dumps(exported), encoding="utf-8")
    assert jobs.get(str(view["job_id"]))["state"] == "done"
    return str(view["job_id"])


def _exported() -> Record:
    document = cast(Record, json.loads(_scheme().model_dump_json()))
    document["scheme_id"] = f"{NAME}-trial-gradient-0123abcd"
    return document


def test_adopt_saves_the_endpoint_scheme_with_its_design_and_starts_one_run(tmp_path: Path) -> None:
    """抓沒做完就讓人存、匯出方案沒過網頁檢查就存、設計設定沒跟過去（結果頁算改善要它）、按兩次開兩筆計算、
    或結果頁的網址指錯。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    jobs = _jobs(data_dir, launcher)
    early = jobs.start(NAME, GradientRequest())
    early_id = str(early["job_id"])
    seen: list[Record] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        refused = await client.post(f"/api/gradient/runs/{early_id}/adopt", json={})
        assert refused.status_code == 409 and "終點做完" in refused.json()["error"]
        _finish(data_dir, launcher, early, 1)
        job_id = _done_job(data_dir, launcher, _exported())
        first = await client.post(f"/api/gradient/runs/{job_id}/adopt", content=b"{}",
                                  headers={"content-type": "application/json"})
        assert first.status_code == 200, first.text
        again = await client.post(f"/api/gradient/runs/{job_id}/adopt", json={})
        assert again.status_code == 200
        seen.extend((first.json(), again.json(), {"job_id": job_id}))

    _serve(data_dir, launcher, scenario)
    first, again, ids = seen
    job_id = str(ids["job_id"])
    name = adopted_name(NAME, job_id)
    adopted = cast(Record, first["adopted"])
    assert adopted["scheme_id"] == name == f"{NAME}-search-{job_id[:8]}"
    assert adopted["result_url"] == f"/results/{adopted['run_id']}"
    assert cast(Record, again["adopted"])["run_id"] == adopted["run_id"]
    saved = Scheme.model_validate_json((data_dir / "schemes" / f"{name}.json").read_text(encoding="utf-8"))
    assert saved.scheme_id == name and saved.scene == _scheme().scene
    design = read_design(data_dir, name)
    original = read_design(data_dir, NAME)
    assert design is not None and original is not None
    assert design.scheme_id == name and design.zones == original.zones
    runs = sorted((data_dir / "runs").glob("*.json"))
    assert [path.stem for path in runs] == [adopted["run_id"]]
    record = json.loads((data_dir / "gradient" / job_id / ADOPTED_FILE).read_text(encoding="utf-8"))
    assert record["run_id"] == adopted["run_id"] and record["scheme_id"] == name


def test_adopt_refuses_a_bad_export_or_a_different_scheme_under_the_same_name(tmp_path: Path) -> None:
    """抓匯出的方案壞了照存、把別人同名的方案蓋掉、或亂給的代號拿去組路徑。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    bad = _exported()
    cast(Record, bad["scene"])["room_m"] = "不是房間"
    bad_job = _done_job(data_dir, launcher, bad)
    seen: list[httpx.Response] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        seen.append(await client.post(f"/api/gradient/runs/{bad_job}/adopt", json={}))
        seen.append(await client.post(f"/api/gradient/runs/{'..'}/adopt", json={}))
        seen.append(await client.get(f"/api/gradient/runs/{'0' * 32}/detail"))

    _serve(data_dir, launcher, scenario)
    refused, traversal, missing = seen
    assert refused.status_code == 400 and "過不了網頁的方案檢查" in refused.json()["error"]
    assert not (data_dir / "schemes" / f"{adopted_name(NAME, bad_job)}.json").exists()
    assert traversal.status_code == 404 and missing.status_code == 404
    assert not (data_dir / ADOPTED_FILE).exists()

    other_dir = _data(tmp_path / "other")
    other = FakeLauncher()
    job_id = _done_job(other_dir, other, _exported())
    taken = cast(Record, json.loads(_scheme("single").model_dump_json())) | {"scheme_id": adopted_name(NAME, job_id)}
    (other_dir / "schemes" / f"{adopted_name(NAME, job_id)}.json").write_text(json.dumps(taken), encoding="utf-8")
    before = (other_dir / "schemes" / f"{adopted_name(NAME, job_id)}.json").read_bytes()
    seen.clear()

    async def taken_scenario(client: httpx.AsyncClient) -> None:
        seen.append(await client.post(f"/api/gradient/runs/{job_id}/adopt", json={}))

    _serve(other_dir, other, taken_scenario)
    assert seen[0].status_code == 409 and "內容不同" in seen[0].json()["error"]
    assert (other_dir / "schemes" / f"{adopted_name(NAME, job_id)}.json").read_bytes() == before


def test_page_and_static_files_are_served(tmp_path: Path) -> None:
    """抓頁面網址沒掛、靜態檔沒登記（瀏覽器拿到 404 整頁空白）、首頁沒有連過來。"""
    data_dir = _data(tmp_path)
    seen: list[httpx.Response] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        for path in ("/gradient", *(f"/static/gradient.{kind}" for kind in ("js", "css")), "/"):
            seen.append(await client.get(path))

    _serve(data_dir, FakeLauncher(), scenario)
    page, script, style, home = seen
    assert page.status_code == 200 and "第三階段搜尋" in page.text and 'src="/static/gradient.' in page.text
    assert script.status_code == 200 and "/api/gradient/runs/" in script.text
    assert style.status_code == 200 and home.status_code == 200 and 'href="/gradient"' in home.text


def test_web_process_with_this_page_never_imports_jax(tmp_path: Path) -> None:
    """抓這一支為了數帳本把物理（JAX）拉進網頁行程。"""
    code = ("import sys; import aosr.gui.app, aosr.gui.gradient_follow; "
            "raise SystemExit(1 if 'jax' in sys.modules else 0)")
    env = forward_cpu_pins(child_process_env())
    env["TMPDIR"] = str(tmp_path)
    done = subprocess.run([sys.executable, "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def test_detail_of_a_launch_failure_shows_the_service_error(tmp_path: Path) -> None:
    """抓開服務那一層失敗（沒留輸出）時細節頁整個讀不出來。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    launcher.fail = "systemd-run 找不到"
    jobs = _jobs(data_dir, launcher)
    with pytest.raises(LaunchFailed, match="systemd-run 找不到"):
        jobs.start(NAME, GradientRequest())
    job_id = next(path.name for path in (data_dir / "gradient").iterdir())
    seen: list[Record] = []

    async def scenario(client: httpx.AsyncClient) -> None:
        seen.append((await client.get(f"/api/gradient/runs/{job_id}/detail")).json())
        seen.append((await client.get(f"/api/gradient/jobs/{job_id}")).json())

    _serve(data_dir, FakeLauncher(), scenario)
    detail, view = seen
    assert detail["state"] == "failed" and detail["failures"] == []
    assert cast(list[Record], view["steps"])[0]["error"] == "systemd-run 找不到"


def _follow(data_dir: Path, launcher: FakeLauncher) -> GradientFollow:
    jobs = _jobs(data_dir, launcher)
    manager = JobManager(data_dir, (sys.executable, "-c", "pass"), "a" * 40, config_path("capabilities.toml"))
    return GradientFollow(jobs, manager, lambda document: [])


def test_adopt_interrupted_before_its_record_reuses_the_started_run(tmp_path: Path,
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    """抓「開了一般計算、``adopted.json`` 還沒落盤就中斷，重按又開第二筆」：一般計算是真的 ``JobManager``
    （執行程式什麼都不做），第一次寫採用紀錄時丟錯，重按要接回同一筆、計算狀態檔只有一份。"""
    data_dir = _data(tmp_path)
    launcher = FakeLauncher()
    job_id = _done_job(data_dir, launcher, _exported())
    follow = _follow(data_dir, launcher)
    original = follow_module._write_atomic
    crashed: list[Path] = []

    def crash_once(path: Path, text: str) -> None:
        if path.name == ADOPTED_FILE and not crashed:
            crashed.append(path)
            raise OSError("模擬採用紀錄落盤前中斷")
        original(path, text)

    monkeypatch.setattr(follow_module, "_write_atomic", crash_once)
    with pytest.raises(OSError, match="落盤前中斷"):
        follow.adopt(job_id)
    started = [path.stem for path in (data_dir / "runs").glob("*.json")]
    assert len(started) == 1 and not (data_dir / "gradient" / job_id / ADOPTED_FILE).exists()
    detail = follow.adopt(job_id)
    assert [path.stem for path in (data_dir / "runs").glob("*.json")] == started
    assert cast(Record, detail["adopted"])["run_id"] == started[0]


def test_job_ids_are_checked_before_any_path_is_built(tmp_path: Path) -> None:
    """抓代號沒先驗格式就拿去組路徑：「..」會讀到工作資料夾外面那一份 adopted.json。網址裡的「..」可能在送出前就被
    併掉，所以直接叫函式考。"""
    data_dir = _data(tmp_path)
    follow = _follow(data_dir, FakeLauncher())
    planted = {"job_id": "0" * 32, "scheme_id": "planted", "run_id": "1" * 32, "created_at": 0.0}
    for folder in (data_dir, data_dir / "gradient"):
        (folder / ADOPTED_FILE).write_text(json.dumps(planted), encoding="utf-8")
    for job_id in ("..", ".", "../gradient", "0" * 31):
        # 訊息是這一支格式檢查的那一句（後端深處也會丟 FileNotFoundError，但訊息只有代號）：證明是先擋下、沒去讀檔。
        with pytest.raises(FileNotFoundError, match="沒有這一跑"):
            follow.adopt(job_id)
        with pytest.raises(FileNotFoundError):
            follow.detail(job_id)


@pytest.mark.parametrize(("mount", "original", "warned"), [
    ("rigid", None, True), ("rigid", "room-original", False), ("on_wall", None, False)])
def test_rigid_mount_without_an_original_scheme_is_warned(tmp_path: Path, mount: str, original: str | None,
                                                          warned: bool) -> None:
    """抓選了「貼在硬底上」又沒指原方案時沒警告（結果頁會把搜尋的區當成沒動、改善算成零），或沒這個問題時亂警告。"""
    data_dir = _data(tmp_path)
    if original is not None:
        design = read_design(data_dir, NAME)
        assert design is not None
        write_design(data_dir, design.model_copy(update={"original_scheme_id": original}), _scheme())
    launcher = FakeLauncher()
    view = _jobs(data_dir, launcher).start(NAME, GradientRequest.model_validate({"mount": mount}))
    detail = _follow(data_dir, launcher).detail(str(view["job_id"]))
    assert (detail["baseline_warning"] == RIGID_BASELINE_WARNING) is warned
    assert (detail["baseline_warning"] is None) is not warned
