"""鏡像 writer（``mirror_receipts.mirror``）與 reader（``cloud_receipts.read_mirror``）同時跑時，
reader 不准看到半份鏡像（票 #235）。writer 先抄收據、最後才寫 provenance.json；兩步之間 reader
沒有協調就讀會拿到「鏡像缺來源檔」的 ToolBroken。這幾題在舊碼上必須因真缺陷失敗。

覆蓋面（設計紙第 13 行點名的幾件事）：
1. 來源釘在「第一次解析到的 commit」上——resolve 之後把 ref 搬走、再加一份收據，鏡像內容、
   檔名集合、每份收據的來源 commit 都還要是第一次那一筆。
2. writer 停接縫、reader 真的撞鎖那一拍必須真的發生，且 writer 沒先崩。
3. 鎖跨程序也有效：parent 握真的 ``mirror_lock``、子程序用 raw ``fcntl.flock(LOCK_NB)``。
4. writer／reader 出錯路徑會清殘留、放鎖。
5. ``main()`` 的份數與 ref 來自鎖住時讀到的那份來源檔，讀不到／讀壞回 2。
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from governance import cloud_receipts
from governance.exit_codes import CLEAN, TOOL_BROKEN, ToolBroken
from governance.mirror_lock import mirror_lock
from governance.status import mirror_receipts
from tests.conftest import GitSandbox

OUT_NAME = "cloud"
READER = "concurrency-test"
WAIT_SECONDS = 60.0
SCHEMA = 1
RECEIPT_NAMES = ("1-1.json", "2-1.json")

# read_mirror 的登記簿：目錄名與新舊版本，形狀同卡上 [settings]。
SETTINGS: dict[str, object] = {
    "mirror_dir": OUT_NAME,
    "receipts_subdir": mirror_receipts.BRANCH_DIR,
    "provenance_file": mirror_receipts.PROVENANCE_FILE,
    "newest_schema": SCHEMA,
}


def receipt_body(name: str) -> str:
    """最小但形狀正確的收據。"""
    return json.dumps({"schema": SCHEMA, "name": name}, ensure_ascii=False) + "\n"


def make_status_branch(sandbox: GitSandbox, names: tuple[str, ...] = RECEIPT_NAMES) -> str:
    """建 status 放這幾份收據、每份一筆提交，回最後一筆提交，最後切回 main。"""
    root = sandbox.root
    (root / "README").write_text("main\n", encoding="utf-8")
    sandbox.git("add", "README")
    sandbox.git("commit", "-q", "-m", "main")
    sandbox.git("switch", "-q", "-c", "status")
    (root / mirror_receipts.BRANCH_DIR).mkdir()
    for name in names:
        (root / mirror_receipts.BRANCH_DIR / name).write_text(receipt_body(name), encoding="utf-8")
        sandbox.git("add", str(Path(mirror_receipts.BRANCH_DIR) / name))
        sandbox.git("commit", "-q", "-m", f"receipt: {name}")
    commit = sandbox.git("rev-parse", "HEAD").stdout.strip()
    sandbox.git("switch", "-q", "main")
    return commit


def wait_for(event: threading.Event, what: str) -> None:
    """有界等待，時間到紅（掛死會被記成 error，不是這一題要的 failure）。"""
    assert event.wait(WAIT_SECONDS), f"等了 {WAIT_SECONDS} 秒還沒看到「{what}」——這一跑不算數"


@contextmanager
def running(target: Callable[[], object], name: str) -> Iterator[threading.Thread]:
    """跑一條執行緒，不管中間怎麼炸最後一定 join。"""
    thread = threading.Thread(target=target, name=name, daemon=True)
    thread.start()
    try:
        yield thread
    finally:
        thread.join(WAIT_SECONDS)


def assert_lock_free(parent: Path) -> None:
    """事後拿「互斥、不等待」的鎖要成功——證明失敗路徑也放了鎖。"""
    fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise AssertionError(f"失敗之後互斥鎖沒有被放掉（{parent}）——鎖漏了") from exc
    finally:
        os.close(fd)


def pause_entry(hold: threading.Event, entered: threading.Event) -> Callable[..., mirror_receipts.Origin]:
    """``origin_of`` 的替身：第一次被叫（第一份收據已寫、來源檔未寫的接縫）停住，其餘照常。"""
    real = mirror_receipts.origin_of
    seen = {"n": 0}

    def paused(root: Path, ref: str, path: str, timeout: int) -> mirror_receipts.Origin:
        if seen["n"] == 0:
            seen["n"] = 1
            entered.set()
            assert hold.wait(WAIT_SECONDS), "writer 等不到放行——這一跑不算數"
        return real(root, ref, path, timeout)

    return paused


def run_writer(root: Path, out: Path, entered: threading.Event, result: list[object]) -> None:
    """writer：真的 ``mirror()``。成功回 provenance 路徑、失敗把例外原樣帶回，最後一定報到。"""
    try:
        result.append(mirror_receipts.mirror(root, "status", out, timeout=30))
    except BaseException as exc:  # noqa: BLE001  # expires=2026-12-08 reason=writer 的失敗要原樣帶回，不准吞
        result.append(exc)
    finally:
        entered.set()


def read_and_record(root: Path, result: list[object], observed: threading.Event) -> None:
    """reader：真的一條執行緒，真的 ``read_mirror``。"""
    try:
        result.append(cloud_receipts.read_mirror(root, SETTINGS, READER))
    except BaseException as exc:  # noqa: BLE001  # expires=2026-12-08 reason=reader 讀到什麼正是這一題要看的
        result.append(exc)
    finally:
        observed.set()


def test_reader_never_sees_a_mirror_without_its_provenance(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """writer 停在接縫、reader 真的讀：舊碼因缺來源檔紅，修好後讀到完整鏡像。"""
    make_status_branch(git_sandbox)
    root = git_sandbox.root
    out = root / OUT_NAME
    target_dir = out / mirror_receipts.BRANCH_DIR
    hold = threading.Event()
    entered = threading.Event()
    observed = threading.Event()
    writer_out: list[object] = []
    reader_out: list[object] = []

    real_flock = fcntl.flock

    def observed_flock(fd: int, operation: int) -> None:
        # 共用鎖被擋下來＝ writer 真的握著互斥鎖那一拍；通知後照約定 blocking 拿。
        if operation & fcntl.LOCK_SH:
            try:
                real_flock(fd, operation | fcntl.LOCK_NB)
            except BlockingIOError:
                observed.set()
                real_flock(fd, operation)
            return
        real_flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", observed_flock)
    monkeypatch.setattr(mirror_receipts, "origin_of", pause_entry(hold, entered))

    with running(lambda: run_writer(root, out, entered, writer_out), name="writer") as writer:
        try:
            wait_for(entered, "writer 走到第一份收據已寫、來源檔未寫的接縫")
            assert not writer_out, f"writer 還沒走到接縫就崩了：{writer_out!r}"
            assert (target_dir / RECEIPT_NAMES[0]).is_file(), "接縫上第一份收據應該已經在磁碟上"
            assert not (out / mirror_receipts.PROVENANCE_FILE).is_file(), "接縫上來源檔不該已經寫好"
            with running(lambda: read_and_record(root, reader_out, observed), name="reader") as reader:
                try:
                    wait_for(observed, "reader 撞上 writer 的互斥鎖（舊碼是否通知由下面斷言決定）")
                finally:
                    hold.set()
            assert not reader.is_alive(), "reader 沒有在期限內結束——這一跑不算數"
        finally:
            hold.set()  # 不管哪一步紅，先放掉 writer 再 join
    assert not writer.is_alive(), "writer 沒有在期限內結束——這一跑不算數"

    assert not (writer_out and isinstance(writer_out[0], BaseException)), (
        f"writer 自己炸了（{writer_out!r}）——這一題要紅的是 reader 看到半份鏡像"
    )
    assert writer_out and isinstance(writer_out[0], Path), f"writer 沒鏡好：{writer_out!r}"

    got = reader_out[0] if reader_out else None
    assert not isinstance(got, ToolBroken), (
        f"reader 在 writer 寫到一半時看到半份鏡像：{got}——writer 先抄收據、最後才寫 provenance.json，"
        "中間那一瞬 reader 就判這一跑不算數；修法是 writer 整段握互斥鎖、reader 整段握共用鎖"
    )
    assert isinstance(got, cloud_receipts.Mirror), f"reader 回的不是 Mirror：{got!r}"

    provenance = json.loads((out / mirror_receipts.PROVENANCE_FILE).read_text(encoding="utf-8"))
    assert got.provenance == provenance, "reader 讀到的來源檔內容跟 writer 最後留在磁碟上的不是同一份"
    assert {r.name for r in got.receipts} == set(RECEIPT_NAMES)
    assert {r.name for r in got.receipts} == set(provenance["files"])


def test_mirror_pins_everything_to_the_commit_it_resolved(
    git_sandbox: GitSandbox, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """resolve_ref 回話之後把 status 搬到別處、再加一份收據：鏡像內容、檔名、每份來源都還是第一次那一筆。"""
    first = make_status_branch(git_sandbox)
    root = git_sandbox.root
    target = Path(mirror_receipts.BRANCH_DIR) / RECEIPT_NAMES[0]
    # 搬分支之前先算好「1-1.json 在第一筆解析那刻的來源 commit」——它不是 mirror 的 HEAD，是
    # 最後一次動到那一份檔的那一筆（1-1.json 與 2-1.json 各是一筆提交）。
    expected_origin = git_sandbox.git("log", "-1", "--format=%H", first, "--", str(target)).stdout.strip()

    git_sandbox.git("switch", "-q", "status")
    (root / target).write_text(receipt_body("moved"), encoding="utf-8")
    third = Path(mirror_receipts.BRANCH_DIR) / "3-1.json"
    (root / third).write_text(receipt_body("3-1.json"), encoding="utf-8")
    git_sandbox.git("add", str(target), str(third))
    git_sandbox.git("commit", "-q", "-m", "receipt: moved and added")
    second = git_sandbox.git("rev-parse", "HEAD").stdout.strip()
    git_sandbox.git("switch", "-q", "main")
    # 切回 main 之後 status 還指在 second：先把它搬回第一次解析那一筆，否則 resolve_ref 解到的
    # 一開始就是 second，這一題要看的「resolve 之後 ref 才被搬走」那條路根本沒被走到。
    git_sandbox.git("branch", "-f", "status", first)

    real_resolve_ref = mirror_receipts.resolve_ref
    moved = {"done": False}

    def move_after_resolve(root: Path, ref: str, timeout: int) -> str:
        resolved = real_resolve_ref(root, ref, timeout)
        git_sandbox.git("branch", "-f", "status", second)
        moved["done"] = True
        return resolved

    monkeypatch.setattr(mirror_receipts, "resolve_ref", move_after_resolve)
    out = tmp_path / OUT_NAME
    provenance = json.loads(mirror_receipts.mirror(root, "status", out, timeout=30).read_text(encoding="utf-8"))

    assert second != first, "兩筆提交撞在一起——這一跑不算數"
    assert moved["done"], "修法根本沒叫到被攔下的 resolve_ref——搬分支沒發生，這一題就沒測到"
    assert git_sandbox.git("rev-parse", "status").stdout.strip() == second, (
        "status 分支沒有搬到 second——resolve 之後會動的 ref 那條路沒被走到"
    )

    assert provenance["commit"] == first, (
        "來源檔記的 commit 不是最初解到的那一筆——writer 邊抄邊看會動的 ref，"
        "抄到的內容與記下來的來源會是不同兩筆提交"
    )
    files = provenance["files"]
    assert files[RECEIPT_NAMES[0]]["commit"] == expected_origin, (
        "第一份收據的來源 commit 不等於搬分支前算出來那一筆——origin_of 還在讀會動的 ref，"
        "或者把每份收據的來源都記成 mirror 的 HEAD"
    )
    assert files[RECEIPT_NAMES[0]]["commit"] != first, "1-1.json 的來源被記成 mirror 的 HEAD 了"
    assert set(files) == set(RECEIPT_NAMES), "來源檔的 files 集合不是第一次解析那一筆的檔名集合"
    mirror_names = {p.name for p in (out / mirror_receipts.BRANCH_DIR).glob("*.json")}
    assert mirror_names == set(RECEIPT_NAMES), "鏡像的收據檔名集合不是第一次解析那一筆——會動的 ref 那條路還被走到"
    body = json.loads((out / target).read_text(encoding="utf-8"))
    assert body["name"] == RECEIPT_NAMES[0], "鏡像裡的收據內容是搬動之後那一筆提交的"
    assert body != json.loads(receipt_body("moved")), "這一題的兩版收據內容必須真的不一樣"


# 子程序只用 raw fcntl 對同一個父目錄上鎖，回「拿到」或「被擋」一個字，不 import 治理層。
CHILD_FLOCK = (
    "import fcntl, os, sys\n"
    "fd = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY)\n"
    "op = fcntl.LOCK_SH if sys.argv[2] == 'sh' else fcntl.LOCK_EX\n"
    "try:\n"
    "    fcntl.flock(fd, op | fcntl.LOCK_NB)\n"
    "    print('acquired')\n"
    "except BlockingIOError:\n"
    "    print('blocked')\n"
)


@pytest.mark.parametrize(
    ("parent_mode", "child_mode", "expected"),
    [
        ("sh", "sh", "acquired"),
        ("ex", "sh", "blocked"),
        ("sh", "ex", "blocked"),
        ("ex", "ex", "blocked"),
    ],
)
def test_mirror_lock_holds_across_processes(
    tmp_path: Path, parent_mode: str, child_mode: str, expected: str
) -> None:
    """parent 握真的 mirror_lock，另一個程序 raw flock 同一個父目錄：互斥擋、同用共享放。"""
    out = tmp_path / OUT_NAME
    with mirror_lock(out, exclusive=parent_mode == "ex"):
        proc = subprocess.run(
            [sys.executable, "-c", CHILD_FLOCK, str(out.parent), child_mode],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=WAIT_SECONDS,
        )
    assert proc.returncode == 0, (
        f"子程序自己死了（離開碼 {proc.returncode}）——先看它的 stderr，別讓 stdout 的空殼把錯誤蓋掉："
        f"{proc.stderr.strip()[:300]!r}"
    )
    assert proc.stdout.strip() == expected, (
        f"跨程序拿鎖的結果是 {proc.stdout.strip()!r}，期望 {expected!r}——"
        "production 那一把鎖不是真的 fcntl flock，或 shared/exclusive 語意錯了"
    )


def _origin_broken(root: Path, ref: str, path: str, timeout: int) -> mirror_receipts.Origin:
    """注入的故障：查來源那一拍直接 ToolBroken。"""
    raise ToolBroken("查來源那一拍自壞（注入的故障）")


def _provenance_write_broken(real: Callable[..., int]) -> Callable[..., int]:
    """包一層 write_text：只有來源檔那一拍丟 OSError，其餘照常。"""

    def write(self: Path, data: str, encoding: str | None = None) -> int:
        if self.name == mirror_receipts.PROVENANCE_FILE:
            raise OSError(28, "寫來源檔那一拍自壞（注入的故障）")
        return real(self, data, encoding=encoding)

    return write


@pytest.mark.parametrize("fail_mode", ["origin", "provenance-write"])
def test_writer_failure_cleans_and_releases_lock(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fail_mode: str
) -> None:
    """writer 中途失敗：回 ToolBroken、把 out 整個清掉、且事後互斥鎖拿得到。"""
    make_status_branch(git_sandbox)
    out = tmp_path / OUT_NAME
    if fail_mode == "origin":
        monkeypatch.setattr(mirror_receipts, "origin_of", _origin_broken)
    else:
        monkeypatch.setattr(Path, "write_text", _provenance_write_broken(Path.write_text))

    with pytest.raises(ToolBroken):
        mirror_receipts.mirror(git_sandbox.root, "status", out, timeout=30)
    assert not out.exists(), "鏡子失敗後不該留下半份鏡像"
    assert_lock_free(out.parent)


def test_interrupted_old_mirror_deletion_is_not_accepted(
    git_sandbox: GitSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    """刪舊鏡像刪到一半被中斷：半份舊收據不准還被當成完整鏡像。

    production 在下 rmtree **之前**先 unlink provenance；拿掉那一步，被中斷後殘下的收據就會
    帶著還在的來源檔被 reader 收下——這一題要紅在那裡。中斷精準打在 rmtree 那一拍（不是
    kill 子程序），模擬沒有 OSError 收尾的突然展開。
    """
    make_status_branch(git_sandbox)
    root = git_sandbox.root
    out = root / OUT_NAME
    # 先用真的 mirror 鏡一份好的：等一下才有「舊鏡像」可刪。
    mirror_receipts.mirror(root, "status", out, timeout=30)

    victim = out / mirror_receipts.BRANCH_DIR / RECEIPT_NAMES[0]

    def interrupted_rmtree(path: object, *args: object, **kwargs: object) -> None:
        # 刪掉一份已知收據，再中斷：磁碟上留下「半份舊收據」，且不像 OSError 那樣被 production 的
        # except 分支收尾清掉——測試注入的中斷就是這個意思。
        victim.unlink()
        raise KeyboardInterrupt

    # monkeypatch 只包住第二次 mirror，避免干擾 fixture 收尾。
    with monkeypatch.context() as patched:
        patched.setattr(shutil, "rmtree", interrupted_rmtree)
        with pytest.raises(KeyboardInterrupt):
            mirror_receipts.mirror(root, "status", out, timeout=30)

    with pytest.raises(ToolBroken, match="缺來源檔"):
        cloud_receipts.read_mirror(root, SETTINGS, READER)
    assert_lock_free(out.parent)


def test_reader_malformed_provenance_releases_lock(tmp_path: Path) -> None:
    """reader 在共用鎖裡遇到壞 JSON：回 ToolBroken，鎖也放掉。"""
    folder = tmp_path / OUT_NAME
    (folder / mirror_receipts.BRANCH_DIR).mkdir(parents=True)
    (folder / mirror_receipts.BRANCH_DIR / RECEIPT_NAMES[0]).write_text(receipt_body(RECEIPT_NAMES[0]), encoding="utf-8")
    (folder / mirror_receipts.PROVENANCE_FILE).write_text("{ 不是 json", encoding="utf-8")

    with pytest.raises(ToolBroken):
        cloud_receipts.read_mirror(tmp_path, SETTINGS, READER)
    assert_lock_free(folder.parent)


def test_reader_missing_parent_creates_nothing(tmp_path: Path) -> None:
    """鏡像目錄的上一層不存在：reader 回 ToolBroken，且不准建出任何東西。"""
    nested = dict(SETTINGS, mirror_dir="deep/cloud")
    with pytest.raises(ToolBroken):
        cloud_receipts.read_mirror(tmp_path, nested, READER)
    assert not (tmp_path / "deep").exists(), "reader 不該在上一層缺席時自己建目錄"


def _mirror_returning(provenance: Path) -> Callable[[Path, str, Path, int], Path]:
    """monkeypatch 用的替身 mirror：不看參數、不寫東西，回準備好的來源檔路徑。"""

    def fake(root: Path, ref: str, out: Path, timeout: int) -> Path:
        return provenance

    return fake


def _root_at(root: Path) -> Callable[[], Path]:
    """monkeypatch 用的替身 repo_root：固定回這個目錄。"""

    def fake() -> Path:
        return root

    return fake


def test_main_reports_two_when_provenance_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """來源檔不是 JSON：main 回 2 並把話記到 stderr。"""
    monkeypatch.setattr(mirror_receipts, "repo_root", _root_at(tmp_path))
    provenance = tmp_path / mirror_receipts.PROVENANCE_FILE
    provenance.write_text("{ 不是 json", encoding="utf-8")
    monkeypatch.setattr(mirror_receipts, "mirror", _mirror_returning(provenance))

    assert mirror_receipts.main(["--out", OUT_NAME]) == TOOL_BROKEN
    assert "鏡不成" in capsys.readouterr().err


def test_main_reports_two_when_provenance_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """來源檔不存在：main 回 2。"""
    monkeypatch.setattr(mirror_receipts, "repo_root", _root_at(tmp_path))
    provenance = tmp_path / mirror_receipts.PROVENANCE_FILE
    monkeypatch.setattr(mirror_receipts, "mirror", _mirror_returning(provenance))

    assert mirror_receipts.main(["--out", OUT_NAME]) == TOOL_BROKEN
    assert "鏡不成" in capsys.readouterr().err


def test_main_reports_the_ref_and_commit_from_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """回報的 ref@commit 是來源檔裡那份，不是命令列的 ref。"""
    monkeypatch.setattr(mirror_receipts, "repo_root", _root_at(tmp_path))
    provenance = tmp_path / mirror_receipts.PROVENANCE_FILE
    provenance.write_text(
        json.dumps({"ref": "release/other", "commit": "deadbeef", "files": {"1.json": {"commit": "x"}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mirror_receipts, "mirror", _mirror_returning(provenance))

    assert mirror_receipts.main(["--out", OUT_NAME]) == CLEAN
    note_out = capsys.readouterr().err
    assert "release/other@deadbeef" in note_out, f"沒回報來源檔裡的 ref@commit：{note_out!r}"
