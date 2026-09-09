#!/usr/bin/env python3
"""雲端那一跑不准無聲死掉。

掃描面是掃描根自己那一層的 ``.github/workflows/*.yml``／``*.yaml``（不含樣本樹裡的道具），
加上 ``run:`` 呼叫、進得了版控的腳本。門檻與名單全部只寫在卡的 ``[settings]`` 裡，
讀不到就回 2（工具自壞），不回 0。四條：

1. **紅了不准不擋**——任何 step 或 job 寫 ``continue-on-error: true`` 就紅。這是 GitHub 上
   把紅漂成綠最直接的一個鍵：那一步失敗了，job 照樣算成功，required check 照樣綠。
   ``${{ ... }}`` 那種算出來才知道的值也算紅——閘的死活不准取決於一個算出來的旗標。
2. **離開碼不准被吞掉**——``run:`` 的內容（以及它呼叫、進得了版控的腳本）出現卡上登記的
   ``swallow_snippets``（把離開碼或錯誤導掉的 shell 字樣）就紅；單獨一行 ``exit 0`` 也紅
   （效果跟接在分號後面一樣，只咬帶分號那種會漏）；沒有 pipefail 的多段管線也紅——GitHub
   在 Linux 上預設的 ``run:`` 是 ``bash -e {0}``，**沒有** pipefail，``cmd | tail`` 只回最後
   一段的離開碼，前面那段紅了看不見。作者自己寫了 pipefail、或把 shell 指成卡上登記的
   ``pipefail_shells``（``shell: bash`` 在 GitHub 上是 ``bash -eo pipefail {0}``）就放行。
   遞迴進腳本是找碴席點名的：只掃 yaml 字面的話，把 ``|| true`` 搬進 ``scripts/x.sh``
   就繞過去了。
3. **job 不准漂綠、不准沒有上限**——job 的 ``if:`` 出現卡上登記的 ``forbidden_job_ifs``
   （``always()``）就紅：前置失敗了它照跑，結論照樣算綠。``timeout-minutes`` 缺席一樣紅，
   沒有上限等於可以無聲卡死（GitHub 的預設 360 分鐘不是宣告，是沒人想過這件事）；
   有寫但超過卡上登記的 ``max_timeout_minutes``、或不是整數字面值，也紅。
4. **不准有空 job**——每個 job 至少要有一步真的在跑檢查或測試（``run:`` 的內容、或它呼叫
   的腳本，命中卡上登記的 ``work_markers``）。只有 checkout／setup 的 job 永遠綠，
   掛成 required check 就是一個只會回綠的閘。

**為什麼用 pyyaml 而不是自己剖析。** 這四條要分得清 job 層與 step 層的同名鍵
（``continue-on-error`` 兩層都能寫，意思不同）、要把 ``timeout-minutes`` 讀成數字比大小、
要看得懂流式寫法與引號、還要拿到 ``run: |`` 區塊真正的內容（區塊摺疊符號由剖析器吃掉，
管線那一條才不會把 YAML 的 ``|`` 當成 shell 的管線）。縮排／正則的子集剖析每一種都繞得
過去，等於一個只會回綠的檢查。pyyaml 不在的時候這支檢查回 2，不會自己退回猜。

**「卡宣告的 job 必須存在且被呼叫」不在這裡**——那一關歸 ``rule-card-required-fields``
（見它的第一關三條交叉驗證）。兩張卡掃同一個對象會互相遮蔽，所以這裡不重複。

**血債那一半**：``guard-of-guards-silently-dies`` 只記關聯不算血債，理由寫在卡的
``related_lessons_why``——那筆事故的五個載體全是本機 hook，這張卡在事故當下咬不到；
留下來的是同一個失敗形狀換了載體（守衛無聲死掉、離開碼被吞掉），今天的載體是 workflow。

**已知的縫**（照抄不遮，見卡面「刻意沒管的事」）：cron 心跳那一段沒做（要上網）；
順序條款沒有機器判準；step 層的 ``if:`` 不管；``jobs.<id>.uses:`` 那種 job 會被第 4 條
誤判成空 job（今天零對象）；管線那一條把管線塞進 ``bash -c "..."`` 的字串就繞得過去。
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from governance.exit_codes import ToolBroken, run  # noqa: E402
from governance.loader import RULES_DIR  # noqa: E402

try:
    import yaml
except ImportError as exc:  # pyyaml 不在就回 2，不准退回猜
    yaml = None
    YAML_IMPORT_ERROR = str(exc)
else:
    YAML_IMPORT_ERROR = ""

# 這支檢查在卡裡的名字。門檻只從「宣告了這支檢查」的那張卡讀。
CHECK_REL = "governance/checks/ci_jobs_cannot_die_quietly.py"

WORKFLOW_DIR = ".github/workflows"
WORKFLOW_SUFFIXES = (".yml", ".yaml")

# 「這一步／這個 job 失敗了也不擋」的鍵。兩層都能寫，兩層都咬。
CONTINUE_KEY = "continue-on-error"
TIMEOUT_KEY = "timeout-minutes"

# ``run:`` 裡叫得動的腳本長什麼樣（只認進得了版控、掃描根底下的檔）。跟
# green-must-be-real-green 用同一個形狀。
SCRIPT_RE = re.compile(r"[\w./-]+\.(?:sh|bash|py)")

# 剝掉引號裡的內容用的：管線那一條不准把 ``grep 'a|b'`` 當成管線。
QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# 邏輯或不是管線。比對前先把它換成這個佔位字元。
OR_PLACEHOLDER = "\x00"

# 門檻的形狀。打錯字的門檻等於沒有門檻，所以多一個鍵、少一個鍵、型別不對，一律回 2。
LIST_KEYS = (
    "swallow_snippets",
    "forbidden_job_ifs",
    "work_markers",
    "pipefail_markers",
    "pipefail_shells",
)
INT_KEYS = ("max_timeout_minutes",)
SETTINGS_KEYS = (*LIST_KEYS, *INT_KEYS)


def _card_settings(scan_root: Path, files: list[Path]) -> dict[str, object]:
    """從掃描根自己的 ``governance/rules/`` 讀這支檢查的門檻。

    找不到、找到多張、或門檻的形狀不對，一律 raise ToolBroken——沒有門檻就不出結論。
    """
    rules_dir = scan_root / RULES_DIR
    mine: list[tuple[Path, dict[str, object]]] = []
    for path in sorted(f for f in files if f.parent == rules_dir and f.suffix == ".toml"):
        try:
            data = tomllib.loads(path.read_bytes().decode("utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ToolBroken(f"讀不開 {path.relative_to(scan_root)}：{exc}") from exc
        if data.get("check") == CHECK_REL:
            mine.append((path, data))
    if len(mine) != 1:
        raise ToolBroken(
            f"{scan_root}/{RULES_DIR} 底下宣告 check={CHECK_REL} 的卡有 {len(mine)} 張，要剛好 1 張"
            "——門檻只寫在卡上，讀不到門檻這一跑就不算數"
        )
    path, data = mine[0]
    rel = str(path.relative_to(scan_root))
    settings = data.get("settings")
    if not isinstance(settings, dict):
        raise ToolBroken(f"{rel} 沒有 [settings] 表（上限與名單都寫在那裡）")
    _settings_problems(settings, rel)
    return settings


def _settings_problems(settings: dict[str, object], rel: str) -> None:
    bad: list[str] = []
    extra = [k for k in settings if k not in SETTINGS_KEYS]
    if extra:
        bad.append(f"多了不認識的鍵 {sorted(extra)}，只認 {list(SETTINGS_KEYS)}（打錯字的門檻等於沒門檻）")
    for key in LIST_KEYS:
        value = settings.get(key)
        if not isinstance(value, list) or not all(isinstance(x, str) and x.strip() for x in value):
            bad.append(f"{key} 必須是字串 list，實際是 {value!r}")
        elif not value:
            bad.append(f"{key} 不准是空 list——空的名單等於這一條沒在管")
    cap = settings.get("max_timeout_minutes")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        bad.append(
            f"max_timeout_minutes 必須是 1 以上的整數（今天實跑時間的合理倍數），實際是 {cap!r}"
            "——上限寫 0 或不寫等於沒有上限"
        )
    if bad:
        raise ToolBroken(f"{rel} 的 [settings] 形狀不對：" + "；".join(bad))


def _read(path: Path, rel: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ToolBroken(f"讀不到 {rel}：{exc}") from exc


def _workflow(path: Path, rel: str) -> dict[str, object]:
    """剖析一份 workflow。看不懂的一律回 2，不准假裝乾淨也不准當成違規。"""
    text = _read(path, rel)
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ToolBroken(f"{rel} 不是解得開的 YAML（{exc}）——我沒看懂就不出結論") from exc
    if not isinstance(data, dict):
        raise ToolBroken(f"{rel} 剖析出來不是一張表（實際是 {type(data).__name__}），我看不懂")
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise ToolBroken(f"{rel} 的 jobs: 解不出任何 job（實際是 {jobs!r}），這份 workflow 我看不懂")
    return data


def _default_shell(holder: object) -> str:
    """``defaults.run.shell``。沒寫就是空字串（GitHub 在 Linux 上的預設是 bash -e，沒有 pipefail）。"""
    if not isinstance(holder, dict):
        return ""
    defaults = holder.get("defaults")
    if not isinstance(defaults, dict):
        return ""
    block = defaults.get("run")
    if not isinstance(block, dict):
        return ""
    shell = block.get("shell")
    return shell if isinstance(shell, str) else ""


def _continue_problems(where: str, holder: dict[str, object]) -> list[str]:
    """第 1 條：``continue-on-error`` 只准缺席或明寫 false。"""
    if CONTINUE_KEY not in holder:
        return []
    value = holder[CONTINUE_KEY]
    if value is False:
        return []
    return [
        f"{where} 寫了 {CONTINUE_KEY}: {value!r}"
        "——那一步／那個 job 紅了也不擋，job 照樣算成功、required check 照樣綠。"
        "雲端那一跑是唯一權威（docs/decisions/cloud-run-is-the-only-authority.md），"
        "權威自己把紅漂成綠，等於沒有閘"
    ]


def _pipelines(body: str) -> list[str]:
    """多段管線的那幾行。先剝掉引號裡的內容、把邏輯或換掉，剩下的單一個 ``|`` 才算管線。"""
    hits: list[str] = []
    for raw in body.splitlines():
        line = QUOTED_RE.sub("", raw)
        line = line.replace("|" + "|", OR_PLACEHOLDER)
        line = line.split("#", 1)[0]
        if "|" not in line:
            continue
        left, _, right = line.partition("|")
        if left.strip() and right.strip():
            hits.append(raw.strip())
    return hits


def _swallow_problems(
    where: str,
    body: str,
    settings: dict[str, object],
    shell: str,
) -> list[str]:
    """第 2 條：離開碼不准被吞掉（``run:`` 的內容，以及它呼叫的腳本，用同一把尺）。"""
    bad: list[str] = []
    for snippet in settings["swallow_snippets"]:  # type: ignore[union-attr]
        if snippet in body:
            bad.append(
                f"{where} 把離開碼吞掉了（命中卡上登記的 {snippet!r}）"
                "——檢查回 1 也會變成 shell 回 0，紅的被漂成綠的"
            )
    for line in body.splitlines():
        if line.strip() == "exit " + "0":
            bad.append(
                f"{where} 有單獨一行 `exit 0`"
                "——前面那幾行的離開碼被它整個蓋掉，跟接在分號後面是同一回事"
            )
            break
    markers = [m for m in settings["pipefail_markers"] if m in body]  # type: ignore[union-attr]
    shells = settings["pipefail_shells"]  # type: ignore[assignment]
    # 刻意比整串、不比第一個詞：GitHub 只有在 shell 寫成關鍵字（就是 `bash` 這個字）時才幫你
    # 加 -eo pipefail；寫成自訂命令（`bash -e {0}`）它照字面跑，沒有 pipefail。比第一個詞會把
    # 後者誤判成合規。
    if not markers and shell.strip() not in shells:
        pipes = _pipelines(body)
        if pipes:
            bad.append(
                f"{where} 用了沒有 pipefail 的多段管線（{pipes[0]!r}）"
                "——GitHub 在 Linux 上預設的 run: 是 `bash -e {0}`，沒有 pipefail，"
                "管線只回最後一段的離開碼，前面那段紅了看不見。"
                f"要嘛自己開 pipefail，要嘛把 shell 指成卡上登記的 {list(shells)}"
            )
    return bad


def _called_scripts(body: str, scan_root: Path, files: list[Path]) -> list[tuple[str, str]]:
    """``run:`` 裡叫到、進得了版控的腳本：[(相對路徑, 內容)]。"""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in SCRIPT_RE.findall(body):
        rel = token.lstrip("./")
        if rel in seen:
            continue
        path = scan_root / rel
        if path not in files or not path.is_file():
            continue
        seen.add(rel)
        out.append((rel, _read(path, rel)))
    return out


def _timeout_problems(where: str, job: dict[str, object], cap: int) -> list[str]:
    """第 3 條後半：``timeout-minutes`` 必須在場，而且是不超過上限的整數字面值。"""
    if TIMEOUT_KEY not in job:
        return [
            f"{where} 沒有 {TIMEOUT_KEY}"
            "——沒有上限等於可以無聲卡死。GitHub 的預設 360 分鐘不是宣告，是沒人想過這件事；"
            f"上限只寫在卡的 [settings]（今天是 {cap} 分鐘）"
        ]
    value = job[TIMEOUT_KEY]
    if isinstance(value, bool) or not isinstance(value, int):
        return [
            f"{where} 的 {TIMEOUT_KEY} 是 {value!r}，必須是整數字面值"
            "——算出來才知道的上限等於沒有上限（算錯就回到 GitHub 的預設）"
        ]
    if value < 1:
        return [f"{where} 的 {TIMEOUT_KEY} 是 {value}，必須是 1 以上的分鐘數"]
    if value > cap:
        return [
            f"{where} 的 {TIMEOUT_KEY} 是 {value}，超過卡上登記的上限 {cap} 分鐘"
            "——開得比實跑時間大太多等於沒有上限，要改寬就改卡、走 PR"
        ]
    return []


def _if_problems(where: str, job: dict[str, object], markers: list[str]) -> list[str]:
    """第 3 條前半：job 的 ``if:`` 不准把紅漂成綠。"""
    raw = job.get("if")
    if not isinstance(raw, str):
        return []
    bad: list[str] = []
    for marker in markers:
        if marker in raw:
            bad.append(
                f"{where} 的 if: 寫了 {raw!r}（命中卡上登記的 {marker!r}）"
                "——前置失敗了它照跑，結論還是算綠，紅被漂成綠"
            )
    return bad


def _job_problems(
    rel: str,
    name: str,
    job: object,
    wf_shell: str,
    settings: dict[str, object],
    scan_root: Path,
    files: list[Path],
) -> list[str]:
    where = f"{rel} 的 job {name!r}"
    if not isinstance(job, dict):
        raise ToolBroken(f"{where} 剖析出來不是一張表（實際是 {type(job).__name__}），我看不懂")

    bad: list[str] = []
    bad += _continue_problems(where, job)
    bad += _if_problems(where, job, list(settings["forbidden_job_ifs"]))  # type: ignore[arg-type]
    bad += _timeout_problems(where, job, int(settings["max_timeout_minutes"]))  # type: ignore[call-overload]

    job_shell = _default_shell(job) or wf_shell
    steps = job.get("steps")
    if not isinstance(steps, list) or not steps:
        return bad + [
            f"{where} 底下沒有任何步驟（steps={steps!r}）"
            "——空 job 永遠綠，掛成 required check 就是一個只會回綠的閘"
        ]

    does_work = False
    markers = list(settings["work_markers"])  # type: ignore[arg-type]
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            raise ToolBroken(f"{where} 第 {index + 1} 步剖析出來不是一張表（{step!r}），我看不懂")
        label = step.get("name") if isinstance(step.get("name"), str) else f"第 {index + 1} 步"
        step_where = f"{where} 的{label}"
        bad += _continue_problems(step_where, step)

        body = step.get("run")
        if not isinstance(body, str):
            continue
        shell = step.get("shell") if isinstance(step.get("shell"), str) else job_shell
        bad += _swallow_problems(f"{step_where} 的 run:", body, settings, shell or "")
        if any(marker in body for marker in markers):
            does_work = True
        for script_rel, text in _called_scripts(body, scan_root, files):
            bad += _swallow_problems(f"{step_where} 呼叫的腳本 {script_rel}", text, settings, "")
            if any(marker in text for marker in markers):
                does_work = True

    if not does_work:
        bad.append(
            f"{where} 沒有任何一步在跑檢查或測試（run: 的內容與它呼叫的腳本都沒命中"
            f"卡上登記的 {markers}）——只有 checkout／setup 的 job 永遠綠，等於沒有檢查"
        )
    return bad


def check(scan_root: Path, files: list[Path]) -> list[str]:
    if yaml is None:
        raise ToolBroken(
            f"pyyaml 不在這個環境裡（{YAML_IMPORT_ERROR}）——這張卡要對 workflow 做真的 YAML 剖析，"
            "沒有剖析器就不出結論。它登記在 pyproject.toml 的正式依賴裡，跑 `uv sync --locked`"
        )
    settings = _card_settings(scan_root, files)

    workflow_dir = scan_root / WORKFLOW_DIR
    workflows = sorted(
        f for f in files if f.parent == workflow_dir and f.suffix.casefold() in WORKFLOW_SUFFIXES
    )
    if not workflows:
        raise ToolBroken(
            f"{scan_root}/{WORKFLOW_DIR} 底下一份版控裡的 workflow 都沒有"
            "——沒掃到 workflow 不等於沒有違規，這一跑不算數"
        )

    bad: list[str] = []
    for path in workflows:
        rel = str(path.relative_to(scan_root))
        data = _workflow(path, rel)
        wf_shell = _default_shell(data)
        jobs = data["jobs"]
        for name, job in jobs.items():  # type: ignore[union-attr]
            bad += _job_problems(rel, str(name), job, wf_shell, settings, scan_root, files)
    return bad


if __name__ == "__main__":
    sys.exit(run(check, description="雲端工作不准無聲死掉：吞離開碼、漂綠、沒有上限、空 job 一律紅"))
