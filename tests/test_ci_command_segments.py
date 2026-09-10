"""ci-jobs-cannot-die-quietly 第⑤條的切段那一半：一行裡串接的每一段各自算一步。

不 spawn 整支檢查（那一半由後設測試的六回合跑，掃描根是必紅樣本樹），這裡直接餵切段函式
（`_split_outside_quotes`，把一行 shell 依分隔符切開）與分類函式（`_receipt_step_problems`，
判每一段是抄寫員還是水管）。判準（抄寫員那一串命令、水管名單）從主卡的 `[settings]` 讀，
不在這裡重寫一份——門檻只有一個家。

為什麼要有這一支：2026-09-10 之前判準比的是**整行**的前幾個字，所以
`uv sync --locked && 跑一支檢查` 開頭是水管就整行放行，後面那支檢查的離開碼不會進收據。
必紅樣本 `case-plumbing-chained-with-check` 咬的是整支檢查的離開碼，這一支咬的是切段與分類
本身的形狀（第幾段、切在哪、引號裡不切），兩層各看一半。
"""
from __future__ import annotations

import tomllib
from pathlib import Path

from governance.checks import ci_jobs_cannot_die_quietly as ci
from governance.loader import setting_text

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "governance" / "rules" / "ci-jobs-cannot-die-quietly.toml"

# 樣本用的命令。抄寫員那一串從卡上讀（見 `scribe()`），這兩個是被它跑的對象與一支檢查。
CHECK = "uv run python -m governance.checks.stub_check --scan-root ."
FETCH = "git fetch origin status"
SEPARATORS = ("&&", ";", "||", "|")


def settings() -> dict[str, object]:
    data = tomllib.loads(CARD.read_text(encoding="utf-8"))
    table = data["settings"]
    assert isinstance(table, dict)
    return table


def scribe() -> str:
    """抄寫員（把那一步真實離開碼記成一片收據的那一層）那一串命令，從卡上讀。"""
    return setting_text(settings(), "wrapper_command")


def segments(line: str) -> list[str]:
    """production 的順序：先剝掉引號外的行內註解，再依分隔符切段。"""
    return ci._split_outside_quotes(ci._strip_comment(line), ci.SEGMENT_SEP_RE)


def problems(body: str) -> list[str]:
    """第⑤條對一段 `run:` 的判決（擋得住合併那幾個 job 才會叫到它）。"""
    return ci._receipt_step_problems("樣本那一步 的 run:", body, settings())


def test_each_separator_cuts_a_line_into_its_own_steps() -> None:
    """`&&`、`;`、`||`、`|` 四種各一例：切出來的段落逐項比對，不比段數。"""
    for separator in SEPARATORS:
        line = f"uv sync --locked {separator} {CHECK}"
        assert segments(line) == ["uv sync --locked", CHECK], line


def test_a_plumbing_first_segment_does_not_cover_the_check_after_it() -> None:
    """水管開頭不再讓整行過關：後面那一段是檢查就紅，而且訊息說得出是第幾段。"""
    for separator in SEPARATORS:
        hits = problems(f"{FETCH} {separator} {CHECK}")
        assert hits, f"水管串一支檢查（{separator}）應該紅，實際沒有任何一筆：{separator}"
        assert [h for h in hits if "第 2 段" in h and CHECK in h] == hits, hits


def test_a_line_whose_every_segment_is_plumbing_is_still_clean() -> None:
    """反向對照：每一段都是水管就不紅——切段不是把整行改判成紅。"""
    assert problems(f"uv sync --locked && {FETCH}") == []


def test_a_separator_inside_quotes_is_not_a_separator() -> None:
    """引號裡的分隔符不切（跟行內註解共用同一把尺 QUOTED_RE）。"""
    quoted = "grep -c 'a && b' pyproject.toml"
    assert ci._split_outside_quotes(quoted, ci.SEGMENT_SEP_RE) == [quoted]
    # 剝行內註解那一步會把引號裡的內容一起剝掉（那把尺本來就這樣），照樣只有一段。
    assert segments(quoted) == ["grep -c  pyproject.toml"]


def test_the_scribe_covers_what_follows_its_double_dash_but_not_the_next_segment() -> None:
    """抄寫員那一段 `--` 之後由它記離開碼；`--` 之後被 `&&` 切出來的下一段照判。"""
    wrapped = f"{scribe()} --name pytest --out-dir governance/receipts/steps -- uv run pytest"
    assert problems(wrapped) == []
    hits = problems(f"{wrapped} && uv run ruff check")
    assert hits, "`--` 之後被 && 切出來的那一段是 shell 層的另一個命令，應該紅"
    assert [h for h in hits if "第 2 段" in h and "uv run ruff check" in h] == hits, hits
