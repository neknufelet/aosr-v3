"""ci-jobs-cannot-die-quietly 第⑤條的切段與分類那一半：一行裡串接的每一段各自算一步。

不 spawn 整支檢查（那一半由後設測試的六回合跑，掃描根是必紅樣本樹），這裡直接餵切段函式
（`_split_outside_quotes`，把一行 shell 依分隔符切開）與分類函式（`_receipt_step_problems`，
判每一段是抄寫員還是水管）。判準（抄寫員那一串命令、水管名單）從主卡的 `[settings]` 讀，
不在這裡重寫一份——門檻只有一個家。

為什麼要有這一支：
* 2026-09-10（#103）之前判準比的是**整行**的前幾個字，所以 `uv sync --locked && 跑一支檢查`
  開頭是水管就整行放行，後面那支檢查的離開碼不會進收據。
* 同一天稍晚（#104）發現判準是 fail-open：分類前先剝掉引號，於是整行被一對引號包住的命令
  剝完變成空字串，被「沒有命令就沒有離開碼可記」當殘骸跳過，段數 0、不紅——而 shell 真的
  會執行那一行。判準因此改成「認不出就紅」，分類用的字串保留引號。

`ADVERSARIAL` 那張表是這一支的主體：每一列一個繞道形狀，寫明期望紅還是放行、以及為什麼。
必紅樣本樹只挑其中三種形狀（整行引號、`$(…)`、結尾 `&`）各做一棵，咬的是整支檢查的離開碼；
其餘每一種靠這張表。兩層各看一半。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple

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
    """production 的順序：先剝掉引號外的行內註解（引號留著），再依分隔符切段。"""
    return ci._split_outside_quotes(ci._command_text(line), ci.SEGMENT_SEP_RE)


def problems(body: str) -> list[str]:
    """第⑤條對一段 `run:` 的判決（擋得住合併那幾個 job 才會叫到它）。"""
    return ci._receipt_step_problems("樣本那一步 的 run:", body, settings())


class Row(NamedTuple):
    """對抗表的一列：輸入、期望（紅還是放行）、備註（為什麼是這個期望）。"""

    line: str
    red: bool
    why: str


def _wrapped(tail: str = "") -> str:
    """一行正常的抄寫員命令，`tail` 接在最後面（用來做背景執行那幾列）。"""
    return f"{scribe()} --name pytest --out-dir governance/receipts/steps -- uv run pytest{tail}"


def adversarial() -> tuple[Row, ...]:
    """對抗清單：每一列都真的餵過分類函式，結果二選一——紅，或列進卡上的已知洞。

    「誤紅」那幾列（`sh -c`／`bash -c`、環境變數前綴、抄寫員後面接 Tab、反斜線續行的第二行）
    照實記在卡的檔尾第 7 條：它們其實留得下離開碼，但這一支認不出來，所以判紅。誤紅是紅得
    安全的方向，今天名單上那個 job 底下零對象；真的踩到要回來改卡，不是在 workflow 那邊繞。
    """
    scribe_line = _wrapped()
    return (
        Row('"blueprint/remap_cards.py"', True, "整行被雙引號包住的腳本路徑：shell 照樣執行，舊版剝完引號變空字串被當殘骸放行"),
        Row(f'"{CHECK}"', True, "整行被雙引號包住的檢查命令：同上，這是 #104 那條已證實的繞道"),
        Row(f"'{CHECK}'", True, "單引號版本：QUOTED_RE 兩種引號都認，判準一樣"),
        Row(f"$({CHECK})", True, "命令替換：`$(` 開頭命不中抄寫員也命不中水管，認不出就紅"),
        Row(f"`{CHECK}`", True, "反引號版本：同上"),
        Row(f'eval "{CHECK}"', True, "eval 一個字串：第一個字是 eval，不在水管名單裡"),
        Row(f'sh -c "{CHECK}"', True, "誤紅，記在卡上：sh 不在水管名單、也不是抄寫員開頭"),
        Row(f'bash -c "{CHECK}"', True, "誤紅，記在卡上：卡上舊版把這一列寫成漏放，實測是誤紅，方向寫反了"),
        Row(f"FOO=1 {CHECK}", True, "環境變數前綴：這次選紅、記成誤紅的已知洞，不開「剝掉 KEY=VALUE 前綴再判」那條路"),
        Row(f"exec {scribe_line}", True, "exec 取代 shell 本身：第一個字是 exec，認不出"),
        Row(f"time {scribe_line}", True, "time 前綴：同上"),
        Row(f"nohup {scribe_line} &", True, "背景執行：shell 不等它，離開碼一定不會被記"),
        Row(_wrapped(" &"), True, "結尾 `&`：連抄寫員本人被丟到背景都紅——收據沒人等"),
        Row(scribe().replace(" -m ", "\t-m ", 1), True, "誤紅，記在卡上：抄寫員後面用 Tab 隔開參數（比的是後面接一個空格）"),
        Row(f"{CHECK} \\", True, "反斜線續行的第一行：本來就該紅（沒包抄寫員）"),
        Row("--scan-root .", True, "誤紅，記在卡上：反斜線續行的第二行被當成獨立命令"),
        Row(";", True, "整行只有一個分隔符：切完一段都不剩，訊息說「這一行我認不出是什麼」"),
        Row(scribe_line, False, "正常的抄寫員行：開頭就是卡上登記的那一串"),
        Row("uv sync --locked", False, "正常的水管行：前兩個字命中卡上登記的水管名單"),
        Row(FETCH, False, "正常的水管行（另一種）：第一個字命中"),
        Row(f"{scribe_line} && uv run ruff check", True, "抄寫員 `--` 後面被 && 切出來的下一段是 shell 層的另一個命令，照判"),
        Row(f"uv sync --locked && {CHECK}", True, "#103 修掉的那條：水管開頭不再讓整行過關（回歸）"),
    )


ADVERSARIAL = adversarial()


def test_every_adversarial_row_lands_on_its_expected_verdict() -> None:
    """對抗表逐列餵分類函式：期望紅的要紅、期望放行的要乾淨，逐列具名比對。"""
    verdicts = {row.line: bool(problems(row.line)) for row in ADVERSARIAL}
    expected = {row.line: row.red for row in ADVERSARIAL}
    assert verdicts == expected, [
        f"{row.line!r}（{row.why}）：期望 {'紅' if row.red else '放行'}"
        for row in ADVERSARIAL
        if verdicts[row.line] != row.red
    ]


def test_every_adversarial_row_carries_a_note() -> None:
    """每一列都要寫得出「為什麼是這個期望」——沒有備註的一列等於一個沒有理由的判準。"""
    assert [row.line for row in ADVERSARIAL if not row.why.strip()] == []


def test_a_line_that_is_only_quotes_is_not_debris() -> None:
    """整行引號那一條的形狀：舊尺剝完是空的，新尺留著引號、切得出一段、分類不出來所以紅。"""
    line = f'"{CHECK}"'
    assert ci._strip_comment(line) == "", "舊尺（管線那一條用的）剝完應該是空字串，這一列才證明得了洞"
    assert segments(line) == [line], "新尺要留著引號，整行就是一段"
    assert [h for h in problems(line) if "認不出" in h] == problems(line)


def test_an_unsplittable_line_says_it_cannot_be_recognised() -> None:
    """切完一段都不剩的行（只有分隔符）不是殘骸，是「認不出」——訊息要說得出來。"""
    assert segments(";") == []
    assert [h for h in problems(";") if "這一行我認不出是什麼" in h] == problems(";")


def test_only_a_blank_or_hash_line_counts_as_debris() -> None:
    """殘骸只認兩種形狀：原始那一行去頭尾空白後是空的、或以 `#` 開頭。"""
    assert ci._command_lines("\n   \n\t\n") == []
    assert ci._command_lines("  # 只是一行註解\n") == []
    assert problems("\n  \n# 只是一行註解\n") == []


def test_a_quoted_token_matches_neither_shape() -> None:
    """引號包住的 token 不算命中任何一種正面形狀——抄寫員與水管兩邊都要驗。"""
    quoted_plumbing = '"uv sync" --locked'
    quoted_scribe = f'"{scribe()}" --name pytest -- uv run pytest'
    assert problems(quoted_plumbing), "整串水管命令被引號包住就不算水管"
    assert problems(quoted_scribe), "整串抄寫員命令被引號包住就不算包"


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
    # 新尺留著引號，所以整行原封不動就是一段（舊尺會把引號裡的內容一起剝掉）。
    assert segments(quoted) == [quoted]


def test_an_inline_comment_is_stripped_but_a_hash_inside_quotes_is_not() -> None:
    """行內註解剝掉、引號裡的 `#` 留著——兩把尺換人之後這件事不准跟著漂。"""
    assert ci._command_text(f"{FETCH}  # 這是註解") == f"{FETCH}  "
    assert ci._command_text('echo "a#b"') == 'echo "a#b"'
    assert problems(f"uv run ruff check  # {scribe()}"), "抄寫員寫在行內註解裡不算包"


def test_the_scribe_covers_what_follows_its_double_dash_but_not_the_next_segment() -> None:
    """抄寫員那一段 `--` 之後由它記離開碼；`--` 之後被 `&&` 切出來的下一段照判。"""
    wrapped = _wrapped()
    assert problems(wrapped) == []
    hits = problems(f"{wrapped} && uv run ruff check")
    assert hits, "`--` 之後被 && 切出來的那一段是 shell 層的另一個命令，應該紅"
    assert [h for h in hits if "第 2 段" in h and "uv run ruff check" in h] == hits, hits
