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
必紅樣本樹挑代表形狀咬整支檢查的離開碼；這張表補齊逐 token 的對抗面。兩層各看一半，
本地套件路徑與只有 `.deb` 結尾的兩種則兩層都有，守住兩種比法不會互相遮蔽。
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import NamedTuple

import pytest

from governance.checks import ci_command_segments as segments_ci
from governance.checks import ci_jobs_cannot_die_quietly as ci
from governance.exit_codes import ToolBroken
from governance.loader import setting_text

REPO = Path(__file__).resolve().parents[1]
CARD = REPO / "governance" / "rules" / "ci-jobs-cannot-die-quietly.toml"

# 樣本用的命令。抄寫員那一串從卡上讀（見 `scribe()`），這幾個是被它跑的對象與一支檢查。
CHECK = "uv run python -m governance.checks.stub_check --scan-root ."
# uv 與 apt 的水管行；前者命中兩個字，後兩者命中三個字。
PLUMBING = "uv sync --locked"
PLUMBING_TOO = "uv sync --no-dev"
APT_UPDATE = "sudo apt-get update -q"
APT_INSTALL = "sudo apt-get install -y -q --no-install-recommends libglu1-mesa"
SLASH = chr(47)
LOCAL_DEB = f".{SLASH}tools{SLASH}x.deb"
ABS_CONFIG = f"{SLASH}tmp{SLASH}x.conf"
# 版控工具與 GitHub 那支命令列工具（gh）的「別名」寫法：底下能跑任意命令，所以它們的名字
# 2026-09-11 起不在水管名單裡。留在這裡當對抗列，不是當放行的樣本。
ALIAS_VCS = f"git -c alias.rc='!{CHECK}' rc"
ALIAS_GH = "gh rc"
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
    return segments_ci._split_outside_quotes(segments_ci._command_text(line), segments_ci.SEGMENT_SEP_RE)


def problems(body: str) -> list[str]:
    """第⑤條對一段 `run:` 的判決（擋得住合併那幾個 job 才會叫到它）。"""
    return segments_ci._receipt_step_problems("樣本那一步 的 run:", body, settings())


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

    「誤紅」那幾列（`sh -c`／`bash -c`、環境變數前綴、抄寫員後面接 Tab）
    照實記在卡的檔尾第 7 條：它們其實留得下離開碼，但這一支認不出來，所以判紅。誤紅是紅得
    安全的方向，今天名單上那個 job 底下零對象；真的踩到要回來改卡，不是在 workflow 那邊繞。
    """
    scribe_line = _wrapped()
    return (
        Row('"blueprint/remap_cards.py"', True, "整行被雙引號包住的腳本路徑：shell 照樣執行，舊版剝完引號變空字串被當殘骸放行"),
        Row(f'"{CHECK}"', True, "整行被雙引號包住的檢查命令：同上，這是 #104 那條已證實的繞道"),
        Row(f"'{CHECK}'", True, "單引號版本：兩種引號都認成字面，判準一樣"),
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
        Row("--scan-root .", True, "單獨一行的參數認不出就紅；接在續行後面時跟上一行接成同一個命令（見多行那幾題）"),
        Row(";", True, "整行只有一個分隔符：切完一段都不剩，訊息說「這一行我認不出是什麼」"),
        Row(ALIAS_VCS, True, "2026-09-11 收窄的那條：版控工具用 -c alias 定義一個驚嘆號別名就能跑任意命令，整個字放行等於沒在管"),
        Row(ALIAS_GH, True, "同上另一個入口：`gh alias set --shell` 之後這一串跑的是任意命令，看起來卻只是一個子命令"),
        Row(scribe_line, False, "正常的抄寫員行：開頭就是卡上登記的那一串"),
        Row(PLUMBING, False, "正常的水管行：前兩個字命中卡上登記的水管名單"),
        Row(PLUMBING_TOO, False, "正常的水管行（同一筆、換個參數）：比的是前兩個字，後面接什麼都可以"),
        Row(APT_UPDATE, False, "正常的 apt 更新水管行：三個字命中，-q 不在禁用選項前綴裡"),
        Row(APT_INSTALL, False, "正常的 apt 安裝水管行：-y、-q 與 --no-install-recommends 都不在禁單"),
        Row("sudo make install", True, "sudo 開頭不算數：水管名單登記的是 apt 的完整三個字"),
        Row(
            "sudo apt-get -o APT::Update::Pre-Invoke::=uv-run update",
            True,
            "apt 全域選項放在子命令前，前三個字命不中水管名單，fail-closed 判紅",
        ),
        Row(
            "sudo apt-get install -y -o DPkg::Pre-Invoke::=uv-run libfoo",
            True,
            "三個字命中後，-o 藏在後面仍要被禁用選項前綴抓到",
        ),
        Row(
            f"sudo apt-get install -y -c {ABS_CONFIG} libfoo",
            True,
            "三個字命中後，-c 指定設定檔仍要被禁用選項前綴抓到",
        ),
        Row(
            f"sudo apt-get install -y {LOCAL_DEB}",
            True,
            "apt 水管收到含斜線的 token 會安裝本地套件；安裝腳本可以跑任意命令，不能免收據",
        ),
        Row(
            "sudo apt-get install -y fake.deb",
            True,
            "沒有斜線但以 .deb 結尾仍會被 apt 當成本地套件檔，不能免收據",
        ),
        Row(f"{scribe_line} && uv run ruff check", True, "抄寫員 `--` 後面被 && 切出來的下一段是 shell 層的另一個命令，照判"),
        Row(f"{PLUMBING} && {CHECK}", True, "#103 修掉的那條：水管開頭不再讓整行過關（回歸）"),
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
    stripped = "".join(chunk for chunk, literal in segments_ci._quoted_chunks(line) if not literal)
    assert stripped == "", "舊尺（剝掉字面再分類）剝完應該是空字串，這一列才證明得了洞"
    assert segments(line) == [line], "新尺要留著引號，整行就是一段"
    assert [h for h in problems(line) if "認不出" in h] == problems(line)


def test_an_unsplittable_line_says_it_cannot_be_recognised() -> None:
    """切完一段都不剩的行（只有分隔符）不是殘骸，是「認不出」——訊息要說得出來。"""
    assert segments(";") == []
    assert [h for h in problems(";") if "這一行我認不出是什麼" in h] == problems(";")


def test_only_a_blank_or_hash_line_counts_as_debris() -> None:
    """殘骸只認兩種形狀：原始那一行去頭尾空白後是空的、或以 `#` 開頭。"""
    assert segments_ci._command_lines("\n   \n\t\n") == []
    assert segments_ci._command_lines("  # 只是一行註解\n") == []
    assert problems("\n  \n# 只是一行註解\n") == []
    # 註解只到行尾：下一行的命令照判。
    assert segments_ci._command_lines(f"{PLUMBING}  # 註解\n{CHECK}") == [PLUMBING, CHECK]
    # 每一行回來時去頭尾空白：空格與 Tab 縮排都不留（分類比的是命令開頭）。
    assert segments_ci._command_lines(f"  {PLUMBING} \n\t{CHECK}\t") == [PLUMBING, CHECK]


def test_a_quoted_token_matches_neither_shape() -> None:
    """引號包住的 token 不算命中任何一種正面形狀——抄寫員與水管兩邊都要驗。"""
    quoted_plumbing = '"uv sync" --locked'
    quoted_scribe = f'"{scribe()}" --name pytest -- uv run pytest'
    assert problems(quoted_plumbing), "整串水管命令被引號包住就不算水管"
    assert problems(quoted_scribe), "整串抄寫員命令被引號包住就不算包"


def test_each_separator_cuts_a_line_into_its_own_steps() -> None:
    """`&&`、`;`、`||`、`|` 四種各一例：切出來的段落逐項比對，不比段數。"""
    for separator in SEPARATORS:
        line = f"{PLUMBING} {separator} {CHECK}"
        assert segments(line) == [PLUMBING, CHECK], line


def test_a_plumbing_first_segment_does_not_cover_the_check_after_it() -> None:
    """水管開頭不再讓整行過關：後面那一段是檢查就紅，而且訊息說得出是第幾段。"""
    for separator in SEPARATORS:
        hits = problems(f"{PLUMBING} {separator} {CHECK}")
        assert hits, f"水管串一支檢查（{separator}）應該紅，實際沒有任何一筆：{separator}"
        assert [h for h in hits if "第 2 段" in h and CHECK in h] == hits, hits


def test_a_line_whose_every_segment_is_plumbing_is_still_clean() -> None:
    """反向對照：每一段都是水管就不紅——切段不是把整行改判成紅。"""
    assert problems(f"{PLUMBING} && {PLUMBING_TOO}") == []


def test_forbidden_option_prefixes_only_bite_plumbing_segments() -> None:
    """水管段的禁用選項逐種形狀都紅；安全 apt 選項與抄寫員段維持放行。"""
    forbidden = [
        "sudo apt-get install -o DPkg::Pre-Invoke::=uv-run libfoo",
        "sudo apt-get install --option=DPkg::Pre-Invoke::=uv-run libfoo",
        "sudo apt-get install -oDPkg::Pre-Invoke::=uv-run libfoo",
        f"sudo apt-get install -c {ABS_CONFIG} libfoo",
        f"sudo apt-get install --config-file={ABS_CONFIG} libfoo",
        f"sudo apt-get install -c{ABS_CONFIG} libfoo",
        "uv sync --locked -oUnsafeHook",
    ]
    verdicts = {line: problems(line) for line in forbidden}
    assert all(verdicts.values()), verdicts
    assert all("禁用選項前綴" in hit for hits in verdicts.values() for hit in hits), verdicts
    assert problems(APT_UPDATE) == []
    assert problems(APT_INSTALL) == []
    assert problems(f"{scribe()} --name apt -- {CHECK} -oUnsafeHook") == []


def test_local_package_arguments_only_bite_plumbing_segments() -> None:
    """水管段的路徑與 .deb 檔必紅；一般套件名、安全選項、uv 水管與抄寫員段維持放行。"""
    forbidden = [
        f"sudo apt-get install -y {LOCAL_DEB}",
        "sudo apt-get install -y fake.deb",
        f"sudo apt-get install -y {SLASH}tmp{SLASH}x.deb",
        f"sudo apt-get install -y .{SLASH}dir{SLASH}",
    ]
    verdicts = {line: problems(line) for line in forbidden}
    assert all(verdicts.values()), verdicts
    assert all(
        "本地套件檔" in hit and "安裝腳本" in hit and "任意命令" in hit
        for hits in verdicts.values()
        for hit in hits
    ), verdicts
    assert problems("sudo apt-get install -y libglu1-mesa") == []
    assert problems("sudo apt-get install -y --no-install-recommends libglu1-mesa") == []
    assert problems("sudo apt-get install -y fake.debian") == []
    assert problems("uv sync --locked") == []
    assert problems(f"{scribe()} --name apt -- {CHECK} {LOCAL_DEB}") == []


def test_forbidden_option_prefix_settings_are_required_and_must_be_strings() -> None:
    """新禁單讀不到或形狀壞時，設定載入必須以 ToolBroken 交給外殼回 2。"""
    missing = settings().copy()
    missing.pop("plumbing_forbidden_option_prefixes")
    malformed = settings().copy()
    malformed["plumbing_forbidden_option_prefixes"] = "-o"
    with pytest.raises(ToolBroken, match="plumbing_forbidden_option_prefixes"):
        ci._settings_problems(missing, "sample.toml")
    with pytest.raises(ToolBroken, match="plumbing_forbidden_option_prefixes"):
        ci._settings_problems(malformed, "sample.toml")


def test_forbidden_argument_marker_settings_are_required_and_have_named_comparisons() -> None:
    """本地套件禁單缺席或不是 contains／ends-with 兩張字串清單時，設定載入必須回 2。"""
    valid = settings().copy()
    valid["plumbing_forbidden_argument_markers"] = {
        "token_contains": ["/"],
        "token_ends_with": [".deb"],
    }
    ci._settings_problems(valid, "sample.toml")

    missing = valid.copy()
    missing.pop("plumbing_forbidden_argument_markers")
    malformed = valid.copy()
    malformed["plumbing_forbidden_argument_markers"] = ["/", ".deb"]
    wrong_comparison = valid.copy()
    wrong_comparison["plumbing_forbidden_argument_markers"] = {
        "token_contains": ["/"],
        "suffix": [".deb"],
    }
    for bad in (missing, malformed, wrong_comparison):
        with pytest.raises(ToolBroken, match="plumbing_forbidden_argument_markers"):
            ci._settings_problems(bad, "sample.toml")


def test_a_separator_inside_quotes_is_not_a_separator() -> None:
    """引號裡的分隔符不切（跟行內註解共用同一把尺 `_quoted_chunks`）。"""
    quoted = "grep -c 'a && b' pyproject.toml"
    assert segments_ci._split_outside_quotes(quoted, segments_ci.SEGMENT_SEP_RE) == [quoted]
    # 新尺留著引號，所以整行原封不動就是一段（舊尺會把引號裡的內容一起剝掉）。
    assert segments(quoted) == [quoted]


def test_an_inline_comment_is_stripped_but_a_hash_inside_quotes_is_not() -> None:
    """行內註解剝掉、引號裡的 `#` 留著——兩把尺換人之後這件事不准跟著漂。"""
    assert segments_ci._command_text(f"{PLUMBING}  # 這是註解") == f"{PLUMBING}  "
    assert segments_ci._command_text('echo "a#b"') == 'echo "a#b"'
    assert problems(f"uv run ruff check  # {scribe()}"), "抄寫員寫在行內註解裡不算包"


def test_the_scribe_covers_what_follows_its_double_dash_but_not_the_next_segment() -> None:
    """抄寫員那一段 `--` 之後由它記離開碼；`--` 之後被 `&&` 切出來的下一段照判。"""
    wrapped = _wrapped()
    assert problems(wrapped) == []
    hits = problems(f"{wrapped} && uv run ruff check")
    assert hits, "`--` 之後被 && 切出來的那一段是 shell 層的另一個命令，應該紅"
    assert [h for h in hits if "第 2 段" in h and "uv run ruff check" in h] == hits, hits


# 藏住後面命令的六種寫法（#595 第 2 步複查與之後補，2026-10-13 bash 實測）：舊版切段器都把被藏的
# `uv run pytest`（沒包抄寫員，本來就該紅）看漏。每一列：run 的內容、被判紅時訊息裡要有的字樣。
HIDDEN = "uv run pytest"
HIDING_SHAPES = {
    "單一個 & 在中間": (f"{_wrapped()} & {HIDDEN}", "有單一個 `&`"),
    "字中的 #": (f"{_wrapped()} --a=b#c && {HIDDEN}", f"段 `{HIDDEN}` 開頭不是抄寫員"),
    "跳脫的雙引號": (f'{_wrapped()} \\" && {HIDDEN} #"', f"段 `{HIDDEN}` 開頭不是抄寫員"),
    "ANSI-C 引號": (f"{_wrapped()} $'\\'' && {HIDDEN} #'", f"段 `{HIDDEN}` 開頭不是抄寫員"),
    "接續行接上 # 開頭的下一行": (f"{_wrapped()}\\\n#x && {HIDDEN}", f"段 `{HIDDEN}` 開頭不是抄寫員"),
    "跨行引號在 # 開頭的那一行收尾": (f"{_wrapped()} 'x\n#' && {HIDDEN}", "有引號沒在同一行收尾"),
}


@pytest.mark.parametrize("name", sorted(HIDING_SHAPES))
def test_a_command_hidden_from_the_old_segmenter_is_red(name: str) -> None:
    """六種寫法裡，那一段沒包的命令在 bash 眼中都在字面外、是一段獨立的命令（`&&` 那幾種是前一段
    成功才執行，`&` 那一種不等前一段）；切段器要照 bash 的讀法切出它、判紅。"""
    body, fragment = HIDING_SHAPES[name]
    hits = problems(body)
    assert hits, f"{name}：{body!r} 應該紅"
    assert [h for h in hits if fragment in h], (name, hits)


HARMLESS_SHAPES = [
    f"{_wrapped()} \\\n#x && {HIDDEN}",
    f"{_wrapped()} \\\n  --maxfail=1",
    f"{_wrapped()} x\\\\\n{PLUMBING}",
]


def test_shapes_bash_reads_as_harmless_stay_clean() -> None:
    """反向對照：bash 不會跑出沒包命令的寫法不准誤紅——續行前有空白時下一行的 # 是註解、
    續行把參數接回抄寫員那一行、`\\\\` 結尾不是續行（下一行自己一行）。"""
    wrapped = _wrapped()
    assert {body: problems(body) for body in HARMLESS_SHAPES} == {body: [] for body in HARMLESS_SHAPES}
    # 斷行那一層自己就要讀對（第⑤條會再重掃一次，重掃會把第一次讀錯的註解又剝掉，看不出來）。
    assert segments_ci._command_lines(f"{wrapped} \\\n#x && {HIDDEN}") == [wrapped]


def test_dollar_runs_decide_whether_a_single_quote_escapes() -> None:
    """奇數個 `$` 緊接的單引號是 `$'…'`（反斜線跳脫 `'`），偶數個是行程號接普通引號；兩邊讀反都會藏住命令。"""
    escaping = f"'\\'' && {HIDDEN} #'"  # 讀成 `$'…'` 才看得到 `&&`（bash 實測 t3、t8）
    plain = f"'a\\' && {HIDDEN} #'"  # 讀成普通引號才看得到 `&&`（bash 實測 t5、t9）
    # 後兩列是反向對照：`$'a\\' …'` 與 `$$'\\'' …'` 在 bash 裡整串是一個字，`&&` 在引號裡（t12、t13）。
    cases = (("$", escaping, True), ("$$$", escaping, True), ("$$", plain, True), ("$$$$", plain, True),
             ("$", plain, False), ("$$", escaping, False))
    verdicts = {dollars + tail: bool(problems(f"{_wrapped()} {dollars}{tail}")) for dollars, tail, _red in cases}
    assert verdicts == {dollars + tail: red for dollars, tail, red in cases}
    # 斷出來的那一行：註解從引號外的 `#` 起剝掉，`&&` 在引號外。
    assert segments_ci._command_lines(f"x $$'a\\' && {HIDDEN} #'") == [f"x $$'a\\' && {HIDDEN}"]
    assert segments_ci._command_lines(f"x $'\\'' && {HIDDEN} #'") == [f"x $'\\'' && {HIDDEN}"]


def test_comment_starts_only_at_the_beginning_of_a_word() -> None:
    """bash 只在字首認 `#`：`$#`、`${#x}`、`a#b`、`"a"#b` 都留著，空白、分號、行首之後的才是註解。"""
    for kept in ("echo $# ${#x} a#b", 'echo "a"#b', "echo \\#x"):
        assert segments_ci._command_text(kept) == kept, kept
    assert segments_ci._command_text("echo a;#b") == "echo a;"
    assert segments_ci._command_text("#整行") == ""
    # 註解起點逐字元一列：bash 的這幾個字元把字隔開，`#` 緊接在後就是新的一個字的開頭、是註解。
    starts = {" ": "a ", "\t": "a\t", "\n": "a\n", ";": "a;", "&": "a&", "|": "a|",
              "(": "a(", ")": "a)", "<": "a<", ">": "a>"}
    assert {char: segments_ci._command_text(f"a{char}#x") for char in starts} == starts


def test_lines_are_logical_lines_and_rescanning_them_changes_nothing() -> None:
    """一段 run 斷出來的每一行，再交給同一把尺斷一次、剝一次註解，要原封不動——第⑤條會把
    每一行重餵 `_command_text` 再切段，兩次看法不一樣就會在第二次多出或少掉一段。"""
    bodies = [row.line for row in ADVERSARIAL] + [body for body, _fragment in HIDING_SHAPES.values()] + HARMLESS_SHAPES
    bodies += [f"{_wrapped()} <<'{PLUMBING} \\'\n{PLUMBING} \\\n{HIDDEN}\n", "a 'b\\\nc' d", "a $'b\\\nc' d"]
    # 複查實例：雙引號跨接續行之後才收尾、後面接註解（舊版第一次把 `" # "` 當字面、重掃才剝掉註解）。
    bodies += ['a "a\\\nb" # "c"', 'a "it\'s \\\nx"']
    lines = [line for body in bodies for line in segments_ci._command_lines(body)]
    assert lines, "至少要斷出一行，這一題才證明得了東西"
    assert [line for line in lines if segments_ci._command_lines(line) != [line]] == []
    assert [line for line in lines if segments_ci._command_text(line) != line] == []


def test_a_quote_that_does_not_close_on_its_line_is_red_and_never_swallows_lines() -> None:
    """引號只在同一行收尾才算：沒收尾的那一行紅（跨行引號刻意不追），後面幾行照常各自一行——
    heredoc（`<<EOF` 把後面幾行當資料）裡的撇號不准把後面的真命令吞進引號裡（整段一起掃的話會）。"""
    heredoc = f"{_wrapped()} <<'EOF'\n{PLUMBING} it's\nEOF\n{HIDDEN}\necho \\'"
    assert segments_ci._command_lines(heredoc) == [
        f"{_wrapped()} <<'EOF'", f"{PLUMBING} it's", "EOF", HIDDEN, "echo \\'",
    ]
    hits = problems(heredoc)
    assert [h for h in hits if f"段 `{HIDDEN}` 開頭不是抄寫員" in h], hits
    assert [h for h in hits if "有引號沒在同一行收尾" in h and f"{PLUMBING} it's" in h], hits
    assert [h for h in problems(f"{_wrapped()} --x='a\nb'") if "有引號沒在同一行收尾" in h]
    assert problems(f"{_wrapped()} --x='a b' \"c\"") == [], "同一行收尾的引號照常是字面"


def test_an_open_single_quote_does_not_join_the_next_line() -> None:
    """bash 在單引號與 `$'…'` 裡不接續行（bash 實測 t15、t17：反斜線與換行原樣留在字串裡），雙引號裡照接
    （t16，見下一題）。單引號沒在那一行收尾就不接，那一行照「引號沒在同一行收尾」紅——接起來的話，
    第⑤條重掃時那對引號在接好的那一行收尾，看不出沒收尾。換行之後照常接；收尾了的單引號不擋。"""
    assert segments_ci._command_lines("a 'b\\\nc' d") == ["a 'b\\", "c' d"]
    assert segments_ci._command_lines("a $'b\\\nc' d") == ["a $'b\\", "c' d"]
    assert segments_ci._command_lines("a 'b\nc \\\nd") == ["a 'b", "c d"]
    assert segments_ci._command_lines("a 'b' \\\nc") == ["a 'b' c"]
    hits = problems(f"{_wrapped()} 'b\\\nc'")
    assert [h for h in hits if "有引號沒在同一行收尾" in h and "第 1 段" in h], hits


def test_a_heredoc_is_red_because_its_end_line_is_not_a_command() -> None:
    """字面外的 `<<` 一律紅。heredoc 的結束行不是命令：`<<'D \\'` 配結束行 `D \\` 時，接續行會把結束行
    跟下一行接成一行、下一行的命令就看不到了（#595 第 2 步之後複查實例；bash 實測 t14：讀完結束行就回到
    命令，下一行照常執行）。here-string 與算式裡的位移順便紅；引號裡、跳脫的 `<<` 不算。"""
    body = f"{_wrapped()} <<'{PLUMBING} \\'\n{PLUMBING} \\\n{HIDDEN}\n"
    assert segments_ci._command_lines(body)[1:] == [f"{PLUMBING} {HIDDEN}"], "結束行與下一行接成一行，這一題才證明得了洞"
    hits = problems(body)
    assert [h for h in hits if "有 `<<`" in h and "第 1 段" in h], hits
    shapes = ("<<EOF", "<<-EOF", "<<<x", "$(( 1 << 2 ))")
    red = {shape: bool([h for h in problems(f"{_wrapped()} {shape}") if "有 `<<`" in h]) for shape in shapes}
    assert red == dict.fromkeys(shapes, True)
    assert problems(f"{_wrapped()} --x='a << b' \"<<\" \\<\\<") == []


def test_an_expansion_is_red_even_inside_double_quotes() -> None:
    """`$(`、`${`、`$[`、反引號、`<(`、`>(` 一律紅（複查實例，bash 實測 t21～t23）：替換式之後的 `#` 不是註解
    （`$(printf x)#b && 命令`）、`${x:- #b}` 裡的也不是，雙引號裡的 `$(cat <<…)` 開得了 heredoc——三種都會讓
    切段器剝掉或接錯後面那段沒包的命令。第二輪複查再兩種：雙引號裡被接續行拆開的 `"$\\` 換行 `(…)"`
    bash 先接起來照樣替換（t26），舊式算術 `$[ '$(…)' ]` 裡的單引號擋不住替換（t27）。單引號、`$'…'` 裡與
    跳脫的不展開，照常放行。"""
    wrapped = _wrapped()
    red = {
        "替換式後的 #": f"{wrapped} $(printf x)#b && {HIDDEN}",
        "參數展開裡的 #": f"{wrapped} ${{x:- #b}} && {HIDDEN}",
        "雙引號裡的 heredoc": f"{wrapped} \"$(cat <<'{PLUMBING} \\')\"\n{PLUMBING} \\\n{HIDDEN}\n",
        "雙引號裡的反引號": f'{wrapped} "`true`"',
        "行程替換": f"{wrapped} <(printf x)#b && {HIDDEN}",
        "輸出的行程替換": f"{wrapped} >(cat)",
        "雙引號裡被接續行拆開的 $(": f'{wrapped} "$\\\n(printf x)"',
        "雙引號裡被接續行拆開的 ${": f'{wrapped} "$\\\n{{x}}"',
        "舊式算術裡的單引號": f"{wrapped} $[ '$(printf x)' ]",
        "雙引號裡的舊式算術": f'{wrapped} "$[1+1]"',
    }
    verdicts = {name: bool([h for h in problems(body) if "有 `$(`" in h]) for name, body in red.items()}
    assert verdicts == dict.fromkeys(red, True)
    clean = [f"{wrapped} --x='$(x) ${{y}} `z`'", f"{wrapped} --x=$'$(x)'", f"{wrapped} --x=$y", f'{wrapped} --x="<(x)"',
             f"{wrapped} --x=\\`x"]
    assert {body: problems(body) for body in clean} == {body: [] for body in clean}


def test_rule_two_reads_lines_with_the_same_ruler() -> None:
    """第②條（管線）也用同一把尺斷行：接續行的下一行以 `|` 開頭，接起來是一條多段管線
    （舊版各行分開認，第二行切完只剩一段）；引號外跳脫的 `\\|` 不是管線。"""
    assert ci._pipelines("uv run pytest \\\n  | tee log") == ["uv run pytest   | tee log"]
    assert ci._pipelines("grep a \\| b") == []
    # 不先剝掉字面再重掃：`''#x` 的 `#` 在字中（bash 實測 t24：管線照建），剝掉 `''` 之後會跑到字首、被當註解。
    shapes = [f"{PLUMBING} ''#x | tee log", f'{PLUMBING} "a"#x | tee log', f"{PLUMBING} \\x#x | tee log"]
    assert {shape: ci._pipelines(shape) for shape in shapes} == {shape: [shape] for shape in shapes}


def test_a_double_quote_does_not_span_lines_either() -> None:
    """雙引號跨真的換行不算收尾；引號裡的接續行照 bash 接下去（t16）：`"b\\` 換行 `c"` 整串是一個字面，
    裡面的撇號不是沒收尾的單引號（t25：`"it's \\` 換行 `x"` bash 讀成 `it's x`）。"""
    assert [h for h in problems(f'{_wrapped()} --x="a\nb"') if "有引號沒在同一行收尾" in h]
    assert segments_ci._command_lines('a "b\\\nc" d') == ['a "b\\\nc" d']
    assert segments_ci._command_lines('a "it\'s \\\nx" d') == ['a "it\'s \\\nx" d']
    # 沒收尾的雙引號也照接續行（bash 在雙引號裡接；只有單引號與 `$'…'` 擋），那一行照樣因為引號沒收尾而紅。
    assert segments_ci._command_lines('a "b \\\nc') == ['a "b c']
    assert problems(f'{_wrapped()} "it\'s \\\nx"') == []
    # 雙引號裡的 `\"` 不收尾、`\\` 是一個反斜線（下一個 `"` 才收尾）：合法的寫法不准誤紅。
    escaped = [f'{_wrapped()} --x="a\\"b"', f'{_wrapped()} --x="a\\\\" && {PLUMBING}']
    assert {body: problems(body) for body in escaped} == {body: [] for body in escaped}


def test_a_redirect_is_not_a_background_ampersand() -> None:
    """`2>&1`、`>&2`、`&>log`、`&&` 與引號裡的 `&` 都不是丟到背景；只有字面外單獨一個才是。"""
    wrapped = _wrapped()
    clean = [f"{wrapped} 2>&1", f"{wrapped} >&2", f"{wrapped} &>log", f"{wrapped} && {PLUMBING}",
             f"{wrapped} --x='a & b'"]
    assert {body: problems(body) for body in clean} == {body: [] for body in clean}
    assert [h for h in problems(f"{wrapped} 2>&1 & {PLUMBING}") if "有單一個 `&`" in h]
    # 背景先判、排在兩種正面形狀前面：水管起頭的那一段也一樣（不然 `水管 & 命令` 整段當水管放行）。
    assert [h for h in problems(f"{PLUMBING} & {HIDDEN}") if "有單一個 `&`" in h and "第 1 段" in h]


def test_rule_card_mount_reads_lines_with_the_same_ruler() -> None:
    """rule-card-required-fields 認「卡宣告的 job 真的叫到檢查」也用同一把尺斷行：接續行接起來才算數，
    註解裡的不算（舊版各行分開認，續行寫法認不出是掛載）。"""
    from governance.checks import rule_card_required_fields as cards

    wrapper, prefix = cards._mount_settings(REPO)
    module = "governance.checks.stub_check"
    continued = {"steps": [{"run": f"{wrapper} --name stub \\\n  -- {prefix} {module} --scan-root .\n"}]}
    commented = {"steps": [{"run": f"{PLUMBING}  # {wrapper} -- {prefix} {module}\n"}]}
    assert cards._job_calls_check(continued, module, REPO) is True
    assert cards._job_calls_check(commented, module, REPO) is False
