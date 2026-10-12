"""ci-jobs-cannot-die-quietly 第⑤條的切段與分類那一半。

主檔接近整檔行數上限，因此把命令切段、抄寫員與水管分類搬到這裡；第②條的管線判斷
也共用這裡的引號與切段零件。這是輔助模組，自己沒有規矩卡；判準由 ci 卡傳入，
設定的形狀仍由主檔驗證。第⑤條的完整說明在 ci_jobs_cannot_die_quietly 主檔。
"""
from __future__ import annotations

import re
import shlex

from governance.exit_codes import ToolBroken
from governance.loader import setting_strings, setting_text

# 認引號用的：**只有這一把尺**，第 2 條（管線）與第 5 條（切段、分類）認引號認的都是它，
# 兩份會各自漂。差別在拿它做什麼：第 2 條把引號裡的內容整段剝掉（``grep 'a|b'`` 不是管線），
# 第 5 條只拿它標出「哪一段是引號裡面」，內容原封不動留著（見 :func:`_command_text`）。
QUOTED_RE = re.compile(r"'[^']*'|\"[^\"]*\"")

# 一段命令丟到背景跑的尾巴。``&`` 結尾的那一段 shell 不等它，離開碼一定不會被記，
# 所以第 5 條一律紅——包了抄寫員也一樣（抄寫員本人被丟到背景，收據跟著沒人等）。
BACKGROUND_SUFFIX = "&"

# 第 5 條把一行 shell 切成一段一段用的分隔符。正則的交替是**有序**的，所以 ``&&`` 與 ``||``
# 排在單一個 ``|`` 前面——邏輯或先被吃掉，不會被當成管線切開（舊版靠一個佔位字元做同一件事，
# 現在切段與管線共用 :func:`_split_outside_quotes`，佔位字元跟著退休）。
SEGMENT_SEP_RE = re.compile(r"&&|\|\||;|\|")

ARGUMENT_MARKERS_KEY = "plumbing_forbidden_argument_markers"
ARGUMENT_MARKER_COMPARISONS = ("token_contains", "token_ends_with")


def _quoted_chunks(text: str) -> list[tuple[str, bool]]:
    """把一行 shell 切成 ``(片段, 這一段在引號裡嗎)``，順序原封不動。

    引號認的是 :data:`QUOTED_RE`。剝註解（:func:`_command_text`）與切段
    （:func:`_split_outside_quotes`）都從這裡拿同一份切法，兩邊才不會對「哪裡是引號裡面」
    有兩種看法。
    """
    chunks: list[tuple[str, bool]] = []
    cursor = 0
    for quoted in QUOTED_RE.finditer(text):
        chunks.append((text[cursor : quoted.start()], False))
        chunks.append((quoted.group(0), True))
        cursor = quoted.end()
    chunks.append((text[cursor:], False))
    return chunks



def _strip_comment(raw: str) -> str:
    """一行 shell 剝掉引號裡的內容、再剝掉行內註解（引號外第一個 ``#`` 之後全丟）。

    **第 2 條（管線）專用**。它要判的是「這一行有沒有多段管線」，引號裡的 ``|`` 不算管線，
    整段剝掉最省事。第 5 條**不能**用這一把尺：剝掉引號之後
    ``"uv run python -m governance.checks.x --scan-root ."``（整行被一對引號包住，shell 照樣
    會執行它）會變成空字串，於是「沒有命令就沒有離開碼可記」把它當殘骸跳過——那是 2026-09-10
    修掉的漏放。第 5 條改用 :func:`_command_text`（保留引號）。
    引號裡的 ``#``（``echo "a#b"``）不是註解，所以先剝引號再切。
    """
    return QUOTED_RE.sub("", raw).split("#", 1)[0]



def _command_text(raw: str) -> str:
    """一行 shell 剝掉行內註解，**引號原封不動留著**（第 5 條分類用的那一把尺）。

    跟 :func:`_strip_comment` 的差別只有一件事：引號裡的內容留著。留著才分得出「這一段是不是
    整行被引號包住」——被包住的那一段不是空的、也不是抄寫員、也不是水管，所以第 5 條判它紅
    （fail-closed：認不出就紅）。``#`` 在引號裡面（``echo "a#b"``）不是註解，所以只在引號
    外面的片段裡找第一個 ``#``。
    """
    kept: list[str] = []
    for chunk, is_quoted in _quoted_chunks(raw):
        if is_quoted:
            kept.append(chunk)
            continue
        if "#" in chunk:
            kept.append(chunk.split("#", 1)[0])
            break
        kept.append(chunk)
    return "".join(kept)



def _split_outside_quotes(text: str, separator: re.Pattern[str]) -> list[str]:
    """依 ``separator`` 把一行 shell 切段，引號裡的分隔符不算分隔符。

    引號認的是 :data:`QUOTED_RE`（經 :func:`_quoted_chunks`）：引號那一段原封不動接回去，
    只有引號外面命中的才切，所以 ``grep 'a|b'``、``sh -c "a && b"`` 都只有一段。
    回來的每一段去掉頭尾空白、空的一律丟掉——``a |  | b`` 中間那一段沒有命令。
    這裡丟掉空段**不是**放行：第 5 條在外面守著「一行切完一段都不剩就紅」（見
    :func:`_receipt_step_problems`），空段從此不會變成一個安靜的放行。

    **引號追蹤是有對象的，不是多餘的防禦**：第 5 條餵進來的是 :func:`_command_text` 的產物
    （引號留著），``sh -c "a && b"`` 的 ``&&`` 在引號裡、不切，整段當一個命令去分類（認不出，
    紅）。第 2 條餵進來的是 :func:`_strip_comment` 的產物（引號已經剝掉），對它來說這幾行
    只是不做事而已——兩條路共用一個函式，是為了「哪裡是引號裡面」只有一種看法。
    """
    parts: list[str] = []
    buffer = ""
    for chunk, is_quoted in _quoted_chunks(text):
        if is_quoted:
            buffer += chunk
            continue
        pieces = separator.split(chunk)
        buffer += pieces[0]
        for piece in pieces[1:]:
            parts.append(buffer)
            buffer = piece
    parts.append(buffer)
    return [part.strip() for part in parts if part.strip()]



def _command_lines(body: str) -> list[str]:
    """一段 ``run:`` 裡真的是命令的那幾行。空行與 ``#`` 開頭的註解行不算。"""
    return [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("#")]



def _is_plumbing(line: str, prefixes: list[str]) -> bool:
    """這一行是不是水管步驟：比命令的**前幾個字**，不是比子字串。

    登記幾個字就比前幾個字：``uv sync`` 比前兩個字，兩筆 apt 水管各比前三個字。這是刻意的
    ——``uv`` 底下什麼都跑得起來，整個 ``uv`` 放行等於這一條沒在管；只登記 ``sudo`` 或
    ``sudo apt-get`` 也會把其他子命令一起放行。同樣的道理，``git``／``gh`` 這種入口
    （``git -c alias.… rc``、``gh alias set --shell`` 之後的自訂子命令）不准整個字登記；
    登記到子命令那一級只是比較窄、不是證明（``--upload-pack`` 這一類參數仍會把值交給本機
    shell 跑，2026-09-11 實測）。名單本身住在卡的 ``[settings]``，這裡不寫死。
    """
    words = line.split()
    return any(words[: len(head)] == head for head in (prefix.split() for prefix in prefixes) if head)



def _forbidden_plumbing_option_verdict(command: str, prefixes: list[str]) -> str:
    """水管段有沒有帶能改寫行為的禁用選項；空字串代表乾淨。

    長選項只認 token 等於前綴或 ``前綴=值``；短選項另認黏著值（例如 ``-oFoo``）。
    shell token 用 :func:`shlex.split` 解析，讓 ``"-o"`` 不能靠引號繞過。水管段若連 token
    都剖析不開就判紅，不把看不懂的命令放進免抄收據的洞。
    """
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return (
            f"命中水管名單，但 shell token（命令字）剖析不開（{exc}）——"
            "看不懂就不能確認它沒有禁用選項，照 fail-closed 判紅"
        )
    for token in tokens:
        for prefix in prefixes:
            is_short = prefix.startswith("-") and not prefix.startswith("--")
            if token == prefix or token.startswith(prefix + "=") or (
                is_short and token.startswith(prefix) and len(token) > len(prefix)
            ):
                return (
                    f"命中水管名單，但 token（命令字）{token!r} 命中卡上登記的"
                    f"禁用選項前綴 {prefix!r}——水管可以用這類選項在執行前塞命令或指定設定檔，"
                    "那一段就不再只是沒有判決可記的水管；要跑這種命令必須改成由抄寫員包住"
                )
    return ""



def _argument_markers(settings: dict[str, object]) -> tuple[list[str], list[str]]:
    """取水管參數的兩種具名比法；卡設定已先由 :func:`_settings_problems` 驗過。"""
    table = settings[ARGUMENT_MARKERS_KEY]
    if not isinstance(table, dict):
        raise ToolBroken(f"{ARGUMENT_MARKERS_KEY} 必須是表，實際是 {table!r}")
    return (
        setting_strings(table, "token_contains"),
        setting_strings(table, "token_ends_with"),
    )



def _forbidden_plumbing_argument_verdict(
    command: str,
    contains: list[str],
    ends_with: list[str],
) -> str:
    """水管段有沒有本地套件路徑或檔名；空字串代表乾淨。"""
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return (
            f"命中水管名單，但 shell token（命令字）剖析不開（{exc}）——"
            "看不懂就不能確認它不是本地套件檔，照 fail-closed 判紅"
        )
    for token in tokens:
        matched = next((marker for marker in contains if marker in token), "")
        comparison = "token_contains"
        if not matched:
            matched = next((marker for marker in ends_with if token.endswith(marker)), "")
            comparison = "token_ends_with"
        if matched:
            return (
                f"命中水管名單，但 token（命令字）{token!r} 以 {comparison} 比法命中卡上登記的"
                f"本地套件參數標記 {matched!r}——apt 會安裝本地套件檔或目錄，而本地套件檔的"
                "安裝腳本（postinst）可以跑任意命令，那一段就不再只是沒有判決可記的水管；"
                "要跑這種命令必須改成由抄寫員包住"
            )
    return ""



def _is_wrapped(command: str, wrapper: str) -> bool:
    """這一行是不是**開頭**就在叫抄寫員。比開頭不比子字串，理由見模組說明第 5 條。

    後面接什麼都可以（``-- 真正要跑的命令``），但抄寫員出現在別的位置一律不算：塞在
    ``--`` 後面的那一串是被抄寫員跑的對象，不是抄寫員本人。
    """
    return command == wrapper or command.startswith(wrapper + " ")



def _segment_verdict(
    command: str,
    wrapper: str,
    plumbing: list[str],
    forbidden_plumbing_options: list[str],
    forbidden_argument_contains: list[str],
    forbidden_argument_ends_with: list[str],
) -> str:
    """一段命令認不認得出來：認得出（留得下離開碼）回空字串，認不出回一句為什麼。

    **fail-closed**：只認兩種正面形狀——開頭等於卡上登記的 ``wrapper_command``（抄寫員：
    把那一步真實離開碼記成一片收據的那一層），或前幾個字命中卡上登記的
    ``plumbing_first_words``（水管步驟：本來就沒有判決可記的那幾種）。其他一律紅，
    包括看不懂的：``$(...)``、反引號、``eval "..."``、``sh -c "..."``、``FOO=1 ...``、
    ``exec``／``time``／``nohup`` 開頭那幾種，這一支都認不出來，所以都紅。

    判定用的字串**保留引號**，所以引號包住的 token 不算命中任何一種：``"uv sync" --locked``
    的第一個字是 ``"uv``，命不中水管；整行被包住的
    ``"uv run python -m governance.status.record_step ..."`` 也命不中抄寫員。

    命中水管之後還要過禁用選項與本地套件參數兩關：只對水管段套用，抄寫員段直接放行。
    兩種判斷各由對應的 ``_forbidden_plumbing_*_verdict`` 負責，設定仍只住在卡上。

    背景執行先判：``&`` 結尾那一段 shell 不等它，離開碼一定不會被記——連抄寫員被丟到背景
    都一樣（收據沒人等），所以這一格排在兩種正面形狀前面。
    """
    if command.endswith(BACKGROUND_SUFFIX):
        return (
            "以 `&` 結尾，被丟到背景跑——shell 不等它，那一段的離開碼一定不會被記，"
            "包了抄寫員也一樣（抄寫員本人在背景，收據沒人等）"
        )
    if _is_wrapped(command, wrapper):
        return ""
    if _is_plumbing(command, plumbing):
        option_verdict = _forbidden_plumbing_option_verdict(command, forbidden_plumbing_options)
        if option_verdict:
            return option_verdict
        return _forbidden_plumbing_argument_verdict(
            command,
            forbidden_argument_contains,
            forbidden_argument_ends_with,
        )
    return (
        f"開頭不是抄寫員（卡上登記的 {wrapper!r}），"
        f"前幾個字也不在卡上登記的水管名單 {plumbing} 裡——這一段我認不出是什麼。"
        "認不出就紅（fail-closed）：離開碼進不了收據時，"
        "0（乾淨）／1（抓到違規）／2（這一跑不算數）在雲端記的紅綠裡分不開，"
        "等於那一步只留下一個顏色。"
        "引號包住的 token 不算命中任何一種（整行包一對引號，shell 照樣執行它）；"
        "抄寫員寫在行內註解裡、或塞在 `--` 後面當被跑的對象，都不算包"
    )



def _receipt_step_problems(
    step_where: str, body: str, settings: dict[str, object]
) -> list[str]:
    """第 5 條：擋合併那幾個 job 底下這一步的每一**段**命令，要嘛包抄寫員、要嘛是水管。

    一行裡用 ``&&``／``;``／``||``／``|`` 串起來的每一段各自算一步（2026-09-10 補的縫）：
    只看整行開頭的話，``uv sync --locked && uv run python -m governance.checks.x --scan-root .``
    會因為開頭那一段是水管就整行放行，後面那支檢查的離開碼進不了收據，而卡面看起來守著。
    切段用 :func:`_split_outside_quotes`，引號裡的分隔符不切。

    **殘骸只認一種形狀**（2026-09-10 改成 fail-closed）：原始那一行去頭尾空白後是空的、
    或以 ``#`` 開頭——那兩種由 :func:`_command_lines` 濾掉。其他每一行都必須被分類，
    分類不出來就紅。舊版拿剝掉引號的版本去切段，於是整行被一對引號包住的命令剝完變成空字串，
    被「沒有命令就沒有離開碼可記」當殘骸跳過、段數 0、不紅——而 shell 真的會執行它，
    那是一條可用的繞道（issue #104）。現在切段吃的是 :func:`_command_text`（引號留著）。

    抄寫員那一段 ``--`` 後的子程序還要有卡上登記的開頭，避免被 true 或 env 掏空。``--`` 之後
    出現的分隔符會切出**下一段**，那是 shell 層的另一個命令（抄寫員管不到它），照判。
    """
    wrapper = setting_text(settings, "wrapper_command")
    child_prefix = setting_text(settings, "required_child_prefix")
    plumbing = setting_strings(settings, "plumbing_first_words")
    forbidden_plumbing_options = setting_strings(
        settings, "plumbing_forbidden_option_prefixes"
    )
    forbidden_argument_contains, forbidden_argument_ends_with = _argument_markers(settings)
    bad: list[str] = []
    for line in _command_lines(body):
        commands = _split_outside_quotes(_command_text(line), SEGMENT_SEP_RE)
        if not commands:
            bad.append(
                f"{step_where} 的 `{line}` 切完一段命令都不剩——這一行我認不出是什麼。"
                "殘骸只認「原始那一行去頭尾空白後是空的、或以 `#` 開頭」那兩種，"
                "其他一律要分類得出來，分類不出來就紅（fail-closed）"
            )
            continue
        for index, command in enumerate(commands, start=1):
            if _is_wrapped(command, wrapper):
                child = command.partition(" -- ")[2]
                if not (child == child_prefix or child.startswith(child_prefix + " ")):
                    bad.append(f"{step_where} 的 `{line}` 第 {index} 段抄寫員 -- 後的子程序開頭不是卡上登記的 {child_prefix!r}")
            verdict = _segment_verdict(
                command,
                wrapper,
                plumbing,
                forbidden_plumbing_options,
                forbidden_argument_contains,
                forbidden_argument_ends_with,
            )
            if not verdict:
                continue
            bad.append(f"{step_where} 的 `{line}` 第 {index} 段 `{command}` {verdict}")
    return bad

