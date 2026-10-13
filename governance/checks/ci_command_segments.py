"""ci-jobs-cannot-die-quietly 第⑤條的切段與分類那一半。

主檔接近整檔行數上限，因此把命令切段、抄寫員與水管分類搬到這裡；第②條的管線判斷
也共用這裡的引號與切段零件。這是輔助模組，自己沒有規矩卡；判準由 ci 卡傳入，
設定的形狀仍由主檔驗證。第⑤條的完整說明在 ci_jobs_cannot_die_quietly 主檔。
"""
from __future__ import annotations

import re
import shlex
from pathlib import PurePosixPath

from governance.exit_codes import ToolBroken
from governance.loader import setting_strings, setting_text

# 一段命令裡丟到背景跑的那個 ``&``：單一個、不是 ``&&``、也不是轉向（``2>&1``、``&>log``、``<&3``）。
# 不只看結尾：``判決 & 水管`` 的判決在背景跑、整步回的是水管的離開碼（#595 第 2 步複查）。
# 那一段 shell 不等它，離開碼一定不會被記，所以第 5 條一律紅——包了抄寫員也一樣（收據跟著沒人等）。
BACKGROUND_RE = re.compile(r"(?<![&<>])&(?![&>])")

# ``#`` 前面是這幾個字元（或在行首）才是註解起點：bash 只在字首認註解，``word#tail``、``$#``、
# ``"a"#b`` 的 ``#`` 都是字的一部分（#595 第 2 步複查：舊版把字中的 ``#`` 當註解，連後面的 ``||`` 一起剝掉）。
COMMENT_AFTER = frozenset(" \t\n;&|()<>")

# 第 5 條把一行 shell 切成一段一段用的分隔符。正則的交替是**有序**的，所以 ``&&`` 與 ``||``
# 排在單一個 ``|`` 前面——邏輯或先被吃掉，不會被當成管線切開（舊版靠一個佔位字元做同一件事，
# 現在切段與管線共用 :func:`_split_outside_quotes`，佔位字元跟著退休）。
SEGMENT_SEP_RE = re.compile(r"&&|\|\||;|\|")

# 展開與替換：``$(…)``、``${…}``、``$[…]``（舊式算術）、反引號在雙引號裡照樣展開，``<(…)``／``>(…)`` 只在引號外。
# 裡面可以藏命令、``#`` 與 heredoc，替換式之後的 ``#`` 也不是註解（``$(printf x)#b``）；``$[ '…' ]`` 裡的單引號
# 擋不住命令替換（bash 實測 t27）——這一支不追，見到就紅。
EXPANSION_MARKS = ("$(", "${", "$[", "`")
PROCESS_SUBSTITUTION_MARKS = ("<(", ">(")

ARGUMENT_MARKERS_KEY = "plumbing_forbidden_argument_markers"
ARGUMENT_MARKER_COMPARISONS = ("token_contains", "token_ends_with")


def _uv_child(tokens: list[str]) -> list[str]:
    """剝 uv run 的選項；有值的選項吃下一字，等號寫法已在同一字。"""
    value_options = {'--with', '--with-editable', '--with-requirements', '--extra', '--group', '--only-group',
                     '--no-group', '--no-extra', '--python', '-p', '-w', '-i', '-P', '-C', '--project', '--directory', '--package',
                     '--index', '--default-index', '--index-url', '--extra-index-url', '--find-links', '-f',
                     '--config-file', '--env-file', '--cache-dir', '--link-mode', '--resolution', '--prerelease',
                     '--python-platform', '--color', '--keyring-provider', '--refresh-package', '--upgrade-package',
                     '--reinstall-package', '--no-binary-package', '--no-build-package', '--no-editable-package',
                     '--index-strategy', '--upgrade-group', '--fork-strategy', '--exclude-newer',
                     '--exclude-newer-package', '--no-sources-package', '--config-setting',
                     '--config-settings-package', '--no-build-isolation-package', '--allow-insecure-host'}
    child = tokens[2:]
    while child and child[0].startswith('-'):
        option = child[0]
        child = child[1:]
        if option == '--':
            break
        if option in value_options:
            child = child[1:]
    return child


def is_pytest_argv(tokens: list[str], wrapper: str) -> bool:
    """片段 argv 的程式位置辨識；字串參數裡的 pytest 不算。"""
    prefix = shlex.split(wrapper)
    child = tokens
    while child:
        if prefix and child[:len(prefix)] == prefix:
            child = child[child.index('--') + 1:] if '--' in child else []
        elif PurePosixPath(child[0]).name == 'uv' and child[1:2] == ['run']:
            child = _uv_child(child)
        elif PurePosixPath(child[0]).name in {'python', 'python3'}:
            return child[1:3] == ['-m', 'pytest']
        else:
            return PurePosixPath(child[0]).name in {'pytest', 'py.test'}
    return False


def _quoted_chunks(text: str) -> list[tuple[str, bool]]:
    """把一段 shell 切成 ``(片段, 這一段是字面嗎)``，順序原封不動；註解與接續行在這裡就拿掉。

    **只有這一把尺**：第 2 條（管線）與第 5 條（切段、分類）、斷行（:func:`_command_lines`）、
    剝註解（:func:`_command_text`）與切段（:func:`_split_outside_quotes`）都從這裡拿同一份切法，
    才不會對「哪裡是字面、哪裡是註解、哪裡斷行」有兩種看法。
    兩條都只在非字面的片段裡找分隔符（``grep 'a|b'`` 不是管線）。shell 先認的三件事在這裡一次做完：

    * 字面（:func:`_literal_end`）：引號與反斜線跳脫；引號要在同一行收尾才算（雙引號可以跨接續行）。
    * 接續行：引號外的反斜線緊接換行，兩個字元一起拿掉，前後接成同一個字——``A\\`` 換行
      ``#x`` 是 ``A#x``，``A \\`` 換行 ``#x`` 的 ``#x`` 才是註解。那一行有單引號沒收尾時不接：
      bash 在單引號與 ``$'…'`` 裡不接續行（雙引號裡照接），接起來第⑤條重掃時那對引號會在
      接好的那一行收尾、看不出沒收尾（#595 第 2 步之後複查，bash 實測 t15、t17）。
    * 註解：字首的 ``#`` 到行尾整段丟掉、換行留著；註解裡的引號與反斜線都不算數。

    舊版一行一行認、不接續行，``A\\`` 換行 ``#x && 命令`` 的第二行被當註解丟掉；``$'…'`` 也藏得住
    後面的命令（#595 第 2 步之後補，2026-10-13 bash 實測，腳本存在工作區 evidence/step2b）。
    跨行的引號刻意不追：整段一起掃的話，heredoc（``<<EOF`` 把後面幾行當資料）裡的一個撇號會把後面
    幾行吞成引號裡、藏住真的命令；所以沒在同一行收尾的引號當普通字元，第⑤條見到它就紅。
    """
    chunks: list[tuple[str, bool]] = []
    plain, index, previous, open_single = "", 0, "\n", False
    while index < len(text):
        char = text[index]
        if text.startswith("\\\n", index) and not open_single:
            index += 2
            continue
        if char == "#" and previous in COMMENT_AFTER:
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            continue
        # ``$'…'`` 裡反斜線會跳脫 ``'``；``$$`` 是行程號，所以只有奇數個 ``$`` 緊接的單引號才是這一種。
        ansi = char == "'" and (len(plain) - len(plain.rstrip("$"))) % 2 == 1
        end = None if text.startswith("\\\n", index) else _literal_end(text, index, ansi)
        if end is None:
            plain += char
            previous = char
            open_single = char != "\n" and (open_single or char == "'")
            index += 1
            continue
        chunks.append((plain, False))
        chunks.append((text[index:end], True))
        plain, index, previous = "", end, "a"  # 字面一定是字的一部分
    chunks.append((plain, False))
    return chunks


def _literal_end(text: str, index: int, ansi: bool) -> int | None:
    """``index`` 起是字面的話回它的結尾，不是回 None。

    字面＝shell 不拿來當分隔、註解或引號的那幾種：一對單引號（裡面什麼都不跳脫）、一對雙引號
    與 ``ansi``（``$'…'``）那一對單引號（裡面 ``\\"``／``\\'`` 不收尾）、引號外反斜線跳脫的那一個字元
    （``\\"`` 不開引號、``\\|`` 不是管線；#595 第 2 步複查：舊版把 ``\\"`` 當引號開頭，藏住後面的 ``||``）。
    沒在同一行收尾的引號當普通字元，分隔照切、第⑤條見到它就紅——認不出就往紅的方向倒。只有雙引號
    可以跨接續行（反斜線緊接換行，bash 在雙引號裡也當接續行）；跨真的換行照樣不算收尾。
    """
    head = text[index]
    if head == "\\":
        return min(index + 2, len(text))
    if head == "'" and not ansi:
        close = text.find("'", index + 1)
        return None if close < 0 or "\n" in text[index:close] else close + 1
    if head in "'\"":
        cursor = index + 1
        while cursor < len(text) and text[cursor] != "\n":
            if text[cursor] == "\\":
                # 雙引號裡反斜線接換行是接續行：bash 拿掉兩個字、字串接到下一行（t16）；``$'…'`` 裡不是（t17）。
                cursor += 2 if head == '"' or text[cursor + 1 : cursor + 2] != "\n" else 1
            elif text[cursor] == head:
                return cursor + 1
            else:
                cursor += 1
    return None



def _command_text(raw: str) -> str:
    """一段 shell 剝掉註解與接續行，**引號原封不動留著**（第 5 條分類用的那一把尺）。

    **引號留著**才分得出「這一段是不是整行被引號包住」——被包住的那一段不是空的、也不是抄寫員、
    也不是水管，所以第 5 條判它紅（fail-closed：認不出就紅）。2026-09-10 之前分類吃的是剝掉字面的
    版本，``"uv run python -m governance.checks.x --scan-root ."`` 剝完是空字串、被當殘骸跳過，
    而 shell 照樣執行它（issue #104）。哪裡是註解由 :func:`_quoted_chunks` 認：字首的 ``#`` 才算，
    引號裡（``echo "a#b"``）與字中（``a#b``）的不算。餵它 :func:`_command_lines` 交出的一行，
    回來的跟餵進去的一樣（那一行的註解與接續行已經拿掉了）。
    """
    return "".join(chunk for chunk, _literal in _quoted_chunks(raw))



def _split_outside_quotes(text: str, separator: re.Pattern[str]) -> list[str]:
    """依 ``separator`` 把一行 shell 切段，引號裡的分隔符不算分隔符。

    字面由 :func:`_quoted_chunks` 認：引號與跳脫字元那一段原封不動接回去，
    只有引號外面命中的才切，所以 ``grep 'a|b'``、``sh -c "a && b"`` 都只有一段。
    回來的每一段去掉頭尾空白、空的一律丟掉——``a |  | b`` 中間那一段沒有命令。
    這裡丟掉空段**不是**放行：第 5 條在外面守著「一行切完一段都不剩就紅」（見
    :func:`_receipt_step_problems`），空段從此不會變成一個安靜的放行。

    **引號追蹤是有對象的，不是多餘的防禦**：``sh -c "a && b"`` 的 ``&&`` 在引號裡、不切，整段當一個
    命令去分類（認不出，紅）。第 2 條與第 5 條都直接餵 :func:`_command_lines` 交出的那一行——舊版
    第 2 條先剝掉字面再交給這裡重掃，``uv sync ''#x | 命令`` 剝完變成 ``uv sync #x | 命令``，字中的
    ``#`` 跑到字首、被當註解，管線就看不到了（#595 第 2 步之後複查）。
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
    """一段 ``run:`` 在 shell 眼中的一行一行：註解拿掉、接續行接起來。

    換行都斷行（字面不跨真的換行，見 :func:`_quoted_chunks`），只有接續行接成同一行（那一行有單引號
    沒收尾時不接；雙引號裡的接續行留在字面裡）；空的、或整行只有註解的不算。每一行回來時已去頭尾空白。
    """
    lines, current = [], ""
    for chunk, literal in _quoted_chunks(body):
        if literal:
            current += chunk
            continue
        head, *rest = chunk.split("\n")
        current += head
        for piece in rest:
            lines.append(current)
            current = piece
    lines.append(current)
    return [line.strip() for line in lines if line.strip()]



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



def _has_expansion(chunk: str, literal: bool) -> bool:
    """這一片有沒有 shell 會展開的寫法：單引號與 ``$'…'`` 裡、引號外反斜線跳脫的那一個字元不展開，
    雙引號裡 ``$(``、``${``、``$[``、反引號照樣展開（``"$(cat <<D)"`` 一樣開得了 heredoc），``<(``／``>(`` 只認引號外。

    找之前先拿掉接續行：雙引號裡的接續行留在字面裡（見 :func:`_literal_end`），bash 卻先接起來再展開，
    ``"$\\`` 換行 ``(…)"`` 一樣是命令替換（#595 第 2 步之後第二輪複查，bash 實測 t26）。拿掉只會把字接在一起、
    不會拆開任何一個記號（記號裡沒有反斜線），所以只可能多紅、不會少紅。"""
    if literal and chunk[:1] in ("'", "\\"):
        return False
    joined = chunk.replace("\\\n", "")
    marks = EXPANSION_MARKS if literal else EXPANSION_MARKS + PROCESS_SUBSTITUTION_MARKS
    return any(mark in joined for mark in marks)



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

    背景執行先判：字面外有單一個 ``&``（不只結尾：``判決 & 水管`` 的判決在背景跑），shell 不等它，
    離開碼一定不會被記——連抄寫員被丟到背景都一樣（收據沒人等），所以這一格排在兩種正面形狀前面。
    沒在同一行收尾的引號接著判，同樣排在正面形狀前面（見 :func:`_quoted_chunks`）。
    字面外的 ``<<``（heredoc，含 ``<<-``、``<<<`` 與算式裡的位移）再接著判：heredoc 的結束行不是命令，
    ``<<'D \\'`` 讓結束行 ``D \\`` 以反斜線收尾，斷行時會跟下一行接成一行、下一行的命令就看不到了
    （#595 第 2 步之後複查實例，bash 實測 t14）；認 heredoc 的資料與結束行等於寫一個 shell，所以整段紅。
    展開與替換（:func:`_has_expansion`）最後判，同樣排在正面形狀前面：``$(printf x)#b`` 的 ``#`` 不是註解、
    ``${x:- #b}`` 裡的也不是，``"$(cat <<D …)"`` 在雙引號裡開了 heredoc——三種複查實例都會讓這一支剝掉或
    接錯後面那段沒包的命令。
    """
    if any(BACKGROUND_RE.search(chunk) for chunk, literal in _quoted_chunks(command) if not literal):
        return (
            "有單一個 `&`，被丟到背景跑——shell 不等它，那一段的離開碼一定不會被記，"
            "包了抄寫員也一樣（抄寫員本人在背景，收據沒人等）"
        )
    if any(quote in chunk for chunk, literal in _quoted_chunks(command) if not literal for quote in "'\""):
        return (
            "有引號沒在同一行收尾——跨行的引號這一支不追（shell 會把後面幾行讀進同一個字串，"
            "藏住接在後面的命令），認不出就紅"
        )
    if any("<<" in chunk for chunk, literal in _quoted_chunks(command) if not literal):
        return (
            "有 `<<`（heredoc：把後面幾行當資料、讀到結束那一行為止）——資料行與結束行 shell 不當命令讀，"
            "結束行以反斜線收尾時，這一支會把它跟下一行接成一行、藏住下一行的命令；heredoc 這一支不追，認不出就紅"
        )
    if any(_has_expansion(chunk, literal) for chunk, literal in _quoted_chunks(command)):
        return (
            "有 `$(`、`${`、`$[`、反引號、`<(` 或 `>(`（命令替換、參數展開、舊式算術、行程替換）——裡面可以藏命令、`#` 與 heredoc，"
            "雙引號也擋不住（`\"$(…)\"` 照樣執行）；這一支不追，認不出就紅"
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

