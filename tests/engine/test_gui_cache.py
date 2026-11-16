"""畫面上游記憶的完整輸入、內容變更與讀檔錯誤控制題；檔案只住暫存目錄。"""
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from tests.engine._gui_cache import _memoize, _result_argument


class _Setting(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str


@pytest.mark.parametrize("change", (
    "result_path", "result_content", "targets_path", "targets_content",
    "capabilities", "directivity", "physics_identity",
))
def test_result_memo_tracks_complete_inputs_and_file_contents(tmp_path: Path, change: str) -> None:
    path, targets = tmp_path / "result.json", tmp_path / "targets.toml"
    path.write_bytes(b"one")
    targets.write_bytes(b"one")
    capabilities = _Setting(value="capabilities")
    directivity, identity = _Setting(value="directivity"), "phys-v1:original"
    calls: list[tuple[object, ...]] = []

    def read(path: Path, *, capabilities: object, directivity: object,
             quality_targets_path: Path, physics_identity: str) -> object:
        calls.append((path, capabilities, directivity, quality_targets_path, physics_identity))
        return object()

    remembered = _memoize(read, serializer=_result_argument)

    def invoke() -> object:
        return remembered(path, capabilities=capabilities, directivity=directivity,
                          quality_targets_path=targets, physics_identity=identity)

    before = (path, capabilities, directivity, targets, identity)
    first = invoke()
    assert invoke() is first
    assert calls == [before]
    if change == "result_path":
        path = tmp_path / "other-result.json"
        path.write_bytes(b"one")
    elif change == "result_content":
        path.write_bytes(b"two")  # 同大小也得重讀。
    elif change == "targets_path":
        targets = tmp_path / "other-targets.toml"
        targets.write_bytes(b"one")
    elif change == "targets_content":
        targets.write_bytes(b"two")
    elif change == "capabilities":
        capabilities = capabilities.model_copy(update={"value": "changed"})
    elif change == "directivity":
        directivity = directivity.model_copy(update={"value": "changed"})
    else:
        identity = "phys-v1:changed"
    second = invoke()
    assert second is not first
    assert invoke() is second
    assert calls == [before, (path, capabilities, directivity, targets, identity)]


@pytest.mark.parametrize("blocked", ("result", "targets"))
@pytest.mark.parametrize("error_type", (FileNotFoundError, PermissionError))
def test_result_memo_preserves_oserror_after_a_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked: str, error_type: type[OSError],
) -> None:
    path, targets = tmp_path / "result.json", tmp_path / "targets.toml"
    path.write_bytes(b"result")
    targets.write_bytes(b"targets")
    calls: list[Path] = []

    def read(path: Path, *, quality_targets_path: Path) -> bytes:
        calls.append(path)
        return path.read_bytes() + quality_targets_path.read_bytes()

    remembered = _memoize(read, serializer=_result_argument)
    assert remembered(path, quality_targets_path=targets) == b"resulttargets"
    assert remembered(path=path, quality_targets_path=targets) == b"resulttargets"
    inaccessible = path if blocked == "result" else targets
    failure = error_type("probe: cannot read", str(inaccessible))
    original = Path.read_bytes

    def cannot_read(actual: Path) -> bytes:
        if actual == inaccessible:
            raise failure
        return original(actual)

    monkeypatch.setattr(Path, "read_bytes", cannot_read)
    with pytest.raises(error_type) as caught:
        remembered(path, quality_targets_path=targets)
    assert caught.value is failure
    assert calls == [path]


def test_memo_does_not_hand_a_value_measured_under_other_jax_settings_to_a_later_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """同一個工人裡，先在沒掛整輪設定夾具的題算出來的身分（量到雙精度關），不准留給後面掛了夾具的題（量到雙精度開）：
    記憶的鍵要帶這個行程此刻量到的 JAX 設定。兩種設定各只真算一次，換回去拿回原來那一份。"""
    from aosr import runtime

    calls: list[tuple[str | None, bool]] = []

    def identity(*, capabilities: object) -> str:
        calls.append(runtime.jax_settings())
        return f"phys-v1:{runtime.jax_settings()}"

    remembered = _memoize(identity)
    setting = _Setting(value="capabilities")
    monkeypatch.setattr(runtime, "jax_settings", lambda **_: ("cpu", False))
    cold = remembered(capabilities=setting)
    monkeypatch.setattr(runtime, "jax_settings", lambda **_: ("cpu", True))
    configured = remembered(capabilities=setting)
    assert cold != configured and remembered(capabilities=setting) == configured
    monkeypatch.setattr(runtime, "jax_settings", lambda **_: ("cpu", False))
    assert remembered(capabilities=setting) == cold
    assert calls == [("cpu", False), ("cpu", True)]
