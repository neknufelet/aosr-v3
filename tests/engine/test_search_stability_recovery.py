"""審查重現：上一代逐點證據跨停止、跳過與兩次 auto 都要保住。"""
from dataclasses import replace
from pathlib import Path

import pytest

from aosr.search import placement_stability_attach as module
from aosr.search.outer_status import OuterStatus
from aosr.search.placement_stability_record import read_summary, summary_path, write_summary
from tests.engine._search_run_cases import RUN_DATE
from tests.engine._stability_attach_cases import ShiftCompute, attach, ready
from tests.engine.test_search_stability_reuse import change_rows


@pytest.mark.parametrize("where", ["pinned", "prepare"])
@pytest.mark.parametrize("stop", [True, False])
def test_early_error_carries_previous_points(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                           where: str, stop: bool) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    change_rows(store)
    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt() if stop else RuntimeError("重建前段失敗原文")
    with monkeypatch.context() as patch:
        patch.setattr(module if where == "pinned" else module._Attacher,
                      "_pinned" if where == "pinned" else "prepare", interrupted)
        with pytest.raises(KeyboardInterrupt if stop else RuntimeError):
            attach(store, registry, status, ShiftCompute(store))
    saved = read_summary(store.path)
    assert saved is not None and saved.state == ("stopped" if stop else "failed")
    assert saved.points == first.points
    assert saved.arithmetic is None and saved.boundaries == ()
    compute = ShiftCompute(store)
    assert attach(store, registry, status, compute).state == "done"
    assert not compute.jobs


@pytest.mark.parametrize("kind", ["identity", "outer", "no_finalists"])
def test_skip_carries_points_then_resume_only_missing(tmp_path: Path, kind: str) -> None:
    from aosr.search.refine import RefineLedger
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    before_book = store.refine_ledger_path.read_bytes()
    skipped_status = status
    wrong = replace(store.identity, program_fingerprint="changed")
    if kind == "outer":
        skipped_status = status.model_copy(update={"outer": OuterStatus(conclusion="user_stopped")})
    if kind == "no_finalists":
        header, rows = RefineLedger.read(store.refine_ledger_path)
        store.refine_ledger_path.unlink()
        book = RefineLedger.create(store.refine_ledger_path, header)
        for row in rows:
            book.append(row.model_copy(update={"outcome": "not_comparable", "total_cost": None}))
    compute = ShiftCompute(store)
    skipped = module.attach_stability(store, status=skipped_status, quality_targets_path=registry,
        run_date=RUN_DATE, probe=lambda: wrong if kind == "identity" else store.identity,
        compute_factory=lambda root: compute)
    assert skipped.state == "skipped" and skipped.points == () and not compute.jobs
    assert skipped.retained_points == first.points
    assert skipped.total_points == skipped.computed_points == 0
    assert skipped.arithmetic is None and skipped.boundaries == ()
    assert skipped.crossover_stamp == ""  # 這題三種跳過都沒有交接摘要。
    if kind == "identity":
        assert skipped.observed_identity is not None
        assert skipped.observed_identity.program_fingerprint == wrong.program_fingerprint
    if kind != "no_finalists":
        current = module.fresh_summary(store, skipped_status)
        assert skipped.rows == current.rows and skipped.rows_fingerprint == current.rows_fingerprint
    store.refine_ledger_path.write_bytes(before_book)
    # 保存的點仍要逐點查出處；不能直接信跳過摘要。
    missing = first.points[0]
    assert missing.result_file is not None
    (store.path / missing.result_file).unlink()
    resumed = ShiftCompute(store)
    assert attach(store, registry, status, resumed).state == "done"
    assert [(j.trial_number, j.shift_name) for j in resumed.jobs] == [(missing.trial_number, missing.shift_name)]


def test_skipped_summary_is_never_reused_whole(tmp_path: Path) -> None:
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    skipped = first.model_copy(update={"state": "skipped", "arithmetic": None, "boundaries": (), "reason_text": "跳過"})
    write_summary(store.path, skipped)
    compute = ShiftCompute(store)
    second = attach(store, registry, status, compute)
    assert second.state == "done" and second.arithmetic is not None and not compute.jobs


def test_two_auto_identity_skips_preserve_all_shift_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from aosr.config.paths import config_path
    from aosr.search import cli
    store, registry, status = ready(tmp_path)
    first = attach(store, registry, status, ShiftCompute(store))
    before = {p: p.read_bytes() for p in summary_path(store.path).parent.iterdir() if p.is_file() and p.name != "summary.json"}
    monkeypatch.setattr(cli, "auto_search", lambda *a, **k: status)
    monkeypatch.setattr(cli, "attach_modal", lambda *a, **k: None)
    monkeypatch.setattr(cli, "attach_crossover", lambda *a, **k: None)
    monkeypatch.setattr(cli, "_identity", lambda *a: replace(store.identity, program_fingerprint="changed"))
    monkeypatch.setattr(cli, "config_path", lambda name: registry if name == "quality_targets.toml" else config_path(name))
    for _ in range(2):
        compute = ShiftCompute(store)
        code = cli.main(["auto", str(store.path), "--engine-commit", "test", "--modal-cache-dir", str(tmp_path / "cache")],
            compute_factory=lambda *a: lambda *b: iter(()), stability_compute_factory=lambda root: compute)
        assert code == 0 and not compute.jobs
        saved = read_summary(store.path)
        assert saved is not None and saved.state == "skipped" and saved.points == ()
        assert saved.retained_points == first.points
        assert saved.total_points == saved.computed_points == 0
        assert {p: p.read_bytes() for p in before if p.exists()} == before


def test_skip_stamps_are_best_effort_and_observation_does_not_age_summary(tmp_path: Path) -> None:
    from aosr.search import crossover_record
    from aosr.search.placement_stability_record import IdentityStamp, is_stale
    store, registry, status = ready(tmp_path)
    crossover_record.write_summary(store.path, crossover_record.fresh_summary(store, status))
    current = module.fresh_summary(store, status)
    wrong = replace(store.identity, physics_identity="changed")
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: wrong, compute_factory=lambda root: ShiftCompute(store))
    assert skipped.rows == current.rows and skipped.crossover_stamp == current.crossover_stamp
    assert not is_stale(skipped, current)
    assert not is_stale(skipped.model_copy(update={"observed_identity": IdentityStamp.of(store.identity)}), current)
    store.refine_ledger_path.unlink()
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: wrong, compute_factory=lambda root: ShiftCompute(store))
    assert skipped.rows == () and skipped.rows_fingerprint == ""
    assert skipped.crossover_stamp == current.crossover_stamp
    crossover_record.summary_path(store.path).unlink()
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: wrong, compute_factory=lambda root: ShiftCompute(store))
    assert skipped.crossover_stamp == ""
    crossover_record.summary_path(store.path).write_text("{broken")
    skipped = module.attach_stability(store, status=status, quality_targets_path=registry, run_date=RUN_DATE,
        probe=lambda: wrong, compute_factory=lambda root: ShiftCompute(store))
    assert skipped.crossover_stamp == ""
