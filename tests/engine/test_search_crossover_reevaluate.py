"""真的重評合成小結果，正式分數逐位對細算帳；不解物理、不讀房間資料。"""
from pathlib import Path

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.physics.report_io import PathTableSection
from aosr.physics.report_source import SourceModelKind
from aosr.reporting.evaluation import reevaluate
from aosr.scoring.ranking import RankingContext, comparison_identity_of
from aosr.search.crossover_sensitivity import attach_crossover
from aosr.search.refine import RefineLedger, RefineRow
from aosr.search.refine_run import header_for
from aosr.search.sampler import Scored
from aosr.search.scoring import screening_outcome
from aosr.search.store import refine_result_name
from tests.engine._crossover_cases import prepared, protected, small_result


def test_official_self_check_uses_real_reevaluate_bit_for_bit(tmp_path: Path) -> None:
    store, registry_path, status = prepared(tmp_path)
    frequencies = tuple(20 * 2**(index / 6) for index in range(54))
    capabilities = load_capabilities(config_path("capabilities.toml"))
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    registry = load_quality_targets(registry_path)
    store.refine_ledger_path.unlink()
    book = RefineLedger.create(store.refine_ledger_path, header_for(store))
    pinned = None
    costs = {}
    for number, energies in ((None, (1, 1)), (7, (2, 1)), (9, (3, 1))):
        result = small_result(store, number, 200, energies, frequencies)
        # 路徑表只放合成的直達路徑；真重評會由保存輸入重建反射窗，沒有 FEM 求解。
        table = PathTableSection(reflection_order_k=3, frequencies_hz=frequencies,
            scattering_coefficient=tuple(0 for _ in frequencies), source_model_kind=SourceModelKind.OMNIDIRECTIONAL,
            rows=())
        result = result.model_copy(update={"pairs": tuple(p.model_copy(update={"report": p.report.model_copy(update={
            "path_table": table, "capability": p.report.capability.model_copy(update={"status": "unchecked"})})}) for p in result.pairs)})
        candidate = reevaluate(result, quality_targets_path=registry_path, capabilities=capabilities, directivity=directivity)
        if number is None:
            scheme = result.scheme
            pinned = comparison_identity_of(candidate, registry, RankingContext(purpose=scheme.purpose,
                receiver_set_fingerprint=scheme.receiver_set.fingerprint, channel_group_fingerprint=scheme.channel_group.fingerprint,
                run_date=result.run_date, engine_version=store.identity.program_fingerprint))
            assert pinned is not None
        outcome, _ = screening_outcome(candidate, result.scheme, registry=registry, run_date=result.run_date,
            engine_version=store.identity.program_fingerprint, pinned=pinned)
        assert isinstance(outcome, Scored)
        costs[number] = outcome.value.hex()
        book.append(RefineRow(round=1, trial_number=number, result_file=refine_result_name(number),
            outcome="scored", total_cost=outcome.value, seconds=0))
        store.refine_result_path(number).write_text(result.model_copy(update={"candidate": candidate}).model_dump_json())
    before = protected(store)
    summary = attach_crossover(store, status=status, quality_targets_path=registry_path)
    assert summary.completed and "自檢不等" not in summary.reason_text
    assert summary.variants and any(v.tested for v in summary.variants)
    assert {r.trial_number: r.cost_hex for r in summary.rows} == costs
    assert protected(store) == before
