"""方案求解、存檔與多份並排比較的命令列。

計算中程式或物理身分改變時不寫結果，以 PROGRAM_CHANGED_EXIT 與固定標記回報。
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from aosr.reporting.calculation_fingerprint import calculation_fingerprint, short_fingerprint

PROGRAM_CHANGED_EXIT = 3
PROGRAM_CHANGED_MARKER = "AOSR_IDENTITY_CHANGED"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.reporting.scheme_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="求解一份方案並存成結果")
    run.add_argument("scheme", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--capabilities", type=Path, required=True)
    run.add_argument("--engine-commit", required=True)
    run.add_argument("--run-date", type=date.fromisoformat)
    run.add_argument("--search-id")
    run.add_argument("--trial-number", type=int)
    run.add_argument("--fem-parts", type=Path, nargs="+")
    sliced = sub.add_parser("fem-slice", help="批次求解一段有限元素頻率並原子存檔")
    sliced.add_argument("schemes", type=Path, nargs="+")
    sliced.add_argument("--slice", dest="slice_index", type=int, required=True)
    sliced.add_argument("--slices", type=int, required=True)
    sliced.add_argument("--out", type=Path, required=True)
    sliced.add_argument("--capabilities", type=Path, required=True)
    sliced.add_argument("--engine-commit", required=True)
    compare = sub.add_parser("compare", help="並排比較多份已存結果")
    compare.add_argument("results", type=Path, nargs="+")
    compare.add_argument("--capabilities", type=Path, required=True)
    compare.add_argument("--run-date", type=date.fromisoformat)
    identity = sub.add_parser("identity", help="印物理身分與整支程式指紋")
    identity.add_argument("--capabilities", type=Path, required=True)
    return parser


def _print_result(result: SchemeResult, ranking: RankingResult) -> None:
    from aosr.reporting.display import LOW_FREQUENCY_DECAY_NOTE
    print(f"方案 {result.scheme.scheme_id}：{ranking.status_of(result.scheme.scheme_id).value}")
    costed = {line.identity.category: line.category_cost
              for row in ranking.rankable if row.candidate_id == result.scheme.scheme_id
              for line in row.categories}
    for evaluation in result.candidate.evaluations:
        cost = costed.get(evaluation.category)
        print(f"  {evaluation.category.value} | {evaluation.state.value} | "
              f"代價 {cost if cost is not None else '未計'} | "
              f"原因 {','.join(reason.value for reason in evaluation.reason_codes) or '無'}")
    print(LOW_FREQUENCY_DECAY_NOTE)


def _calculation_start(capabilities: Path) -> tuple[str, CapabilityTable, DirectivityDefaults, str]:
    """指紋先於計算程式載入，物理身分取自開跑時讀到的設定。"""
    # 先量指紋才載入計算程式；順序顛倒會讓載入後、量指紋前的修改變成「舊程式算、新指紋」。
    # 載入之後的修改由寫檔前第二次量測攔下。
    before = calculation_fingerprint(capabilities_path=capabilities)
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.reporting.physics_identity import physics_identity
    table = load_capabilities(capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    physics_before = physics_identity(capabilities=table, directivity=directivity)
    return before, table, directivity, physics_before


def _changed(reason: str) -> int:
    """物理或整支程式身分不同，一律留下同一行機器標記。"""
    sys.stderr.write(PROGRAM_CHANGED_MARKER + "\n" + reason + "\n")
    return PROGRAM_CHANGED_EXIT


def _identities_changed(capabilities: Path, before: str, physics_before: str) -> bool:
    """寫檔前重讀設定、重量兩種身分；先完成兩次量測才判斷。"""
    after = calculation_fingerprint(capabilities_path=capabilities)
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.reporting.physics_identity import physics_identity
    physics_after = physics_identity(capabilities=load_capabilities(capabilities),
                                     directivity=load_directivity_defaults(
                                         config_path("directivity_defaults.toml")))
    if physics_before != physics_after:
        _changed("計算中物理計算的程式或設定被改了，這一跑不算，請重算物理")
        return True
    if before != after:
        _changed(f"計算中程式或設定被改了（開跑 {short_fingerprint(before)}／"
                 f"寫檔前 {short_fingerprint(after)}），這一跑不算，請重算")
        return True
    return False


def _read_fem_parts(paths: list[Path]) -> list[FemShard]:
    """讀外層包裝，只把 shard 交給物理模型驗證；資源統計不參與物理身分。"""
    from aosr.reporting.fem_slices import FemShard

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"分片 JSON 有重複鍵：{key}")
            result[key] = value
        return result

    shards = []
    for path in paths:
        envelope = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        if not isinstance(envelope, dict) or "shard" not in envelope:
            raise ValueError(f"分片檔 {path} 缺少 shard 包裝欄位")
        shards.append(FemShard.model_validate(envelope["shard"]))
    return shards


def _run(args: argparse.Namespace) -> int:
    before, table, directivity, physics_before = _calculation_start(args.capabilities)
    from aosr.config.paths import config_path
    from aosr.config.quality_targets import load_quality_targets
    from aosr.reporting import fem_slices
    from aosr.reporting.compare import compare_results
    from aosr.reporting.pipeline import run_scheme
    from aosr.reporting.result import save_result
    from aosr.reporting.scheme import load_scheme
    target_path = config_path("quality_targets.toml")
    run_date = args.run_date or date.today()
    scheme = load_scheme(args.scheme)
    energies = None
    if args.fem_parts is not None:
        try:
            energies = fem_slices.energies_from_shards(
                _read_fem_parts(args.fem_parts), scheme=scheme, capabilities=table,
                directivity=directivity, physics_identity=physics_before)
        except fem_slices.ShardIdentityMismatch as exc:
            return _changed(str(exc))
    result = run_scheme(scheme, capabilities=table,
                        directivity=directivity, quality_targets_path=target_path,
                        engine_commit=args.engine_commit, program_fingerprint=before,
                        physics_identity=physics_before,
                        run_date=run_date, origin=_origin(args), fem_energies=energies)
    if _identities_changed(args.capabilities, before, physics_before):
        return PROGRAM_CHANGED_EXIT
    save_result(result, args.out)
    timing = result.timings
    print(f"秒數：求解 {timing.solve_s:.3f}，輸出與反射 {timing.output_s:.3f}，"
          f"評估 {timing.evaluate_s:.3f}，全程 {timing.total_s:.3f}")
    ranking = compare_results([result], quality_targets=load_quality_targets(target_path),
                              run_date=run_date)
    _print_result(result, ranking)
    return 0


def _write_fem_part(shard: FemShard, out: Path, wall_s: float) -> None:
    """原子保存 {"shard": FemShard JSON, "wall_s": 秒數, "max_rss_kib": 峰值記憶體}。

    wall_s 是命令開始到寫檔前的牆鐘秒；max_rss_kib 取本行程 RUSAGE_SELF。
    暫存檔和 OUT 同目錄，完整關檔後才 os.replace；可處理的失敗會清理暫存檔。
    """
    envelope = {"shard": shard.model_dump(mode="json"), "wall_s": wall_s,
                "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=out.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(envelope, ensure_ascii=False))
        temporary.replace(out)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fem_slice(args: argparse.Namespace) -> int:
    """批次解一段頻率，身份穩定才原子寫包裝；成功不印字。"""
    start = time.perf_counter()
    before, table, directivity, physics_before = _calculation_start(args.capabilities)
    from aosr.reporting import fem_slices
    from aosr.reporting.scheme import load_scheme
    shard = fem_slices.solve_slice(
        [load_scheme(path) for path in args.schemes], capabilities=table,
        directivity=directivity, slices=args.slices, slice_index=args.slice_index,
        physics_identity=physics_before)
    if _identities_changed(args.capabilities, before, physics_before):
        return PROGRAM_CHANGED_EXIT
    _write_fem_part(shard, args.out, time.perf_counter() - start)
    return 0


def _comparison_table(results: list[SchemeResult], ranking: RankingResult) -> None:
    """同一類橫向列各份代價；不能同表的格子明印狀態。"""
    from aosr.scoring.contract import QualityCategory
    costs = {row.candidate_id: {line.identity.category: line.category_cost
                                for line in row.categories}
             for row in ranking.rankable}
    print("類別 | " + " | ".join(item.scheme.scheme_id for item in results))
    for category in QualityCategory:
        if category is QualityCategory.LOW_FREQUENCY_DECAY:
            continue
        cells = []
        for item in results:
            candidate_id = item.scheme.scheme_id
            if candidate_id not in costs:
                cells.append(ranking.status_of(candidate_id).value)
                continue
            evaluation = next((part for part in item.candidate.evaluations
                               if part.category is category), None)
            if evaluation is None:
                cells.append("未評估")
            elif category in costs[candidate_id]:
                cells.append(str(costs[candidate_id][category]))
            else:
                reasons = ",".join(reason.value for reason in evaluation.reason_codes) or "無"
                cells.append(f"{evaluation.state.value}：{reasons}")
        print(f"{category.value} | " + " | ".join(cells))


def _ranking_reason(ranking: RankingResult, candidate_id: str) -> str:
    """逐份狀態的原因取自排名結果所屬的列。"""
    from aosr.reporting.compare import identity_difference
    for missing_row in ranking.not_evaluated:
        if missing_row.candidate_id == candidate_id:
            return ",".join(f"{part.category.value}:{part.reason.value}" + (
                f"({','.join(reason.value for reason in part.evaluator_reason_codes)})"
                if part.evaluator_reason_codes else "") for part in missing_row.missing)
    for eliminated_row in ranking.eliminated:
        if eliminated_row.candidate_id == candidate_id:
            reasons = [reason.value for reason in eliminated_row.reasons]
            reasons.extend(f"{part.category.value}:{part.reason.value}"
                           for part in eliminated_row.missing)
            return ",".join(reasons)
    for incompatible_row in ranking.not_comparable.rows:
        if incompatible_row.candidate_id == candidate_id:
            return identity_difference(ranking, incompatible_row.identity)
    return "無"


def _compare(args: argparse.Namespace) -> int:
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path
    from aosr.config.quality_targets import load_quality_targets
    from aosr.reporting.physics_identity import physics_identity
    from aosr.reporting.compare import compare_results
    from aosr.reporting.evaluation import load_result
    table = load_capabilities(args.capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    target_path = config_path("quality_targets.toml")
    physics = physics_identity(capabilities=table, directivity=directivity)
    loaded = [load_result(path, capabilities=table, directivity=directivity,
                          quality_targets_path=target_path, physics_identity=physics) for path in args.results]
    results = [item.result for item in loaded]
    for path, item in zip(args.results, loaded, strict=True):
        print(f"{path.name} | 讀回等級 {item.standing.value}")
    ranking = compare_results(results,
                              quality_targets=load_quality_targets(target_path),
                              run_date=args.run_date or date.today())
    first_fingerprint = results[0].scheme.receiver_set.fingerprint
    for path, result in zip(args.results, results, strict=True):
        fingerprint = result.scheme.receiver_set.fingerprint
        relation = "相同" if fingerprint == first_fingerprint else "不同"
        status = ranking.status_of(result.scheme.scheme_id)
        reason = _ranking_reason(ranking, result.scheme.scheme_id)
        print(f"{path.name} | 座位組指紋 {fingerprint} | 與第一份{relation} | "
              f"{status.value} | 原因 {reason}")
    for result in results:
        _print_result(result, ranking)
    _comparison_table(results, ranking)
    return 0


def _identity(args: argparse.Namespace) -> int:
    """只讀設定與靜態原始碼，不求解或評分。"""
    from aosr.config.capabilities import load_capabilities
    from aosr.config.directivity_defaults import load_directivity_defaults
    from aosr.config.paths import config_path

    from aosr.reporting.physics_identity import physics_identity

    table = load_capabilities(args.capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    print(f"物理身分 {physics_identity(capabilities=table, directivity=directivity)}")
    print(f"整支程式指紋 {calculation_fingerprint(capabilities_path=args.capabilities)}")
    return 0


def _origin(args: argparse.Namespace) -> ResultOrigin:
    """搜尋代號單獨指定代表原方案；有試算編號才是候選。"""
    from aosr.reporting.result import ResultOrigin

    kind = "run" if args.search_id is None else "search_baseline" if args.trial_number is None else "search_candidate"
    return ResultOrigin.model_validate({"kind": kind, "search_id": args.search_id,
                                        "trial_number": args.trial_number})


def main(argv: list[str] | None = None) -> int:
    """選擇 run 或 compare；日期只在此層從時鐘取得。亦可印 identity 物理身分。

    fem-slice 存批次分片；run --fem-parts 讀包裝注入，拒收時不退回有限元素求解。
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.trial_number is not None and (args.search_id is None or args.trial_number < 0):
            parser.error("trial_number 必須非負且同時提供 search_id")
        if args.search_id is not None and not args.search_id.strip():
            parser.error("search_id 不可空白")
        return _execute_calculation(args)
    if args.command == "fem-slice":
        return _execute_calculation(args)
    if args.command == "identity":
        return _identity(args)
    if len(args.results) < 2:
        parser.error("compare 至少需要兩份結果")
    return _compare(args)


def _execute_calculation(args: argparse.Namespace) -> int:
    """輸入、分片或檔案錯誤留下原因並非零離開。"""
    if args.command == "run" and args.fem_parts is None:
        return _run(args)
    try:
        return _fem_slice(args) if args.command == "fem-slice" else _run(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if TYPE_CHECKING:
    from aosr.config.capabilities import CapabilityTable
    from aosr.config.directivity_defaults import DirectivityDefaults
    from aosr.reporting.fem_slices import FemShard
    from aosr.reporting.result import ResultOrigin, SchemeResult
    from aosr.scoring.ranking_models import RankingResult


if __name__ == "__main__":
    raise SystemExit(main())
