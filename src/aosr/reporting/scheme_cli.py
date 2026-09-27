"""方案求解、存檔與多份並排比較的命令列。"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from aosr.config.capabilities import load_capabilities
from aosr.config.directivity_defaults import load_directivity_defaults
from aosr.config.paths import config_path
from aosr.config.quality_targets import load_quality_targets
from aosr.scoring.ranking_models import ComparisonIdentity, RankingResult
from aosr.scoring.contract import QualityCategory
from aosr.reporting.compare import compare_results
from aosr.reporting.pipeline import run_scheme
from aosr.reporting.result import SchemeResult, load_result, save_result
from aosr.reporting.scheme import load_scheme


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aosr.reporting.scheme_cli")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="求解一份方案並存成結果")
    run.add_argument("scheme", type=Path)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--capabilities", type=Path, required=True)
    run.add_argument("--engine-commit", required=True)
    run.add_argument("--run-date", type=date.fromisoformat)
    compare = sub.add_parser("compare", help="並排比較多份已存結果")
    compare.add_argument("results", type=Path, nargs="+")
    compare.add_argument("--capabilities", type=Path, required=True)
    compare.add_argument("--run-date", type=date.fromisoformat)
    return parser


def _print_result(result: SchemeResult, ranking: RankingResult) -> None:
    print(f"方案 {result.scheme.scheme_id}：{ranking.status_of(result.scheme.scheme_id).value}")
    costed = {line.identity.category: line.category_cost
              for row in ranking.rankable if row.candidate_id == result.scheme.scheme_id
              for line in row.categories}
    for evaluation in result.candidate.evaluations:
        cost = costed.get(evaluation.category)
        print(f"  {evaluation.category.value} | {evaluation.state.value} | "
              f"代價 {cost if cost is not None else '未計'} | "
              f"原因 {','.join(reason.value for reason in evaluation.reason_codes) or '無'}")
    print("低頻拖尾：尚未評估")


def _run(args: argparse.Namespace) -> int:
    table = load_capabilities(args.capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    target_path = config_path("quality_targets.toml")
    run_date = args.run_date or date.today()
    result = run_scheme(load_scheme(args.scheme), capabilities=table,
                        directivity=directivity, quality_targets_path=target_path,
                        engine_commit=args.engine_commit, run_date=run_date)
    save_result(result, args.out)
    timing = result.timings
    print(f"秒數：求解 {timing.solve_s:.3f}，輸出與反射 {timing.output_s:.3f}，"
          f"評估 {timing.evaluate_s:.3f}，全程 {timing.total_s:.3f}")
    ranking = compare_results([result], quality_targets=load_quality_targets(target_path),
                              run_date=run_date)
    _print_result(result, ranking)
    return 0


def _comparison_table(results: list[SchemeResult], ranking: RankingResult) -> None:
    """同一類橫向列各份代價；不能同表的格子明印狀態。"""
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


def _identity_difference(ranking: RankingResult, identity: Sequence[ComparisonIdentity]) -> str:
    """不能同表的原因：主表有、這一份沒有；這一份有、主表沒有；同一類但身分不同。"""
    main = {part.category: part for part in ranking.header.main_table_identity}
    mine = {part.category: part for part in identity}
    groups = (
        ("少了", [category for category in main if category not in mine]),
        ("多了", [category for category in mine if category not in main]),
        ("同一類但身分不同", [category for category in mine
                             if category in main and mine[category] != main[category]]),
    )
    detail = "；".join(f"{label} {','.join(category.value for category in categories)}"
                      for label, categories in groups if categories)
    return "與主表的比較身分不同" + (f"：{detail}" if detail else "")


def _ranking_reason(ranking: RankingResult, candidate_id: str) -> str:
    """逐份狀態的原因取自排名結果所屬的列。"""
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
            return _identity_difference(ranking, incompatible_row.identity)
    return "無"


def _compare(args: argparse.Namespace) -> int:
    table = load_capabilities(args.capabilities)
    directivity = load_directivity_defaults(config_path("directivity_defaults.toml"))
    target_path = config_path("quality_targets.toml")
    results = [load_result(path, capabilities=table, directivity=directivity,
                           quality_targets_path=target_path)
               for path in args.results]
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


def main(argv: list[str] | None = None) -> int:
    """選擇 run 或 compare；日期只在此層從時鐘取得。"""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run(args)
    if len(args.results) < 2:
        parser.error("compare 至少需要兩份結果")
    return _compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
