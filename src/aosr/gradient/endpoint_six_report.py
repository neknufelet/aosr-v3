"""六類終點三組數的人看段落；不求值、不改任何結果。"""
from __future__ import annotations

from aosr.gradient.endpoint_six import EndpointSix, SixComparison

_LABELS = ('音色平衡', '聆聽區穩定性', '反射與回聲', '殘響', '聲道匹配', '低頻響應（產品頻帶、無代理）')


def _number(value: float | None) -> str:
    return '不可估' if value is None else f'{value:.12g}'


def _comparison(label: str, value: SixComparison, search: str, product: str) -> list[str]:
    lines = [f'  {label}替身誤差：J（含代理、低頻 {search} 的搜尋目標），'
        f'替身 {_number(value.reduced.objective_total)}、完整 {_number(value.full.objective_total)}；'
        f'完整減替身 {_number(value.surrogate_difference)}',
        f'  {label}實作差：product_total（完整聲壓的無代理產品總分、低頻 {product}），'
        f'JAX（雙精度評分）{_number(value.full.product_total)}、numpy（正式六類評分）{_number(value.numpy_total)}；'
        f'JAX 減 numpy {_number(value.implementation_difference)}',
        f'  {label}逐類表：類別｜替身產品代價｜完整產品代價｜numpy（正式評分）代價｜JAX 減 numpy']
    for name, reduced, full, reference, difference in zip(_LABELS, value.reduced.categories, value.full.categories,
                                             value.numpy_categories, value.implementation_categories, strict=True):
        lines.append(f'    {name}｜{_number(reduced)}｜{_number(full)}｜{_number(reference)}｜{_number(difference)}')
    if value.numpy_unavailable_reason is not None:
        lines.append(f'  {label}numpy（正式評分）不可估：{value.numpy_unavailable_reason}')
    for name, scores in (('替身', value.reduced), ('完整', value.full)):
        lines.append(f'  {label}{name}：proxy（大峰代理）{_number(scores.proxy_db)} dB；'
            f'低頻搜尋目標 {_number(scores.low_frequency_objective)}；J（六類搜尋目標）{_number(scores.objective_total)}；'
            f'product_total（無代理產品合分）{_number(scores.product_total)}')
    notes = value.crossover
    lines.append(f'  {label}：f_s（Schroeder 交接頻率，Hz）{_number(notes.schroeder_frequency_hz)}；'
        f'capped_by_upper_limit（交接超過上限而截斷）{notes.capped_by_upper_limit}；'
        f'殘響頻帶數 {notes.reverberation_band_count}')
    return lines


def six_endpoint_lines(value: EndpointSix) -> list[str]:
    """基準與終點各列三組數；改善只比較同一定義，正數代表終點較低。"""
    search = '～'.join(f'{f:g}' for f in value.search_range_hz) + ' Hz'
    product = '～'.join(f'{f:g}' for f in value.product_range_hz) + ' Hz'
    lines = [*_comparison('終點', value.endpoint, search, product), *_comparison('原牆基準', value.baseline, search, product)]
    gain = value.improvement
    for name, scores in (('替身', gain.reduced), ('完整', gain.full)):
        lines.append(f'  改善（原牆減終點，{name}）：J（六類搜尋目標）{_number(scores.objective_total)}；'
            f'product_total（無代理產品合分）{_number(scores.product_total)}；proxy（大峰代理）{_number(scores.proxy_db)} dB')
        lines.extend(f'    {label}改善 {_number(cost)}' for label, cost in zip(_LABELS, scores.categories, strict=True))
    lines.append(f'  numpy（正式評分）產品總分改善 {_number(gain.numpy_total)}')
    lines.extend(f'    {label}改善 {_number(cost)}' for label, cost in zip(_LABELS, gain.numpy_categories, strict=True))
    return lines
