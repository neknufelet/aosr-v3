"""分片的身分、物理鑰匙、座標、頻率覆蓋與 hex 存讀必須完整相合。"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from aosr.config import frequency_axis
from aosr.config.capabilities import CapabilityTable, load_capabilities
from aosr.geometry.shoebox import Point, Room
from aosr.reporting import scheme as scheme_module
from tests.engine import _fem_shared_cases as cases
from tests.engine._directivity import DIRECTIVITY
from tests.engine.test_three_lane_reports_batch import operators as operators
from tests.engine import _scoring_source_model_control as control
from aosr.physics.fem_helmholtz import P2Operators

if TYPE_CHECKING:
    from aosr.reporting.fem_slices import FemShard

IDENTITY = "phys-v1:" + "0" * 64


@pytest.fixture
def small_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frequency_axis, "FEM_LANE_FREQUENCIES_HZ", cases.FREQUENCIES)


def _table() -> CapabilityTable:
    return load_capabilities(control.TARGETS.with_stem("capabilities"))


def _shards(schemes: tuple[scheme_module.Scheme, ...], slices: int) -> list['FemShard']:
    from aosr.reporting.fem_slices import solve_slice
    table = _table()
    return [solve_slice(schemes, capabilities=table, directivity=DIRECTIVITY,
                        slices=slices, slice_index=index, physics_identity=IDENTITY)
            for index in range(slices)]


@pytest.mark.parametrize("slices", (1, 2, 3, len(cases.FREQUENCIES)))
def test_slice_indices_balanced_and_complete(slices: int) -> None:
    from aosr.reporting.fem_slices import slice_indices

    segments = [slice_indices(len(cases.FREQUENCIES), slices, index) for index in range(slices)]
    assert tuple(i for segment in segments for i in segment) == tuple(range(len(cases.FREQUENCIES)))
    assert all(segments)
    assert max(map(len, segments)) - min(map(len, segments)) <= 1


@pytest.mark.parametrize("n,slices,index", ((0, 1, 0), (6, 0, 0), (6, 7, 0), (6, 2, -1), (6, 2, 2)))
def test_slice_indices_reject_empty_or_invalid(n: int, slices: int, index: int) -> None:
    from aosr.reporting.fem_slices import slice_indices

    with pytest.raises(ValueError):
        slice_indices(n, slices, index)


@pytest.mark.parametrize("field", ("room", "walls", "density", "speed", "axis"))
def test_fem_key_rejects_different_physics(small_axis: None, field: str) -> None:
    original = cases.small_scheme()
    updates: dict[str, dict[str, object]] = {
        "room": {"room_m": Room(1.7, 1.2, 1.0)},
        "walls": {"impedance_pa_s_per_m_by_wall": {
            key: value * 1.01 for key, value in original.scene.impedance_pa_s_per_m_by_wall.items()}},
        "density": {"density_kg_m3": original.scene.density_kg_m3 * 1.01},
        "speed": {"sound_speed_m_s": original.scene.sound_speed_m_s * 1.01},
        "axis": {"low_frequency_axis": frequency_axis.LowFrequencyAxis.VERIFICATION},
    }
    changed = original.model_copy(update={"scheme_id": "different", "scene":
                                          original.scene.model_copy(update=updates[field])})
    with pytest.raises(ValueError, match="鑰匙"):
        _shards((original, changed), 2)


@pytest.mark.parametrize("field", ("source_model", "scattering", "receiver", "order"))
def test_fem_key_allows_non_matrix_changes(small_axis: None, field: str) -> None:
    original = cases.small_scheme()
    changed = original.model_copy(update={"scheme_id": "different"})
    if field == "source_model":
        changed = changed.model_copy(update={"source_model": "product_default"})
    elif field == "receiver":
        points = tuple(point.model_copy(update={"position_m": (1.1, 0.6, 0.5)})
                       for point in changed.receiver_set.points)
        changed = changed.model_copy(update={"receiver_set": changed.receiver_set.model_copy(update={"points": points})})
    else:
        update = ({"scattering_by_wall": {key: 0.2 for key in original.scene.impedance_pa_s_per_m_by_wall}}
                  if field == "scattering" else {"reflection_order_k": 2})
        changed = changed.model_copy(update={"scene": changed.scene.model_copy(update=update)})
    shards = _shards((original, changed), 2)
    assert all(set(shard.candidates) == {original.scheme_id, changed.scheme_id} for shard in shards)


def test_duplicate_candidates_rejected(small_axis: None) -> None:
    original = cases.small_scheme()
    with pytest.raises(ValueError):
        _shards((original, original), 2)


@pytest.mark.parametrize("slices", (1, 2, 3, len(cases.FREQUENCIES)))
def test_shards_round_trip_and_merge_old_hex(
    small_axis: None, operators: P2Operators, tmp_path: Path, slices: int,
) -> None:
    from aosr.reporting.fem_slices import FemShard, energies_from_shards

    schemes = tuple(cases.small_scheme(name).model_copy(update={
        "speakers": dict(sources), "receiver_set": cases.small_scheme().receiver_set.model_copy(update={
            "points": tuple(point.model_copy(update={"position_m": receivers[point.receiver_id].as_tuple()})
                            for point in cases.small_scheme().receiver_set.points)})})
        for name, (sources, receivers) in cases.candidates().items())
    shards = _shards(schemes, slices)
    restored = []
    for index, shard in enumerate(shards):
        target = tmp_path / f"part-{index}"
        target.write_text(shard.model_dump_json(), encoding="utf-8")
        restored.append(FemShard.model_validate_json(target.read_text(encoding="utf-8")))
    assert restored == shards
    expected = cases.old_energies(operators, cases.candidates())
    for scheme in schemes:
        actual = energies_from_shards(list(reversed(restored)), scheme=scheme, capabilities=_table(), directivity=DIRECTIVITY, physics_identity=IDENTITY)
        assert cases.energy_hex(actual) == cases.energy_hex(expected[scheme.scheme_id])


@pytest.mark.parametrize("change", ("identity", "frequency", "key", "coordinate", "missing_candidate",
                                    "hole", "overlap", "slices", "energy", "pair", "extra", "truncated",
                                    "receiver_coordinate", "hex_overflow", "hex_invalid", "negative", "short_energy"))
def test_invalid_shards_rejected(small_axis: None, tmp_path: Path, change: str) -> None:
    from aosr.reporting.fem_slices import FemShard, ShardIdentityMismatch, energies_from_shards

    scheme = cases.small_scheme()
    shards = _shards((scheme,), 2)
    document = shards[0].model_dump(mode="json")
    if change == "identity":
        document["physics_identity"] = "other"
    elif change == "frequency":
        document["frequency_hex"][0] = float(999).hex()
    elif change == "key":
        document["fem_key"]["density_hex"] = float(999).hex()
    elif change == "coordinate":
        document["candidates"][scheme.scheme_id]["speakers"]["left"][0] = float(0.31).hex()
    elif change == "receiver_coordinate":
        document["candidates"][scheme.scheme_id]["receivers"]["front"][0] = float(1.21).hex()
    elif change == "missing_candidate":
        document["candidates"] = {"other": document["candidates"][scheme.scheme_id]}
    elif change == "hole":
        shards = shards[1:]
    elif change == "overlap":
        shards.append(shards[0])
    elif change == "slices":
        document["slices"] += 1
    elif change == "energy":
        document["candidates"][scheme.scheme_id]["energies"][0]["energy_hex"][0] = "nan"
    elif change in {"hex_overflow", "hex_invalid", "negative"}:
        replacement = {"hex_overflow": "0x1p+999999", "hex_invalid": "invalid", "negative": float(-1).hex()}[change]
        document["candidates"][scheme.scheme_id]["energies"][0]["energy_hex"][0] = replacement
    elif change == "short_energy":
        document["candidates"][scheme.scheme_id]["energies"][0]["energy_hex"].pop()
    elif change == "pair":
        document["candidates"][scheme.scheme_id]["energies"].pop()
    elif change == "extra":
        document["extra"] = "forbidden"
    error = ShardIdentityMismatch if change == "identity" else ValueError
    with pytest.raises(error):
        if change == "truncated":
            target = tmp_path / "truncated"
            target.write_text(shards[0].model_dump_json()[:-1], encoding="utf-8")
            FemShard.model_validate_json(target.read_text(encoding="utf-8"))
        else:
            if change not in {"hole", "overlap"}:
                shards[0] = FemShard.model_validate(document)
            energies_from_shards(shards, scheme=scheme, capabilities=_table(), directivity=DIRECTIVITY, physics_identity=IDENTITY)
