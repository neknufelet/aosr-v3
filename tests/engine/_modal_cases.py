"""秒級模態考卷的明列資料；沒有求解、真實資料夾或環境依賴。"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from aosr.config.paths import config_path
from aosr.geometry.shoebox import Wall
from aosr.physics.fem_modal import FemModalSpectrum, FemMode
from aosr.physics.fem_modal_check import FemModalCheck
from aosr.physics.modal_convention import ModalKind
from aosr.reporting.modal_diagnosis import modal_identity, room_layer_from_spectrum
from aosr.reporting.modal_diagnosis_cache import CachedRoom
from aosr.reporting.modal_diagnosis_model import ModalDiagnosis, ModalDiagnosisState, ModalKey, PlacementLayer, PlacementPair
from aosr.reporting.scheme import Scheme, load_scheme


def scheme() -> Scheme:
    return load_scheme(config_path("capabilities.toml").parents[4] / "blueprint" / "scheme_reference_room.json")


def sample(value: Scheme | None = None) -> tuple[ModalDiagnosis, CachedRoom]:
    value = value or scheme()
    scene = value.scene
    key = ModalKey.from_inputs(room=scene.room_m,
        wall_impedances={Wall.from_name(w): z for w, z in scene.impedance_pa_s_per_m_by_wall.items()},
        density_kg_m3=scene.density_kg_m3, sound_speed_m_s=scene.sound_speed_m_s)
    # 很寬的峰寬讓它們連成一組；兩端在參考線範圍外，擺位大小順序跟頻率順序不同。
    modes = tuple(FemMode(omega=complex(2 * math.pi * f, 900), raw_omega=complex(2 * math.pi * f, 900),
        frequency_hz=f, t60_s=3 * math.log(10) / 900, q=math.pi * f / 900,
        kind=ModalKind.RESONANCE, residual=0, linearized_residual=0, shift_index=0,
        shift_hz=100, shape=np.ones(2, dtype=np.complex128)) for f in (25.0, 100.0, 108.0, 270.0))
    check = FemModalCheck(0, math.inf, (), 0, 0, 0, 0, 0, "明列資料，未作完備性保證")
    spectrum = FemModalSpectrum(modes, (), 0, 0, 2, check)
    room = room_layer_from_spectrum(spectrum, key=key, identity=modal_identity(),
                                    mesh_sha256="a" * 64, solve_seconds=0)
    pairs = tuple(PlacementPair(speaker_id=s, speaker_position_m=p.as_tuple(),
        receiver_id=r.receiver_id, receiver_position_m=r.position_m,
        resonance_indices=tuple(range(len(modes))), resonance_magnitude=(1, 4, 2, 0),
        group_magnitude=(5, 7, 6, 3)) for s, p in value.speakers.items() for r in value.receiver_set.points)
    diagnosis = ModalDiagnosis(state=ModalDiagnosisState.DIAGNOSED_NOT_SCORED, key=key,
        modal_identity=room.modal_identity, room_layer=room, placement_layer=PlacementLayer(pairs=pairs))
    return diagnosis, CachedRoom(room, spectrum)


def runner(folder: Path, diagnosis: ModalDiagnosis, *, wait: bool = False, stderr: str = "",
           cache_miss: bool = False) -> tuple[str, ...]:
    """子行程只複製明列診斷；可由考卷放行，並記住參數。"""
    import sys
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "modal-fixture.json"
    source.write_text(diagnosis.model_dump_json(), encoding="utf-8")
    script = folder / "modal-runner.py"
    script.write_text("import sys,time,shutil,json\nfrom pathlib import Path\n"
        f"sys.stderr.write({stderr!r}); sys.stderr.flush()\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.with_suffix('.args.json').write_text(json.dumps(sys.argv))\n"
        f"Path({str(folder / 'modal-args.json')!r}).write_text(json.dumps(sys.argv))\n"
        f"while {wait!r} and not Path({str(folder / 'modal-release')!r}).exists():\n    time.sleep(0.02)\n"
        f"if {cache_miss!r} and '--cache-only' in sys.argv:\n"
        "    out.write_text(json.dumps({'state': 'not_computed'}))\n"
        f"else:\n    shutil.copyfile({str(source)!r}, out)\n", encoding="utf-8")
    return sys.executable, str(script)
