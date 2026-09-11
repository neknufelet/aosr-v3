"""舊家 ``test_config.py`` 的 perceptual 那一段帶過來，只改 import 與設定檔路徑。

來源：上一代那一棵樹的 config 考卷（整檔 18 題；不在這個 repo 裡）。帶過來的是**只綁這一塊**那一段
（perceptual 的 5 支考卷裡最主要的一支）；下面兩支**不帶**：

* ``test_experiment_builds_third_octave_axis_in_range``——它讀 ``experiment.yaml``、
  用 ``load_experiment``（``experiment_schema``）與 ``receiver_grid``。``experiment_schema``
  搬去材料那一塊（#134），這一刀沒有它。
* ``test_materials_load_into_registry``——同上，再加上 ``load_materials``（``material_loader``，
  也搬去 #134）。

**這兩支去了哪裡**：跟著 #134 走（那兩支要驗的是 experiment 那份 schema 與材料註冊表，
不是 perceptual／physics 的值）；這一刀不假裝它們還在。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from aosr.config.paths import config_path
from aosr.config.perceptual import (
    ModalDecayCurve,
    PerceptualProfilesConfig,
    RfzProfileConfig,
    load_material_rfz_profile,
    load_perceptual,
)
from aosr.config.physics_constants import load_physics_constants


CONFIG = config_path("perceptual.toml").parent


def test_physics_constants_and_derived_rho_c() -> None:
    pc = load_physics_constants(CONFIG / "physics_constants.toml")
    assert pc.air_density_kg_m3 == 1.2
    assert pc.sound_speed_m_s == 343.0
    assert pc.air_viscosity_pa_s == pytest.approx(1.81e-5)
    assert pc.rho_c == pytest.approx(1.2 * 343.0)


def test_physics_constants_are_frozen() -> None:
    pc = load_physics_constants(CONFIG / "physics_constants.toml")
    with pytest.raises(ValidationError):  # pydantic frozen ⇒ ValidationError on set
        pc.sound_speed_m_s = 340.0


def test_perceptual_block() -> None:
    pcfg = load_perceptual(CONFIG / "perceptual.toml")
    assert pcfg.active_profile == "monitoring"
    assert pcfg.target_curve.slope_db_oct == -0.7
    assert pcfg.rfz.window_ms == 15.0
    assert pcfg.rfz.window_mode is None
    assert pcfg.rfz.threshold_db == -10.0
    assert pcfg.rfz.role == "hard_gate"
    assert pcfg.modal_decay.hz == (32.0, 63.0, 100.0, 150.0, 200.0)
    assert pcfg.modal_decay.t60_s == (0.90, 0.50, 0.35, 0.25, 0.20)
    assert pcfg.rt60.model == "itu_bs1116_3"
    assert pcfg.itdg.target_ms is None
    # P5.4 D6 — [early_reflection] rides PerceptualConfig as a required field
    assert pcfg.early_reflection.window_ms == 15.0
    assert pcfg.early_reflection.required_attenuation_db == 10.0
    assert pcfg.early_reflection.check_band_hz == (200.0, 8000.0)


@pytest.mark.parametrize(
    ("profile", "threshold_db", "window_ms", "window_mode", "role", "t60_s"),
    [
        ("critical", -25.0, None, "dynamic", "soft_target", (0.90, 0.30, 0.28, 0.18, 0.18)),
        ("monitoring", -10.0, 15.0, None, "hard_gate", (0.90, 0.50, 0.35, 0.25, 0.20)),
        ("home_theater", -10.0, 15.0, None, "hard_gate", (0.90, 0.50, 0.35, 0.25, 0.20)),
    ],
)
def test_perceptual_profile_selection(
    profile: str,
    threshold_db: float,
    window_ms: float | None,
    window_mode: str | None,
    role: str,
    t60_s: tuple[float, ...],
) -> None:
    pcfg = load_perceptual(CONFIG / "perceptual.toml", profile=profile)
    assert pcfg.active_profile == profile
    assert pcfg.rfz.threshold_db == threshold_db
    assert pcfg.rfz.window_ms == window_ms
    assert pcfg.rfz.window_mode == window_mode
    assert pcfg.rfz.role == role
    assert pcfg.modal_decay.t60_s == t60_s


def test_perceptual_default_is_monitoring() -> None:
    path = CONFIG / "perceptual.toml"
    assert load_perceptual(path) == load_perceptual(path, profile="monitoring")


@pytest.mark.parametrize("profile", ["unknown", ""])
def test_perceptual_unknown_profile_lists_available_profiles(profile: str) -> None:
    with pytest.raises(
        ValueError,
        match=r"available profiles: critical, home_theater, material_rfz, monitoring",
    ):
        load_perceptual(CONFIG / "perceptual.toml", profile=profile)


@pytest.mark.parametrize(
    "rfz",
    [
        {"threshold_db": -10.0, "role": "hard_gate"},
        {
            "threshold_db": -10.0,
            "role": "hard_gate",
            "window_ms": 15.0,
            "window_mode": "dynamic",
        },
    ],
)
def test_rfz_profile_requires_exactly_one_window(rfz: dict[str, object]) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        RfzProfileConfig.model_validate(rfz)


def test_modal_decay_curve_requires_equal_axis_lengths() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        ModalDecayCurve(hz=(32.0, 63.0), t60_s=(0.9,))


def test_profile_modal_decay_level_must_exist_in_curves() -> None:
    source = _valid_perceptual_source()
    source["profiles"] = {
        "monitoring": {
            "rfz": {"threshold_db": -10.0, "role": "hard_gate", "window_ms": 15.0},
            "modal_decay": {"level": "missing"},
        }
    }
    with pytest.raises(ValidationError, match="missing from modal_decay.curves"):
        PerceptualProfilesConfig.model_validate(source)


def test_default_profile_must_exist() -> None:
    source = _valid_perceptual_source()
    source["default_profile"] = "missing"
    with pytest.raises(ValidationError, match="default_profile"):
        PerceptualProfilesConfig.model_validate(source)


def test_perceptual_toml_forbids_extra_key(tmp_path: Path) -> None:
    text = (CONFIG / "perceptual.toml").read_text()
    path = tmp_path / "perceptual.toml"
    path.write_text(
        text.replace(
            'default_profile = "monitoring"',
            'default_profile = "monitoring"\nbogus = true',
        )
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        load_perceptual(path)


def test_perceptual_early_reflection_type_is_the_new_home_one() -> None:
    """``EarlyReflectionConfig`` 是**新家這一層**那個型別，不是上一代 zoning 那一支的。

    這一刀把型別沉到第 2 層（``aosr.config.early_reflection``），``perceptual`` 從本層拿。
    這一條把「又宣告了第二份」釘住：載入器吐出來的物件，它的類別必須**就是**那一支裡的
    那一個物件（不是同名同形的另一個）。
    """
    from aosr.config.early_reflection import EarlyReflectionConfig

    pcfg = load_perceptual(CONFIG / "perceptual.toml")
    assert type(pcfg.early_reflection) is EarlyReflectionConfig
    assert "aosr.config.early_reflection" == EarlyReflectionConfig.__module__
    zoning_module = "lib.zoning.config"
    assert zoning_module not in sys.modules, "考卷自己把上一代 zoning 那一支拉進來了"


def test_load_perceptual_requires_a_path_like_the_previous_generation() -> None:
    """**不傳路徑要丟 ``TypeError``**——上一代就是這樣（獨立驗證量到的行為差異，修 A）。

    這一條原本寫的是「不傳路徑就走套件預設」，那是我這一刀自己加的行為（上一代
    `load_perceptual(path, *, profile=None)` 的 `path` 是必填）。票的合約是「行為與數值
    跟上一代一致」，所以那一條被換成這一條：把「`None` 是 `TypeError`」釘住。
    """
    with pytest.raises(TypeError):
        load_perceptual()  # type: ignore[call-arg]  # expires=2026-12-08 reason=這一條要驗的就是「少給必填參數會炸」，所以刻意少給
    with pytest.raises(TypeError):
        load_material_rfz_profile()  # type: ignore[call-arg]  # expires=2026-12-08 reason=同上


def _valid_perceptual_source() -> dict[str, object]:
    return {
        "default_profile": "monitoring",
        "target_curve": {"slope_db_oct": -0.7},
        "rt60": {
            "model": "itu_bs1116_3",
            "tol_s": 0.1,
            "target": {
                "nominal_coeff": 0.25,
                "volume_ref_m3": 100.0,
                "hz": (63.0, 200.0, 8000.0),
                "tol_hi_s": (0.30, 0.05, 0.10),
                "tol_lo_s": (0.05, 0.05, 0.10),
            },
        },
        "jnd": {
            "edt_pct": 5.0,
            "c80_db": 0.9,
            "c50_db": 1.1,
            "d50": 0.05,
            "g_db": 1.0,
            "sti": 0.03,
        },
        "modal_decay": {
            "curves": {
                "music": {"hz": (32.0,), "t60_s": (0.9,)},
            }
        },
        "itdg": {},
        "early_reflection": {
            "window_ms": 15.0,
            "required_attenuation_db": 10.0,
            "check_band_hz": (200.0, 8000.0),
        },
        "profiles": {
            "monitoring": {
                "rfz": {
                    "threshold_db": -10.0,
                    "role": "hard_gate",
                    "window_ms": 15.0,
                },
                "modal_decay": {"level": "music"},
            }
        },
    }
