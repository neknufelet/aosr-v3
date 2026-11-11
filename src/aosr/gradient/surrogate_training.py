"""任意切法的替身訓練材料與留出材料（#1098）：網頁替新切法建替身時用，全走產品函式。

照 #990 r4 與 #561 分區示範房那兩次的做法推廣到任意面片清單（面片序照 ``scheme_patch_keys``）。每一片三種之一：
- 原牆（original）：方案那一片「貼上去之前」的牆。方案存的是貼在原牆上的構造（``MountedOnWall``）就取它記的原牆，
  其他材料本身就是原牆；
- 方案存檔（saved）：方案那一片現在存的材料，原樣（固定區放的那一組材料就在這裡）；
- 貼某家族的樣板（mounted）：家族與樣板照搜尋設定 ``[[family]]`` 的 ``template_preset``，厚度、空腔在搜尋的盒界內，
  總深度照家族上限切（:func:`depth_room`）；背後接那一片的原牆（``on_wall``）或剛背（``rigid``），一組一種。
阻抗路徑跟終點同一條：``run.designs_for`` → θ → ``endpoint.materials_from_surfaces`` → ``endpoint.material_impedances``。

訓練：全部原牆、方案存檔（跟全部原牆不同時）、每個家族的角點（每一片同一個角點）、一組混搭、設定檔那幾組隨機。
留出：全部原牆、方案存檔（跟全部原牆不同時）、一組厚度與空腔都取中間值的混搭、另一個種子的隨機組。
留出只拿來驗、不拿來建；替身准不准用照舊由精度契約那兩把尺判（:mod:`aosr.physics.fem_reduced_check`）。

一份切法換材料共用同一份替身（``aosr.gui.surrogate_slots``）：原牆與方案存檔取的是觸發建模那一份方案的材料，
別份同切法的方案換了固定材料，用的仍是這份替身；終點用完整模型重算，最後的數字不靠替身。
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from aosr.config.material_gradient import MOUNT_ON_WALL, MOUNT_RIGID, MaterialGradientSettings, Mount
from aosr.config.surrogate_training import SurrogateTrainingSettings
from aosr.gradient import run
from aosr.gradient.boundary import BoundaryKey, surface_id
from aosr.gradient.endpoint import material_impedances, materials_from_surfaces
from aosr.gradient.material_gradient import SurfaceDesign, checked_theta, depth_limits, depth_room
from aosr.materials.construction_models import MountedOnWall, SurfaceMaterial
from aosr.config.physics_constants import PhysicsConstants
from aosr.config.porous_model_ranges import PorousModelRanges
from aosr.physics.fem_reduced_boundary import FrequencyBoundaryImpedances

FaceKind = Literal["original", "saved", "mounted"]


@dataclass(frozen=True)
class FaceSpec:
    kind: FaceKind
    family: str | None = None
    thickness_m: float | None = None
    cavity_m: float | None = None

    def document(self) -> dict[str, object]:
        if self.kind != "mounted":
            return {"kind": self.kind}
        return {"kind": self.kind, "family": self.family, "thickness_m": self.thickness_m, "cavity_m": self.cavity_m}


@dataclass(frozen=True)
class MaterialSet:
    label: str
    mount: Mount
    faces: tuple[FaceSpec, ...]


@dataclass(frozen=True)
class _FamilyBox:
    """一個家族樣板的盒界（不帶使用者深度上限：替身要涵蓋整個家族範圍，上限只會切小）。"""

    thickness: tuple[float, float]
    cavity: tuple[float, float] | None
    max_total_m: float | None

    def room(self, thickness: float) -> float:
        """這個厚度底下空腔最多到哪（家族空腔上界與總深度上限取小）。"""
        assert self.cavity is not None
        if self.max_total_m is None:
            return self.cavity[1]
        return min(self.cavity[1], float(depth_room(np.float64(self.max_total_m), np.float64(thickness))))


ORIGINAL = FaceSpec("original")
SAVED = FaceSpec("saved")


def original_walls[K: BoundaryKey](saved: Mapping[K, SurfaceMaterial]) -> dict[K, SurfaceMaterial]:
    """每一片貼上去之前的牆：貼在原牆上的構造取它記的原牆，其他材料原樣。"""
    return {key: material.wall if isinstance(material, MountedOnWall) else material for key, material in saved.items()}


class TrainingSets:
    """一份方案（面片序、存檔材料）＋搜尋設定＋抽法設定 → 訓練組、留出組、每組的材料與阻抗。"""

    def __init__(self, saved: Mapping[BoundaryKey, SurfaceMaterial], settings: MaterialGradientSettings,
                 sampling: SurrogateTrainingSettings, *, air: PhysicsConstants, ranges: PorousModelRanges,
                 frequencies_hz: Sequence[float]) -> None:
        self.keys = tuple(saved)
        self.names = tuple(surface_id(key) for key in self.keys)
        self.saved = dict(saved)
        self.originals = original_walls(saved)
        self.settings = settings
        self.sampling = sampling
        self.air, self.ranges = air, ranges
        self.axis = tuple(float(value) for value in frequencies_hz)
        families = tuple(family.id for family in settings.family)
        weights = sampling.sampling.family_weights
        if set(weights) != set(families):
            raise ValueError(f"抽法設定的家族 {sorted(weights)} 跟搜尋設定的家族 {sorted(families)} 不同，"
                             "兩邊要一樣（每個搜尋家族都要抽得到）")
        self.families = families
        self.weights = tuple(weights[family] for family in families)
        self.boxes = {family: self._box(family) for family in families}

    # ---- 家族盒界 ----

    def _box(self, family_id: str) -> _FamilyBox:
        preset = self.settings.family_by_id(family_id).template_preset
        design = run.designs_for([preset] * len(self.keys), self.settings, kept=self.originals, mount=MOUNT_ON_WALL)[0]
        cavity = (design.lower[1], design.upper[1]) if design.cavity_variable is not None else None
        return _FamilyBox((design.lower[0], design.upper[0]), cavity, design.max_total_depth_m)

    def _mounted(self, family: str, thickness: float, cavity: float | None = None) -> FaceSpec:
        box = self.boxes[family]
        if box.cavity is None:
            return FaceSpec("mounted", family, float(thickness), None)
        return FaceSpec("mounted", family, float(thickness), float(box.cavity[0] if cavity is None else cavity))

    def _corners(self, family: str) -> list[FaceSpec]:
        box = self.boxes[family]
        low, high = box.thickness
        thicknesses = [low] if high <= low else [low, high]
        if box.cavity is None:
            return [self._mounted(family, t) for t in thicknesses]
        corners = [self._mounted(family, low, box.cavity[0]),
                   self._mounted(family, low, (box.cavity[0] + box.room(low)) / 2),
                   self._mounted(family, low, box.room(low))]
        if high > low:
            corners += [self._mounted(family, high, box.cavity[0]), self._mounted(family, high, box.room(high))]
        return corners

    def _thinnest(self, family: str) -> FaceSpec:
        box = self.boxes[family]
        low = box.thickness[0]
        if box.cavity is None:
            return self._mounted(family, low)
        return self._mounted(family, low, (box.cavity[0] + box.room(low)) / 2)

    def _middle(self, family: str) -> FaceSpec:
        box = self.boxes[family]
        thickness = (box.thickness[0] + box.thickness[1]) / 2
        if box.cavity is None:
            return self._mounted(family, thickness)
        return self._mounted(family, thickness, (box.cavity[0] + box.room(thickness)) / 2)

    def _random_face(self, rng: np.random.Generator) -> FaceSpec:
        family = self.families[int(rng.choice(len(self.families), p=list(self.weights)))]
        box = self.boxes[family]
        low, high = box.thickness
        thickness = float(rng.uniform(low, high)) if high > low else float(low)
        if box.cavity is None:
            return self._mounted(family, thickness)
        return self._mounted(family, thickness, float(rng.uniform(box.cavity[0], box.room(thickness))))

    # ---- 組 ----

    def _uniform(self, label: str, face: FaceSpec, mount: Mount = MOUNT_ON_WALL) -> MaterialSet:
        return MaterialSet(label, mount, tuple(face for _ in self.keys))

    def _cycle(self, label: str, faces: Sequence[FaceSpec]) -> MaterialSet:
        return MaterialSet(label, MOUNT_ON_WALL, tuple(faces[index % len(faces)] for index in range(len(self.keys))))

    def _random_sets(self, seed: int, count: int, word: str) -> list[MaterialSet]:
        rng = np.random.default_rng(seed)
        probabilities = self.sampling.sampling.bare_probabilities
        every = self.sampling.sampling.rigid_every
        sets = []
        for index in range(count):
            bare = probabilities[index % len(probabilities)]
            mount: Mount = MOUNT_RIGID if index % every == every - 1 else MOUNT_ON_WALL
            faces = tuple(ORIGINAL if rng.random() < bare else self._random_face(rng) for _ in self.keys)
            sets.append(MaterialSet(f"{word}隨機 {index}（原牆機率 {bare:g}、{_mount_word(mount)}）", mount, faces))
        return sets

    def _base_sets(self) -> list[MaterialSet]:
        sets = [self._uniform("全部原牆", ORIGINAL)]
        if any(self.saved[key] != self.originals[key] for key in self.keys):
            sets.append(self._uniform("方案存檔那一份", SAVED))
        return sets

    def training_sets(self) -> list[MaterialSet]:
        sets = self._base_sets()
        for family in self.families:
            for corner in self._corners(family):
                sets.append(self._uniform(f"{family} 角點（{_face_word(corner)}）", corner))
        sets.append(self._cycle("混搭（原牆與各家族最薄、空腔中間輪流）",
                                [ORIGINAL, *(self._thinnest(family) for family in self.families)]))
        training = self.sampling.training
        return sets + self._random_sets(training.seed, training.random_sets, "訓練")

    def holdout_sets(self) -> list[MaterialSet]:
        sets = self._base_sets()
        sets.append(self._cycle("混搭（原牆與各家族厚度、空腔都取中間值輪流）",
                                [*(self._middle(family) for family in self.families), ORIGINAL]))
        holdout = self.sampling.holdout
        return sets + self._random_sets(holdout.seed, holdout.random_sets, "留出")

    # ---- 材料與阻抗 ----

    def _designs(self, material_set: MaterialSet) -> tuple[SurfaceDesign, ...]:
        presets = [self.settings.family_by_id(face.family).template_preset if face.kind == "mounted" and face.family
                   else None for face in material_set.faces]
        return run.designs_for(presets, self.settings, kept=self.originals, mount=material_set.mount)

    def materials(self, material_set: MaterialSet) -> dict[BoundaryKey, SurfaceMaterial | None]:
        if len(material_set.faces) != len(self.keys):
            raise ValueError(f"「{material_set.label}」有 {len(material_set.faces)} 片，方案有 {len(self.keys)} 片")
        designs = self._designs(material_set)
        values: list[float] = []
        for face, design in zip(material_set.faces, designs, strict=True):
            if design.material is None:
                continue
            assert face.thickness_m is not None
            values.append(face.thickness_m)
            if design.cavity_variable is not None:
                assert face.cavity_m is not None
                values.append(face.cavity_m)
        theta = checked_theta(np.asarray(values, dtype=np.float64), designs)
        for limit in depth_limits(designs):
            cavity = theta[limit.cavity_index] if limit.cavity_index is not None else 0.0
            if theta[limit.thickness_index] + cavity > np.float64(limit.max_total_m):
                raise ValueError(f"「{material_set.label}」厚度加空腔超過總深度上限")
        exported = materials_from_surfaces(designs, theta, self.settings)
        if tuple(exported) != self.keys:
            raise ValueError(f"「{material_set.label}」匯出的面片序 {list(exported)} 跟方案的 {list(self.keys)} 不同")
        return {key: self.saved[key] if face.kind == "saved" else exported[key]
                for key, face in zip(self.keys, material_set.faces, strict=True)}

    def impedances(self, material_set: MaterialSet) -> FrequencyBoundaryImpedances:
        rows = material_impedances(self.materials(material_set), self.axis, air=self.air, ranges=self.ranges)
        for key, row in rows.items():
            if row is not None and not np.all(np.isfinite(np.asarray(row))):
                raise ValueError(f"「{material_set.label}」{surface_id(key)} 的阻抗不是有限數")
        return rows

    def document(self, training: Sequence[MaterialSet], holdout: Sequence[MaterialSet]) -> dict[str, object]:
        """寫在替身旁邊的訓練說明：抽法設定、每一組每一片是什麼、匯出的材料。"""
        def entry(material_set: MaterialSet) -> dict[str, object]:
            materials = self.materials(material_set)
            return {"label": material_set.label, "mount": material_set.mount,
                    "faces": {name: face.document() for name, face in zip(self.names, material_set.faces, strict=True)},
                    "materials": {name: None if (material := materials[key]) is None else material.model_dump(mode="json")
                                  for name, key in zip(self.names, self.keys, strict=True)}}

        return {"version": "aosr.gradient.surrogate_training.v1", "patches": list(self.names),
                "axis_points": len(self.axis), "families": list(self.families),
                "sampling": self.sampling.model_dump(mode="json"),
                "training": [entry(item) for item in training], "holdout": [entry(item) for item in holdout]}


def _mount_word(mount: Mount) -> str:
    return "剛背" if mount == MOUNT_RIGID else "貼在原牆上"


def _face_word(face: FaceSpec) -> str:
    if face.cavity_m is None:
        return f"厚 {face.thickness_m:g} m"
    return f"厚 {face.thickness_m:g} m、空腔 {face.cavity_m:g} m"
