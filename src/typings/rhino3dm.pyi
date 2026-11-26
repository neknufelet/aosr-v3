from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from uuid import UUID

__version__: str

class UnitSystem(Enum):
    Meters = ...
    @property
    def name(self) -> str: ...

class Point3f:
    X: float
    Y: float
    Z: float

class Transform:
    M01: float
    M03: float
    M11: float
    M13: float
    M21: float
    M23: float

class GeometryBase: ...

class MeshVertexList:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Point3f: ...

class MeshFaceList:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> tuple[int, int, int, int]: ...

class Mesh(GeometryBase):
    Vertices: MeshVertexList
    Faces: MeshFaceList

class InstanceReference(GeometryBase):
    ParentIdefId: UUID
    Xform: Transform

class ObjectAttributes:
    LayerIndex: int
    Name: str

class File3dmObject:
    Geometry: GeometryBase
    Attributes: ObjectAttributes

class Layer:
    FullPath: str

class File3dmLayerTable:
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Layer: ...

class File3dmObjectTable:
    def __iter__(self) -> Iterator[File3dmObject]: ...

class InstanceDefinitionGeometry:
    Name: str

class File3dmInstanceDefinitionTable:
    def FindId(self, id: UUID) -> InstanceDefinitionGeometry | None: ...

class File3dmSettings:
    ModelUnitSystem: UnitSystem

class File3dm:
    Settings: File3dmSettings
    Layers: File3dmLayerTable
    Objects: File3dmObjectTable
    InstanceDefinitions: File3dmInstanceDefinitionTable
    @staticmethod
    def Read(path: str) -> File3dm | None: ...
