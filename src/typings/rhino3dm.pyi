from __future__ import annotations

from collections.abc import Iterator
from uuid import UUID

__version__: str

class UnitSystem:
    Meters: UnitSystem
    def __init__(self, value: int) -> None: ...
    @property
    def name(self) -> str: ...

class ObjectMode:
    Normal: ObjectMode
    InstanceDefinitionObject: ObjectMode
    def __init__(self, value: int) -> None: ...

class Point3d:
    X: float
    Y: float
    Z: float
    def __init__(self, x: float, y: float, z: float) -> None: ...

class Transform:
    def __init__(self, diagonalValue: float) -> None: ...
    @staticmethod
    def Translation(x: float, y: float, z: float) -> Transform: ...
    M01: float
    M03: float
    M11: float
    M13: float
    M21: float
    M23: float

class GeometryBase: ...

class MeshVertexList:
    def __len__(self) -> int: ...
    def Point3dAt(self, index: int) -> Point3d: ...
    def Add(self, x: float, y: float, z: float) -> int: ...

class MeshFaceList:
    def __len__(self) -> int: ...
    def __getitem__(self, arg0: int) -> tuple[int, int, int, int]: ...
    def AddFace(self, vertex1: int, vertex2: int, vertex3: int) -> int: ...

class Mesh(GeometryBase):
    Vertices: MeshVertexList
    Faces: MeshFaceList
    def __init__(self) -> None: ...

class InstanceReference(GeometryBase):
    ParentIdefId: UUID
    Xform: Transform
    def __init__(self, arg0: UUID, arg1: Transform) -> None: ...

class ObjectAttributes:
    LayerIndex: int
    Name: str
    Mode: ObjectMode
    def __init__(self) -> None: ...

class File3dmObject:
    Geometry: GeometryBase
    Attributes: ObjectAttributes

class Layer:
    FullPath: str

class File3dmLayerTable:
    def __len__(self) -> int: ...
    def __getitem__(self, arg0: int) -> Layer: ...

class File3dmObjectTable:
    def __iter__(self) -> Iterator[File3dmObject]: ...
    def AddPoint(self, point: Point3d, attributes: ObjectAttributes) -> UUID: ...
    def AddInstanceObject(self, idef: InstanceReference, attributes: ObjectAttributes | None = ...) -> object: ...

class InstanceDefinition:
    Name: str
    Id: UUID

class File3dmInstanceDefinitionTable:
    def FindId(self, id: UUID) -> InstanceDefinition | None: ...
    def FindIndex(self, index: int) -> InstanceDefinition | None: ...
    def Add(self, name: str, description: str, url: str, urlTag: str, basePoint: Point3d,
            geometry: tuple[GeometryBase, ...], attributes: tuple[ObjectAttributes, ...]) -> int: ...

class File3dmSettings:
    ModelUnitSystem: UnitSystem

class File3dm:
    Settings: File3dmSettings
    Layers: File3dmLayerTable
    Objects: File3dmObjectTable
    InstanceDefinitions: File3dmInstanceDefinitionTable
    @staticmethod
    def Read(path: str) -> File3dm | None: ...
    def Write(self, path: str, version: int = ...) -> bool: ...
