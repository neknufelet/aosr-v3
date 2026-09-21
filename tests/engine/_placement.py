"""手造分類評估共用的明確擺位；正式入口考卷不用這份假資料。"""
from typing import Final

from aosr.scoring.placement import Placement


POINT_PLACEMENT: Final[Placement] = Placement(
    speaker_positions_m=(("left", (1.2, 1.3, 1.1)),),
    receiver_positions_m=(("main-seat", (4.7, 2.8, 1.4)),),
)

EMPTY_PLACEMENT: Final[Placement] = Placement(
    speaker_positions_m=(),
    receiver_positions_m=(),
)


def channel_point_placement(receiver_id: str, role: str) -> Placement:
    """舊聲道考卷手造單點輸入的代號與座標。"""
    speaker_positions = {
        "left": (0.2, 0.3, 1.1),
        "right": (0.2, 0.7, 1.1),
        "center": (0.2, 0.5, 1.1),
    }
    receiver_positions = {
        "main": (0.0, 0.0, 1.2),
        "front": (0.0, 0.5, 1.2),
    }
    return Placement(
        speaker_positions_m=((f"speaker-{role}", speaker_positions[role]),),
        receiver_positions_m=(
            (receiver_id, receiver_positions.get(receiver_id, (9.0, 9.0, 9.0))),
        ),
    )
