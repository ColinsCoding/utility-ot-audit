from dataclasses import dataclass

@dataclass(frozen=True)
class SensorSpec:
    sensor_id: str
    x: int
    y: int
    range_cells: float
    fov_deg: float = 360.0
    heading_deg: float = 0.0
    sensor_type: str = "camera"
