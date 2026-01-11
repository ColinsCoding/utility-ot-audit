from pathlib import Path
import pandas as pd
from .models import SensorSpec

def load_sensors_csv(path: str):
    df = pd.read_csv(Path(path))
    required = {"sensor_id","x","y","range_cells"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if "fov_deg" not in df.columns: df["fov_deg"] = 360.0
    if "heading_deg" not in df.columns: df["heading_deg"] = 0.0
    if "sensor_type" not in df.columns: df["sensor_type"] = "camera"
    return [
        SensorSpec(
            sensor_id=str(r.sensor_id),
            x=int(r.x), y=int(r.y),
            range_cells=float(r.range_cells),
            fov_deg=float(r.fov_deg),
            heading_deg=float(r.heading_deg),
            sensor_type=str(r.sensor_type),
        )
        for r in df.itertuples(index=False)
    ]
