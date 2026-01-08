import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from odat2.models import CableRecord, ValidationIssue


@dataclass(frozen=True)
class PhysicsContext:
    """Optional engineering context for physics checks.

    If you don't have these fields in the CSV yet, you can:
      - keep them None (validator will skip the relevant checks), or
      - enrich records before validation (future: a config file or a mapping table).
    """

    cable_type: Optional[str] = None  # "power" | "fiber" | "cat6" | "signal"
    wire_gauge: Optional[str] = None  # e.g., "12AWG"
    current_amps: Optional[float] = None
    nominal_voltage_v: float = 120.0


class CablePhysicsValidator:
    """Practical cable-physics validation.

    This file is where your 'theory vs measurement' split lives:

    - Theory: compact closed-form models (Ohm's law, Ampere/Gauss symmetry, attenuation models).
    - Measurement: your as-built data (lengths, types) and later: TDR traces, BER, power-quality logs.

    Change of coordinates shows up because most cable fields are naturally cylindrical:
      r, θ, z around a conductor → simplified E and B magnitudes.
    We don't solve full PDEs here; we use engineering bounds that are audit-friendly.
    """

    # DC resistance per km (Ω/km) at ~75°C for copper (order-of-magnitude; refine per standard you use)
    WIRE_RESISTANCE_OHM_PER_KM = {
        "14AWG": 8.28,
        "12AWG": 5.21,
        "10AWG": 3.28,
        "8AWG": 2.06,
        "6AWG": 1.30,
        "4AWG": 0.82,
        "3AWG": 0.65,
        "2AWG": 0.52,
        "1AWG": 0.41,
        "1/0AWG": 0.33,
        "2/0AWG": 0.26,
        "3/0AWG": 0.21,
        "4/0AWG": 0.17,
    }

    # Very rough ampacity at 75°C insulation (A). Real ampacity depends on insulation, bundling, ambient, etc.
    AMPACITY_A = {
        "14AWG": 15,
        "12AWG": 20,
        "10AWG": 30,
        "8AWG": 55,
        "6AWG": 75,
        "4AWG": 95,
        "3AWG": 110,
        "2AWG": 130,
        "1AWG": 150,
        "1/0AWG": 170,
        "2/0AWG": 195,
        "3/0AWG": 225,
        "4/0AWG": 260,
    }

    def __init__(self, max_voltage_drop_percent: float = 3.0):
        self.max_voltage_drop_percent = max_voltage_drop_percent

        # Heuristics for inferring type from ids/symbols (works with messy exports)
        self._fiber_hint = re.compile(r"(FIB|FBR|SMF|MMF|FO)", re.IGNORECASE)
        self._cat_hint = re.compile(r"(CAT\s*6|CAT6|RJ45|ETH|LAN)", re.IGNORECASE)
        self._power_hint = re.compile(r"(PWR|AC|DC|480|277|120|24V|\bHV\b)", re.IGNORECASE)

    def infer_context(self, record: CableRecord) -> PhysicsContext:
        text = " ".join([record.symbol or "", record.cable_id or "", record.location or ""])
        ctype = None
        if self._fiber_hint.search(text):
            ctype = "fiber"
        elif self._cat_hint.search(text):
            ctype = "cat6"
        elif self._power_hint.search(text):
            ctype = "power"
        else:
            ctype = "signal"
        return PhysicsContext(cable_type=ctype)

    def calculate_voltage_drop_percent(
        self, *, current_amps: float, length_m: float, wire_gauge: str, nominal_voltage_v: float
    ) -> Optional[float]:
        r_per_km = self.WIRE_RESISTANCE_OHM_PER_KM.get(wire_gauge)
        if r_per_km is None:
            return None
        length_km = length_m / 1000.0
        # Round-trip conductor length (out + back) approximation
        v_drop = 2.0 * current_amps * length_km * r_per_km
        return (v_drop / nominal_voltage_v) * 100.0 if nominal_voltage_v > 0 else None

    def calculate_signal_loss_db(self, *, length_m: float, cable_type: str) -> Optional[float]:
        length_km = length_m / 1000.0
        if cable_type == "fiber":
            # Typical single-mode fiber attenuation at 1550nm is ~0.2-0.3 dB/km; add connector loss budget
            return 0.3 * length_km + 0.5
        if cable_type == "cat6":
            # Cat6 is specified by channel insertion loss; simplest audit: flag length > 100 m
            return 2.0 * (length_m / 100.0)  # ~2 dB/100m @ 100 MHz
        return None

    def emi_risk_bucket(self, *, parallel_length_m: float, separation_m: float, frequency_hz: float = 60.0) -> str:
        # Very simple: longer parallel run, tighter separation, higher freq => higher coupling risk.
        # This is a policy-style classifier, not a field solver.
        if separation_m <= 0:
            return "high"
        score = (parallel_length_m / max(separation_m, 0.01)) * math.sqrt(max(frequency_hz, 1.0) / 60.0)
        if score > 5000:
            return "high"
        if score > 1500:
            return "medium"
        return "low"

    def validate(self, record: CableRecord, context: Optional[PhysicsContext] = None) -> List[ValidationIssue]:
        ctx = context or self.infer_context(record)
        issues: List[ValidationIssue] = []

        # --- Signal / comms checks (spec-style) ---
        if ctx.cable_type in ("cat6", "fiber"):
            loss_db = self.calculate_signal_loss_db(length_m=record.cable_length_m, cable_type=ctx.cable_type)
            if ctx.cable_type == "cat6" and record.cable_length_m > 100:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="exceeds_cable_spec",
                        message=f"Cat6 channel length {record.cable_length_m:.1f} m exceeds 100 m. "
                        "Use fiber, add a switch, or re-route.",
                        cable_id=record.cable_id,
                        device_tag=record.device_tag,
                        drawing_id=record.drawing_id,
                    )
                )
            if ctx.cable_type == "fiber" and loss_db is not None and loss_db > 15.0:
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        issue_type="high_signal_loss",
                        message=f"Estimated fiber loss {loss_db:.2f} dB is high; consider cleaning/connector budget, "
                        "splices, or amplifier/repeater if required.",
                        cable_id=record.cable_id,
                        device_tag=record.device_tag,
                        drawing_id=record.drawing_id,
                    )
                )

        # --- Power checks (needs current + gauge) ---
        if ctx.cable_type == "power" and ctx.current_amps is not None and ctx.wire_gauge is not None:
            vdp = self.calculate_voltage_drop_percent(
                current_amps=ctx.current_amps,
                length_m=record.cable_length_m,
                wire_gauge=ctx.wire_gauge,
                nominal_voltage_v=ctx.nominal_voltage_v,
            )
            if vdp is not None and vdp > self.max_voltage_drop_percent:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="excessive_voltage_drop",
                        message=f"Estimated voltage drop {vdp:.2f}% exceeds {self.max_voltage_drop_percent:.1f}%. "
                        "Consider larger conductor, shorter run, or higher distribution voltage.",
                        cable_id=record.cable_id,
                        device_tag=record.device_tag,
                        drawing_id=record.drawing_id,
                    )
                )

            amp = self.AMPACITY_A.get(ctx.wire_gauge)
            if amp is not None and ctx.current_amps > amp:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        issue_type="ampacity_exceeded",
                        message=f"Load {ctx.current_amps:.1f} A exceeds {ctx.wire_gauge} rough ampacity {amp} A. "
                        "Fire/thermal risk — upsize conductor or reduce load.",
                        cable_id=record.cable_id,
                        device_tag=record.device_tag,
                        drawing_id=record.drawing_id,
                    )
                )

        # --- Measurement-vs-theory placeholders (future hooks) ---
        # If you later add TDR/Chirp traces or BER/CRC logs, this validator is the right home:
        #  - Chirps: frequency-sweep input to characterize channel impulse response / reflections.
        #  - Parity/CRC: integrity checks to relate observed BER to topology, EMI risk, and length budgets.

        return issues


class ThermalValidator:
    """Simple thermal bounding model (steady-state-ish).

    NOTE: Real thermal modeling is a PDE on a 3D domain; in practice, we do:
      - bounds (ampacity, insulation temp limits)
      - conservative derates
      - spot measurement (IR camera, temperature sensors)
    """

    def __init__(self, ambient_temp_c: float = 40.0, max_conductor_temp_c: float = 90.0):
        self.ambient_temp_c = ambient_temp_c
        self.max_conductor_temp_c = max_conductor_temp_c

    def estimate_temp_rise_c(self, *, current_amps: float, resistance_ohm_per_km: float, length_m: float) -> float:
        # Power loss (I^2 R) for one conductor over length; this is intentionally conservative/simplified.
        r_total = resistance_ohm_per_km * (length_m / 1000.0)
        p_loss = (current_amps ** 2) * r_total
        # Collapse environment into an effective thermal resistance (C/W)
        # This is a knob you can calibrate against measurement.
        r_th = 0.04  # °C/W (arbitrary; tune later)
        return p_loss * r_th

    def validate(self, record: CableRecord, *, current_amps: float, wire_gauge: str) -> List[ValidationIssue]:
        issues: List[ValidationIssue] = []
        rpkm = CablePhysicsValidator.WIRE_RESISTANCE_OHM_PER_KM.get(wire_gauge)
        if rpkm is None:
            return issues

        rise = self.estimate_temp_rise_c(current_amps=current_amps, resistance_ohm_per_km=rpkm, length_m=record.cable_length_m)
        final_temp = self.ambient_temp_c + rise
        if final_temp > self.max_conductor_temp_c:
            issues.append(
                ValidationIssue(
                    severity="error",
                    issue_type="thermal_overload",
                    message=f"Estimated conductor temp {final_temp:.1f}°C exceeds {self.max_conductor_temp_c:.1f}°C. "
                    "Reduce load, improve cooling/derating, or upsize conductor.",
                    cable_id=record.cable_id,
                    device_tag=record.device_tag,
                    drawing_id=record.drawing_id,
                )
            )
        return issues
