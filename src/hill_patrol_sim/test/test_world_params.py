"""Validation tests for hill simulation world parameters (T12).

Verifies Section 5.3 specifications:
- Every configuration parameter must include a non-empty 'source' field.
- Flight step segments sum to total step count (500 up, 500 down).
- Each flight segment has <= 18 steps (GB 50352-2019 6.8.5).
- Total step climb = step_count * step_height = 75.0 m.
- Total horizontal stair length = 182.4 m per side.
- Small world scaled parameters match design specifications.
"""

from copy import deepcopy
from pathlib import Path
from numbers import Real
import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
HILL_WORLD_YAML = CONFIG_DIR / "hill_world.yaml"
HILL_WORLD_SMALL_YAML = CONFIG_DIR / "hill_world_small.yaml"

REQUIRED_FIELDS = {
    "world_name": str,
    "step_count_up": int,
    "step_count_down": int,
    "step_height": Real,
    "tread_depth": Real,
    "flight_max_steps": int,
    "flight_steps_up": list,
    "flight_steps_down": list,
    "intermediate_landing_depth": Real,
    "step_width": Real,
    "ramp_slope_deg": Real,
    "ramp_length": Real,
    "summit_length": Real,
    "summit_roughness_max": Real,
    "summit_seed": int,
    "trees_seed": int,
    "trees_radius": Real,
    "trees_height": Real,
    "trees_spacing_min": Real,
    "trees_spacing_max": Real,
    "trees_offset_y": Real,
    "stair_zones_count": int,
    "origin_latitude": Real,
    "origin_longitude": Real,
    "origin_altitude": Real,
}


def load_raw_world_params(yaml_path: Path) -> dict:
    """Load raw yaml content containing {param: {value: ..., source: ...}}."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"World param file not found: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Root YAML object must be a dictionary in {yaml_path}")
    return data


def validate_world_params(raw_data: dict) -> dict:
    """Validate parameter schema, sources, and physical/geometric constraints.

    Returns extracted value dictionary {param_name: value}.
    """
    if not isinstance(raw_data, dict) or not raw_data:
        raise ValueError("Parameter dictionary must be non-empty")

    extracted = {}

    for key, expected_type in REQUIRED_FIELDS.items():
        if key not in raw_data:
            raise ValueError(f"Missing required parameter: {key}")

        entry = raw_data[key]
        if not isinstance(entry, dict):
            raise ValueError(
                f"Parameter '{key}' must be a dictionary containing 'value' and 'source'"
            )

        if "source" not in entry:
            raise ValueError(f"Parameter '{key}' is missing 'source' field")
        source = entry["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"Parameter '{key}' has empty or non-string 'source'")

        if "value" not in entry:
            raise ValueError(f"Parameter '{key}' is missing 'value' field")
        val = entry["value"]

        # Type checks
        if expected_type is int:
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(f"Parameter '{key}' must be an integer, got {type(val)}")
        elif expected_type is Real:
            if isinstance(val, bool) or not isinstance(val, Real):
                raise TypeError(f"Parameter '{key}' must be a real number, got {type(val)}")
            val = float(val)
        elif expected_type is str:
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"Parameter '{key}' must be a non-empty string")
        elif expected_type is list:
            if not isinstance(val, list) or not val:
                raise ValueError(f"Parameter '{key}' must be a non-empty list")
            for item in val:
                if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                    raise ValueError(
                        f"Elements in list '{key}' must be positive integers, got {item}"
                    )

        extracted[key] = val

    # Verify every extra key in raw_data also has source
    for key, entry in raw_data.items():
        if not isinstance(entry, dict) or "source" not in entry or "value" not in entry:
            raise ValueError(f"Field '{key}' must contain 'value' and 'source'")
        if not isinstance(entry["source"], str) or not entry["source"].strip():
            raise ValueError(f"Field '{key}' has empty or invalid 'source'")

    # Geometric & physical constraints
    if extracted["step_height"] <= 0:
        raise ValueError("step_height must be positive")
    if extracted["tread_depth"] <= 0:
        raise ValueError("tread_depth must be positive")
    if extracted["step_width"] <= 0:
        raise ValueError("step_width must be positive")
    if extracted["flight_max_steps"] <= 0 or extracted["flight_max_steps"] > 18:
        raise ValueError("flight_max_steps must be in 1..18 per GB 50352-2019 6.8.5")

    # Sum of flights must equal step counts
    flight_sum_up = sum(extracted["flight_steps_up"])
    if flight_sum_up != extracted["step_count_up"]:
        raise ValueError(
            f"Sum of flight_steps_up ({flight_sum_up}) does not match "
            f"step_count_up ({extracted['step_count_up']})"
        )

    flight_sum_down = sum(extracted["flight_steps_down"])
    if flight_sum_down != extracted["step_count_down"]:
        raise ValueError(
            f"Sum of flight_steps_down ({flight_sum_down}) does not match "
            f"step_count_down ({extracted['step_count_down']})"
        )

    # Every flight segment <= flight_max_steps (<= 18)
    for s in extracted["flight_steps_up"]:
        if s > extracted["flight_max_steps"]:
            raise ValueError(
                f"Flight segment in flight_steps_up exceeds flight_max_steps: {s} > {extracted['flight_max_steps']}"
            )

    for s in extracted["flight_steps_down"]:
        if s > extracted["flight_max_steps"]:
            raise ValueError(
                f"Flight segment in flight_steps_down exceeds flight_max_steps: {s} > {extracted['flight_max_steps']}"
            )

    if extracted["intermediate_landing_depth"] <= 0:
        raise ValueError("intermediate_landing_depth must be positive")
    if extracted["ramp_slope_deg"] <= 0 or extracted["ramp_slope_deg"] >= 90:
        raise ValueError("ramp_slope_deg must be in (0, 90)")
    if extracted["ramp_length"] <= 0:
        raise ValueError("ramp_length must be positive")
    if extracted["summit_length"] <= 0:
        raise ValueError("summit_length must be positive")
    if extracted["summit_roughness_max"] < 0:
        raise ValueError("summit_roughness_max must be non-negative")
    if extracted["stair_zones_count"] != 2:
        raise ValueError("stair_zones_count must be 2 (stair_up, stair_down)")

    return extracted


def compute_derived_params(params: dict) -> dict:
    """Compute analytical climb and horizontal run dimensions."""
    step_climb_up = params["step_count_up"] * params["step_height"]
    step_climb_down = params["step_count_down"] * params["step_height"]

    landings_up = len(params["flight_steps_up"]) - 1
    landings_down = len(params["flight_steps_down"]) - 1

    horiz_len_up = (
        params["step_count_up"] * params["tread_depth"]
        + landings_up * params["intermediate_landing_depth"]
    )
    horiz_len_down = (
        params["step_count_down"] * params["tread_depth"]
        + landings_down * params["intermediate_landing_depth"]
    )

    return {
        "step_climb_up": step_climb_up,
        "step_climb_down": step_climb_down,
        "landings_up": landings_up,
        "landings_down": landings_down,
        "stair_horizontal_length_up": horiz_len_up,
        "stair_horizontal_length_down": horiz_len_down,
    }


# =============================================================================
# Pytest Test Suite
# =============================================================================

def test_full_world_params_exist_and_validate():
    """Verify hill_world.yaml validates and strictly matches 5.3 specifications."""
    raw = load_raw_world_params(HILL_WORLD_YAML)
    params = validate_world_params(raw)

    assert params["world_name"] == "hill_world"
    assert params["step_count_up"] == 500
    assert params["step_count_down"] == 500
    assert params["step_height"] == pytest.approx(0.15)
    assert params["tread_depth"] == pytest.approx(0.30)
    assert params["flight_max_steps"] == 18
    assert params["intermediate_landing_depth"] == pytest.approx(1.2)
    assert params["step_width"] == pytest.approx(1.5)
    assert params["ramp_slope_deg"] == pytest.approx(15.0)
    assert params["ramp_length"] == pytest.approx(20.0)
    assert params["summit_length"] == pytest.approx(30.0)
    assert params["summit_roughness_max"] == pytest.approx(0.10)
    assert params["stair_zones_count"] == 2


def test_full_world_derived_climb_and_length():
    """Acceptance criteria:推算爬升 75 m、每侧台阶水平长度 182.4 m."""
    raw = load_raw_world_params(HILL_WORLD_YAML)
    params = validate_world_params(raw)
    derived = compute_derived_params(params)

    # 500 * 0.15 = 75.0 m
    assert derived["step_climb_up"] == pytest.approx(75.0, abs=1e-6)
    assert derived["step_climb_down"] == pytest.approx(75.0, abs=1e-6)

    # 27 segments * 18 + 1 segment * 14 = 28 segments -> 27 intermediate landings
    assert derived["landings_up"] == 27
    assert derived["landings_down"] == 27

    # 500 * 0.30 + 27 * 1.2 = 150.0 + 32.4 = 182.4 m
    assert derived["stair_horizontal_length_up"] == pytest.approx(182.4, abs=1e-6)
    assert derived["stair_horizontal_length_down"] == pytest.approx(182.4, abs=1e-6)


def test_small_world_params_exist_and_validate():
    """Verify hill_world_small.yaml scaled parameters."""
    raw = load_raw_world_params(HILL_WORLD_SMALL_YAML)
    params = validate_world_params(raw)

    assert params["world_name"] == "hill_world_small"
    assert params["step_count_up"] == 18
    assert params["step_count_down"] == 18
    assert params["flight_steps_up"] == [18]
    assert params["flight_steps_down"] == [18]
    assert params["ramp_length"] == pytest.approx(5.0)
    assert params["summit_length"] == pytest.approx(5.0)

    derived = compute_derived_params(params)
    assert derived["step_climb_up"] == pytest.approx(18 * 0.15, abs=1e-6)
    assert derived["landings_up"] == 0
    assert derived["stair_horizontal_length_up"] == pytest.approx(18 * 0.30, abs=1e-6)


def test_validation_rejects_missing_or_empty_source():
    """Every parameter must have a valid non-empty 'source' annotation."""
    raw = load_raw_world_params(HILL_WORLD_YAML)

    # Missing source
    tampered = deepcopy(raw)
    del tampered["step_height"]["source"]
    with pytest.raises(ValueError, match="missing 'source'"):
        validate_world_params(tampered)

    # Empty source
    tampered = deepcopy(raw)
    tampered["step_height"]["source"] = "   "
    with pytest.raises(ValueError, match="empty or non-string 'source'"):
        validate_world_params(tampered)


def test_validation_rejects_invalid_flight_segments():
    """Rejects if flights sum doesn't match total or if a flight > 18."""
    raw = load_raw_world_params(HILL_WORLD_YAML)

    # Segment sum mismatch
    tampered = deepcopy(raw)
    tampered["flight_steps_up"]["value"] = [18] * 27  # sum = 486 != 500
    with pytest.raises(ValueError, match="does not match step_count_up"):
        validate_world_params(tampered)

    # Segment > 18
    tampered = deepcopy(raw)
    tampered["flight_steps_up"]["value"] = [19] + [18] * 25 + [31]  # 19 > 18
    with pytest.raises(ValueError, match="exceeds flight_max_steps"):
        validate_world_params(tampered)


def test_validation_rejects_missing_fields_and_non_dict():
    """Ensure malformed structure is caught immediately."""
    with pytest.raises(ValueError, match="Parameter dictionary must be non-empty"):
        validate_world_params({})

    raw = load_raw_world_params(HILL_WORLD_YAML)
    tampered = deepcopy(raw)
    del tampered["world_name"]
    with pytest.raises(ValueError, match="Missing required parameter"):
        validate_world_params(tampered)

    tampered = deepcopy(raw)
    tampered["ramp_slope_deg"] = 15.0  # not a dict with value/source
    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_world_params(tampered)
