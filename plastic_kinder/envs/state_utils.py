from __future__ import annotations

from typing import Any


def _object_name(obj: Any) -> str:
    return str(getattr(obj, "name", ""))


def _object_type_name(obj: Any) -> str:
    obj_type = getattr(obj, "type", None)
    return str(getattr(obj_type, "name", obj_type or ""))


def _matches_object_type(obj: Any, object_type: str) -> bool:
    return object_type in _object_name(obj) or object_type == _object_type_name(obj)


def _xy_from_state_object(state: Any, obj: Any) -> dict[str, Any]:
    return {
        "name": _object_name(obj),
        "type": _object_type_name(obj),
        "x": state.get(obj, "x"),
        "y": state.get(obj, "y"),
    }


def get_objects_by_type(state: Any, object_type: str) -> list[Any]:
    """Return all objects of a given type from a KinDER object-centric state.

    This is intentionally defensive because KinDER's object-centric state API
    may expose objects differently across versions.
    """
    if hasattr(state, "get_objects_by_type"):
        return list(state.get_objects_by_type(object_type))

    if hasattr(state, "data") and isinstance(state.data, dict):
        return [
            _xy_from_state_object(state, obj)
            for obj in state.data
            if _matches_object_type(obj, object_type)
        ]

    if hasattr(state, "objects"):
        objects = state.objects
        if isinstance(objects, dict):
            return [
                obj
                for obj_name, obj in objects.items()
                if object_type in str(obj_name) or _matches_object_type(obj, object_type)
            ]
        return [
            obj
            for obj in objects
            if _matches_object_type(obj, object_type)
        ]

    # Fallback: inspect public attributes.
    matched = []
    for name in dir(state):
        if name.startswith("_"):
            continue
        value = getattr(state, name)
        if object_type in name:
            if isinstance(value, list):
                matched.extend(value)
            else:
                matched.append(value)

    return matched


def get_first_object_by_type(state: Any, object_type: str) -> Any:
    objects = get_objects_by_type(state, object_type)
    if not objects:
        raise ValueError(f"No object found for type/name: {object_type}")
    return objects[0]


def get_xy(obj: Any) -> tuple[float, float]:
    """Extract x,y coordinates from a KinDER object."""
    if hasattr(obj, "x") and hasattr(obj, "y"):
        return float(obj.x), float(obj.y)

    if isinstance(obj, dict):
        return float(obj["x"]), float(obj["y"])

    raise TypeError(f"Could not extract x,y from object: {obj}")


def robot_xy(state: Any) -> tuple[float, float]:
    return get_xy(get_first_object_by_type(state, "robot"))


def target_block_xy(state: Any) -> tuple[float, float]:
    return get_xy(get_first_object_by_type(state, "target_block"))


def target_surface_xy(state: Any) -> tuple[float, float]:
    return get_xy(get_first_object_by_type(state, "target_surface"))


def obstruction_xy(state: Any, index: int = 0) -> tuple[float, float]:
    obstructions = get_objects_by_type(state, "obstruction")
    if not obstructions:
        # In pretty_str, obstruction is under type rectangle but named obstruction0.
        obstructions = get_objects_by_type(state, "rectangle")
    if index >= len(obstructions):
        raise IndexError(f"Requested obstruction {index}, found {len(obstructions)}")
    return get_xy(obstructions[index])
