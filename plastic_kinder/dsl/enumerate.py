from __future__ import annotations

from itertools import product

from plastic_kinder.dsl.primitives import ActionPrimitive
from plastic_kinder.dsl.program import Program


def enumerate_programs(
    primitives: tuple[ActionPrimitive, ...],
    max_length: int,
) -> list[Program]:
    """Enumerate all primitive sequences up to max_length."""
    if max_length < 1:
        raise ValueError("max_length must be >= 1.")

    programs: list[Program] = []
    for length in range(1, max_length + 1):
        for primitive_seq in product(primitives, repeat=length):
            programs.append(Program(primitives=primitive_seq))

    return programs