"""Reproducible, geometry-checked block task generation."""


def generate(*args, **kwargs):
    from .generator import generate as implementation
    return implementation(*args, **kwargs)


__all__ = ["generate"]
