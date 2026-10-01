"""New restart-v2 assignment-only finalization candidate (not historical recovery)."""

def finalize(*args, **kwargs):
    from .finalizer import finalize as implementation
    return implementation(*args, **kwargs)
