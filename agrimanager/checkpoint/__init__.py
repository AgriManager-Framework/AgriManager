"""Checkpoint post-processing: merge sharded trainer output and publish it.

Nothing is imported eagerly. ``publish`` is also run as ``python -m
agrimanager.checkpoint.publish``, and importing it here would load the module
twice under different names, which runpy warns about.
"""

__all__ = ["PublishError", "PublishSpec", "publish_checkpoint"]


def __getattr__(name: str):
    if name in __all__:
        from . import publish

        return getattr(publish, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
