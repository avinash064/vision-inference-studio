"""Auto Annotation v0.1 domain module.

The module orchestrates immutable Vision recipes; it intentionally contains no
model runtime or adapter implementation of its own.
"""

from .manager import AutoAnnotationManager

__all__ = ["AutoAnnotationManager"]
