"""waterqual — shared configuration and helpers for the water-quality UMAP
reproducibility pipeline.

The analytical stages live in ``scripts/``; this package holds the pieces they
share: configuration loading with relative-path resolution (``config``) and the
embedding-quality metrics used across tasks (``metrics``).
"""

from .config import Config, load_config, REPO_ROOT

__all__ = ["Config", "load_config", "REPO_ROOT"]
