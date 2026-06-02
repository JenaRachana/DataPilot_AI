"""
Artifact directory helper.

All generated files (EDA charts, trained model, deployment scripts) are written
under ``artifacts/<thread_id>/`` so file output stays confined to a single,
predictable location per run.
"""

from pathlib import Path
from typing import Optional, Dict, List, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ARTIFACTS_ROOT = Path("artifacts")


def get_artifacts_dir(config: Optional[Dict] = None) -> Path:
    """Return (creating if needed) the artifacts directory for this run."""
    thread_id = "default"
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id", "default")

    run_dir = ARTIFACTS_ROOT / str(thread_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_figures(
    figures: List[Any],
    run_dir: Path,
    phase: str = "eda",
) -> List[Dict[str, str]]:
    """
    Persist matplotlib figures produced by generated code.

    ``figures`` may contain either ``(title, fig)`` tuples or bare figures.
    Returns a list of ``{"title", "path", "phase"}`` artifact records and closes
    each figure to free memory.
    """
    saved: List[Dict[str, str]] = []
    for i, item in enumerate(figures or []):
        if isinstance(item, (tuple, list)) and len(item) == 2:
            title, fig = item
        else:
            title, fig = f"{phase} figure {i + 1}", item

        path = run_dir / f"{phase}_{i + 1}.png"
        try:
            fig.savefig(path, bbox_inches="tight")
        except Exception:
            continue
        finally:
            try:
                plt.close(fig)
            except Exception:
                pass

        saved.append({"title": str(title), "path": str(path), "phase": phase})
    return saved

