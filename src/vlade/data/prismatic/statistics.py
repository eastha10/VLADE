# Derived from eastha10/AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456.
# Apache-2.0 / inherited MIT: ../../models/teachers/AVA_LICENSE.txt.
# Only the active LIBERO/L1 input contract is retained.
"""Write the unchanged dataset_statistics.json checkpoint contract."""
import json
from pathlib import Path
from typing import Any
import numpy as np


def save_dataset_statistics(statistics: dict, directory: Path) -> None:
    """Serialize prepared-data statistics without depending on an upstream package."""

    def native(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: native(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [native(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "dataset_statistics.json").write_text(
        json.dumps(native(statistics), indent=2), encoding="utf-8"
    )
