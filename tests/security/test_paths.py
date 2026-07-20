from pathlib import Path

import pytest

from recommender.exceptions import ArtifactError
from recommender.utils.io import ensure_within


@pytest.mark.security
def test_path_traversal_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    with pytest.raises(ArtifactError, match="escapes"):
        ensure_within(root / ".." / "secret", root)
