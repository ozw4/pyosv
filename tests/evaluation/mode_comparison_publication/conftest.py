from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any

import pytest

from pyosv.evaluation.synthetic_mode_comparison import (
    SyntheticModeComparisonConfig,
    run_mode_comparison,
    write_artifact_bundle,
)
from tests.evaluation.f3d_mode_comparison.test_integration import (
    _run_fixture,
    _write_fixture,
)


def snapshot_files(root: Path) -> dict[str, tuple[bytes, int, int, str]]:
    snapshot = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        payload = path.read_bytes()
        stat = path.stat()
        snapshot[path.relative_to(root).as_posix()] = (
            payload,
            stat.st_size,
            stat.st_mtime_ns,
            sha256(payload).hexdigest(),
        )
    return snapshot


@pytest.fixture(scope="session")
def source_bundles(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    pytest.importorskip("matplotlib")
    root = tmp_path_factory.mktemp("mode-comparison-publication-sources")
    synthetic_config = SyntheticModeComparisonConfig(
        case_ids=("weak_noisy_plane", "single_vertical_plane"),
        trial_seeds=(20260707, 20260708),
        shape=(9, 9, 9),
    )
    synthetic_result = run_mode_comparison(synthetic_config)
    synthetic_bundle = write_artifact_bundle(
        synthetic_result,
        root / "synthetic",
        config=synthetic_config,
    )

    data_root = root / "f3-data"
    spec = _write_fixture(data_root)
    f3_bundle = root / "f3"
    with pytest.MonkeyPatch.context() as patch:
        _run_fixture(
            data_root,
            f3_bundle,
            spec,
            Counter(),
            resume=False,
            monkeypatch=patch,
        )

    return {
        "root": root,
        "synthetic": synthetic_bundle,
        "f3": f3_bundle,
        "data_root": data_root,
        "synthetic_snapshot": snapshot_files(synthetic_bundle),
        "f3_snapshot": snapshot_files(f3_bundle),
        "data_snapshot": snapshot_files(data_root),
    }


@pytest.fixture
def publication_bundle(
    source_bundles: dict[str, Any], tmp_path: Path
) -> tuple[Path, dict[str, Any]]:
    from pyosv.evaluation.mode_comparison_publication import generate_publication_bundle

    output = tmp_path / "publication"
    generate_publication_bundle(
        source_bundles["synthetic"],
        source_bundles["f3"],
        source_bundles["data_root"],
        output,
    )
    return output, source_bundles
