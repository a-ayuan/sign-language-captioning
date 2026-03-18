import json
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

from slc.utils.io import ensure_dir

VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
EXPECTED_SPLITS = ("train", "val", "test")


class KaggleDownloadError(RuntimeError):
    """Raised when the Kaggle dataset cannot be downloaded or staged."""


def _has_video_children(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    for item in path.rglob("*"):
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES:
            return True
    return False


def _looks_like_ready_split_root(path: Path) -> bool:
    return all((path / split).exists() and _has_video_children(path / split) for split in EXPECTED_SPLITS)


def _iter_candidate_roots(root: Path) -> Iterable[Path]:
    yield root
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            yield path


def find_ready_split_root(root: Path) -> Path:
    for candidate in _iter_candidate_roots(root):
        if _looks_like_ready_split_root(candidate):
            return candidate
    raise KaggleDownloadError(
        "Could not find train/val/test split folders with video files in the downloaded Kaggle dataset."
    )


def _safe_symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def stage_ready_split_root(src_root: Path, dst_root: Path) -> Path:
    dst_root = ensure_dir(dst_root)
    for split in EXPECTED_SPLITS:
        _safe_symlink_or_copy(src_root / split, dst_root / split)
    return dst_root


def download_kaggle_dataset(dataset_slug: str, cache_root: Path) -> Path:
    cache_root = ensure_dir(cache_root)
    try:
        import kagglehub  # type: ignore
    except ImportError as exc:
        raise KaggleDownloadError(
            "kagglehub is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    try:
        downloaded_path = Path(kagglehub.dataset_download(dataset_slug))
    except Exception as exc:  # pragma: no cover - network/auth dependent
        raise KaggleDownloadError(
            "Kaggle download failed. Make sure you have a Kaggle API token configured via "
            "~/.kaggle/kaggle.json or the KAGGLE_USERNAME/KAGGLE_KEY environment variables."
        ) from exc

    if downloaded_path.exists():
        return downloaded_path
    raise KaggleDownloadError("Kaggle reported a download path, but the path does not exist locally.")


def _sanitize_label(label: str) -> str:
    cleaned = label.strip().lower()
    cleaned = re.sub(r"[\\/]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_\-\s]+", "", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    return cleaned or "unknown"


def _find_annotation_json(root: Path) -> Path:
    candidates = sorted(root.rglob("WLASL_v0.3.json"))
    if candidates:
        return candidates[0]

    json_candidates = sorted(root.rglob("*.json"))
    for candidate in json_candidates:
        name = candidate.name.lower()
        if "wlasl" in name:
            return candidate

    raise KaggleDownloadError("Could not find a WLASL annotation JSON file in the downloaded dataset.")


def _find_videos_root(root: Path) -> Path:
    direct_candidates = [path for path in sorted(root.rglob("videos")) if path.is_dir()]
    for candidate in direct_candidates:
        if _has_video_children(candidate):
            return candidate

    # fallback: find any directory that contains video files
    for candidate in _iter_candidate_roots(root):
        if _has_video_children(candidate):
            return candidate

    raise KaggleDownloadError("Could not find a videos directory with sign-language video files.")


def _resolve_video_path(videos_root: Path, video_id: str) -> Path | None:
    for suffix in VIDEO_SUFFIXES:
        candidate = videos_root / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate

    # fallback: recursive search by stem
    matches = [path for path in videos_root.rglob("*") if path.is_file() and path.stem == video_id]
    for match in matches:
        if match.suffix.lower() in VIDEO_SUFFIXES:
            return match

    return None


def stage_from_wlasl_metadata(downloaded_root: Path, target_root: Path) -> Path:
    target_root = ensure_dir(target_root)

    annotation_path = _find_annotation_json(downloaded_root)
    videos_root = _find_videos_root(downloaded_root)

    try:
        with annotation_path.open("r", encoding="utf-8") as handle:
            annotations = json.load(handle)
    except json.JSONDecodeError as exc:
        raise KaggleDownloadError(f"Failed to parse WLASL annotation JSON: {annotation_path}") from exc

    if not isinstance(annotations, list):
        raise KaggleDownloadError("WLASL annotation JSON did not contain the expected top-level list.")

    staged_count = 0
    missing_count = 0

    for entry in annotations:
        if not isinstance(entry, dict):
            continue

        gloss_raw = str(entry.get("gloss", "")).strip()
        if not gloss_raw:
            continue
        gloss = _sanitize_label(gloss_raw)

        instances = entry.get("instances", [])
        if not isinstance(instances, list):
            continue

        for instance in instances:
            if not isinstance(instance, dict):
                continue

            split = str(instance.get("split", "")).strip().lower()
            if split not in EXPECTED_SPLITS:
                continue

            video_id = str(instance.get("video_id", "")).strip()
            if not video_id:
                continue

            source_path = _resolve_video_path(videos_root, video_id)
            if source_path is None:
                missing_count += 1
                continue

            destination_dir = target_root / split / gloss
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination_path = destination_dir / source_path.name
            _safe_symlink_or_copy(source_path, destination_path)
            staged_count += 1

    if staged_count == 0:
        raise KaggleDownloadError(
            "The Kaggle WLASL dataset was downloaded, but no videos could be staged from the metadata."
        )

    print(f"Staged {staged_count} videos into {target_root}")
    if missing_count > 0:
        print(f"Warning: skipped {missing_count} metadata entries because the source video file was missing.")

    return target_root


def prepare_kaggle_wlasl_root(dataset_slug: str, target_root: Path, cache_root: Path | None = None) -> Path:
    target_root = Path(target_root)
    if _looks_like_ready_split_root(target_root):
        return target_root

    cache_root = Path(cache_root) if cache_root is not None else Path.home() / ".cache" / "slc_kaggle"
    downloaded_root = download_kaggle_dataset(dataset_slug=dataset_slug, cache_root=cache_root)

    try:
        ready_root = find_ready_split_root(downloaded_root)
        return stage_ready_split_root(ready_root, target_root)
    except KaggleDownloadError:
        return stage_from_wlasl_metadata(downloaded_root=downloaded_root, target_root=target_root)