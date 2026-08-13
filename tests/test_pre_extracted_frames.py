import json
from pathlib import Path

from po_pddl.domain_generation.infrastructure.episode_video import (
    extract_episode_step_frames,
)


def test_frame_only_episode_uses_manifest_without_video(tmp_path: Path) -> None:
    episode = tmp_path / "episode001"
    frames = episode / "frames" / "step_000"
    frames.mkdir(parents=True)
    image = frames / "frame.jpg"
    image.write_bytes(b"not decoded by manifest loading")
    manifest_path = episode / "frames" / "frame_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "fps": 2.0,
                "image_extension": ".jpg",
                "camera_order_top_to_bottom": ["cam_high"],
                "steps": [
                    {
                        "step_index": 0,
                        "frame_paths": ["frames/step_000/frame.jpg"],
                        "sample_timestamps_sec": [0.0],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    annotation_path = episode / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "episode_name": "episode001",
                "pre_extracted_frames": {"manifest_path": "frames/frame_manifest.json"},
                "steps": [
                    {
                        "step_index": 0,
                        "start_time_sec": None,
                        "end_time_sec": None,
                        "action_text": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = extract_episode_step_frames(
        annotation_path,
        tmp_path / "output",
        fps=2.0,
    )

    assert result.video_path is None
    assert result.steps[0].frame_paths == [str(image.resolve())]
    assert (tmp_path / "output" / "frame_manifest.json").is_file()
