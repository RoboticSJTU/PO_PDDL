# Demonstration data format

The input directory contains one folder per episode. Each episode uses either
`annotations.json` or `episode.json` and contains an initial step (`step_index`
0) followed by temporally ordered actions.

```json
{
  "episode_name": "episode001",
  "fps": 30.0,
  "instruction": "Put the cup in the drawer.",
  "video": {"file_path": "video_cam_high.mp4", "camera": "cam_high"},
  "steps": [
    {
      "step_index": 0,
      "start_time_sec": null,
      "end_time_sec": null,
      "action_text": null
    },
    {
      "step_index": 1,
      "start_time_sec": 0.0,
      "end_time_sec": 3.5,
      "action_text": "open the drawer"
    }
  ]
}
```

## Video layouts

For a single camera, declare `video.file_path` or place a discoverable
`video_cam_high.mp4` in the episode. Named multi-camera files use the
`video_<camera>.mp4` convention, for example `video_cam_high.mp4` and
`video_right.mp4`. Select and order views with `--annotation-video-types`.

## Pre-extracted frames

An episode can be distributed without video by declaring a frame manifest:

```json
"pre_extracted_frames": {"manifest_path": "frames/frame_manifest.json"}
```

The manifest contains `steps`, each with `step_index`, non-empty `frame_paths`,
and optional `sample_timestamps_sec`. Paths may be relative to the episode or
manifest directory, but must stay inside the episode directory. The pipeline
uses action text and time boundaries from the annotation file and images from
the manifest, and validates that every annotated step is represented.

## Problem-generation input

The problem pipeline accepts a single image or a directory of images. Directory
images are sorted by filename and stitched vertically. Use filenames that make
the camera order explicit, such as `01_camera_high.jpg` and `02_camera_right.jpg`.
