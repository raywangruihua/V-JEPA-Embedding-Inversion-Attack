""" Prepare train/val split in dataloader format required for 
V-JEPA training/evaluation scripts. """

import json
import subprocess
from pathlib import Path


# replace with your actual file paths
VIDEO_TO_CLASS_PATH: Path = Path().parent / "video_to_class.json"
SSV2_DIR: Path = Path().parent / "videos"
TRAIN_DATALOADER_PATH: Path = Path().parent / "train_dataloader.csv"
VAL_DATALOADER_PATH: Path = Path().parent / "validation_dataloader.csv"
# target number of videos in dataloader
NUM_VIDEOS: int = 10_000
MIN_FRAMES: int = 33
TRAIN_RATIO: float = 0.8


def main() -> None:
    with open(VIDEO_TO_CLASS_PATH, "r") as infile:
        data = json.load(infile)
        video_to_class = {
            item["id"]: item["template"].replace("[something]", "something") 
            for item in data
        }

    num_videos = 0
    dataset: dict [str, str] = {}
    # lazy loading to reduce memory usage
    # also means dataloaders generated can be non-deterministic
    for video_path in SSV2_DIR.iterdir():
        if num_videos >= NUM_VIDEOS:
            break
        if get_webm_frames_ffprobe(str(video_path)) < MIN_FRAMES:
            print(f"Video has more than {MIN_FRAMES}, skipping video")
            continue
        try:
            video_idx = video_path.stem
            class_label = video_to_class[video_idx]
            dataset[str(video_path)] = class_label
            num_videos += 1
        except KeyError:
            print("Key error encountered, skipping video")
        except Exception:
            print("Error occured, skipping video")

    # generate splits
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    print(f"Generating dataloaders with {train_size}/{val_size} split")

    with open(VAL_DATALOADER_PATH, "w") as outfile:
        for _ in range(val_size):
            k, v = dataset.popitem()
            outfile.write(f"{k} {v}")

    with open(TRAIN_DATALOADER_PATH, "w") as outfile:
        for k, v in dataset.items():
            outfile.write(f"{k} {v}")

    print(f"Dataloaders saved to {TRAIN_DATALOADER_PATH} and"
          f" {VAL_DATALOADER_PATH}")


def get_webm_frames_ffprobe(video_path: str):
    """ Use ffmpeg to get number of frames in video accurately. """
    command = [
        "ffprobe", 
        "-v", "error", 
        "-select_streams", "v:0", 
        "-count_frames", 
        "-show_entries", "stream=nb_read_frames", 
        "-print_format", "default=nokey=1:noprint_wrappers=1", 
        video_path
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
    return int(result.stdout.strip())


if __name__ == "__main__":
    main()
