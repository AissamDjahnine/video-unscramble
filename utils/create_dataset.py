import os
import glob
import random
import cv2
import numpy as np
from PIL import Image

VIDEO_DIR = "/home/aissam/Downloads/video_uns/videos"
IMAGES_DIR = "/home/aissam/Downloads/video_uns/images"
NUM_INSERTS = 14
SHUFFLE_STRENGTH = 0.9

def partial_shuffle(frames, strength):
    """
    Partially shuffle a list of frames based on a given strength factor.

    Args:
        frames: List of video frames to be shuffled in-place.
        strength: Float between 0 and 1 indicating shuffle intensity.

    Returns:
        None. The list is modified in-place.
    """
    if strength <= 0:
        return
    n = len(frames)
    if strength >= 1:
        random.shuffle(frames)
        return
    for i in range(n):
        if random.random() < strength:
            j = random.randint(0, n-1)
            frames[i], frames[j] = frames[j], frames[i]

def tamper_video_with_insertions(video_path, insert_frames, shuffle_strength):
    """
    Insert images into a video, shuffle the frames, and save as a tampered version.

    Args:
        video_path: Path to the input video file.
        insert_frames: List of images (as frames) to be inserted.
        shuffle_strength: Degree of frame shuffle (0=no shuffle, 1=full shuffle).

    Returns:
        None. Saves tampered video to disk.
    """
    print(f"Processing: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    for ins in insert_frames:
        pos = random.randint(0, len(frames))
        frames.insert(pos, ins)

    partial_shuffle(frames, shuffle_strength)

    base, _ = os.path.splitext(video_path)
    out_path = f"{base}_tampered.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    if not out.isOpened():
        raise IOError("Cannot open VideoWriter.")
    for f in frames:
        out.write(f)
    out.release()
    print(f"Saved tampered video: {out_path}\n")

def load_insert_frames(image_dir, num_inserts, target_size):
    """
    Load and resize a specified number of images for insertion.

    Args:
        image_dir: Directory containing image files.
        num_inserts: Number of images to load.
        target_size: Tuple (width, height) to resize each image.

    Returns:
        List of BGR images resized to target dimensions.
    """
    img_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        img_paths.extend(glob.glob(os.path.join(image_dir, ext)))
    if len(img_paths) < num_inserts:
        raise ValueError(f"Need at least {num_inserts} images in {image_dir}")

    selected_paths = random.sample(img_paths, num_inserts)
    insert_frames = []
    for img_path in selected_paths:
        with Image.open(img_path) as pil_img:
            pil_img = pil_img.convert("RGB")
            pil_img = pil_img.resize(target_size, Image.LANCZOS)
            img_np = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            insert_frames.append(img_bgr)
    return insert_frames

def process_all_videos(video_dir, image_dir, num_inserts, shuffle_strength):
    """
    Process all videos in a folder by tampering them with inserted frames.

    Args:
        video_dir: Directory containing videos to tamper.
        image_dir: Directory containing images to insert.
        num_inserts: Number of images to insert per video.
        shuffle_strength: Degree of shuffle to apply to the frames.

    Returns:
        None
    """
    for video_path in glob.glob(os.path.join(video_dir, "*.mp4")):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError("Cannot open video file.")
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            insert_frames = load_insert_frames(image_dir, num_inserts, (width, height))
            tamper_video_with_insertions(video_path, insert_frames, shuffle_strength)

        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            try:
                os.remove(video_path)
                print(f"Deleted corrupt or unreadable video: {video_path}\n")
            except Exception as rm_err:
                print(f"Failed to delete {video_path}: {rm_err}\n")
    print("All videos processed.")

if __name__ == "__main__":
    process_all_videos(VIDEO_DIR, IMAGES_DIR, NUM_INSERTS, SHUFFLE_STRENGTH)