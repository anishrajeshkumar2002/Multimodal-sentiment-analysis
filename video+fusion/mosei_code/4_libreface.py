import os
import cv2
import pandas as pd  # Required for DataFrame type checking
from tqdm import tqdm
from libreface import get_frames_from_video_ffmpeg, get_aligned_video_frames

input_root = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Segmented_5fps"
output_root = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/FaceCropped_LibreFace_mp4"
os.makedirs(output_root, exist_ok=True)

def process_video(input_path, output_path):
    try:
        frames = get_frames_from_video_ffmpeg(input_path)
        if frames is None or len(frames) == 0:
            print(f"❌ No frames found in: {input_path}")
            return

        result = get_aligned_video_frames(frames)

        if isinstance(result, tuple):
            aligned_frames = result[0]
        else:
            aligned_frames = result

        if not aligned_frames or not isinstance(aligned_frames, list):
            print(f"⚠️ Invalid aligned frames for: {input_path}")
            return

        if isinstance(aligned_frames[0], pd.DataFrame):
            print(f"⚠️ Skipping DataFrame returned for: {input_path}")
            return

        if isinstance(aligned_frames[0], list):
            aligned_frames = [f for sublist in aligned_frames for f in sublist]

        aligned_images = [
            cv2.imread(p) for p in aligned_frames
            if isinstance(p, str) and p.strip() and os.path.exists(p)
        ]
        aligned_images = [img for img in aligned_images if img is not None]

        if not aligned_images:
            print(f"⚠️ No aligned images loaded for {input_path}")
            return

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0 or fps > 120:
            fps = 25

        height, width = aligned_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in aligned_images:
            writer.write(frame)
        writer.release()

        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {input_path}: {type(e).__name__}: {e}")

def process_all_videos():
    input_files = set(f for f in os.listdir(input_root) if f.endswith(".mp4"))
    output_files = set(os.listdir(output_root))
    files_to_process = input_files - output_files  # Only process new files

    for fname in tqdm(sorted(files_to_process), desc="Extracting aligned face videos"):
        input_path = os.path.join(input_root, fname)
        output_path = os.path.join(output_root, fname)
        process_video(input_path, output_path)

if __name__ == "__main__":
    process_all_videos()



'''import os
import cv2
from tqdm import tqdm
from libreface import get_frames_from_video_ffmpeg, get_aligned_video_frames

input_root = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Segmented_5fps"
output_root = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/FaceCropped_LibreFace_mp4"
os.makedirs(output_root, exist_ok=True)

def process_video(input_path, output_path):
    try:
        frames = get_frames_from_video_ffmpeg(input_path)
        if frames is None or len(frames) == 0:
            print(f"❌ No frames found in: {input_path}")
            return

        result = get_aligned_video_frames(frames)

        if isinstance(result, tuple):
            aligned_frames = result[0]
        else:
            aligned_frames = result

        # ⛔ Skip if LibreFace returned DataFrame (error case)
        if isinstance(aligned_frames, list) and isinstance(aligned_frames[0], pd.DataFrame):
            print(f"⚠️ Skipping DataFrame returned for: {input_path}")
            return

        if isinstance(aligned_frames[0], list):
            aligned_frames = [f for sublist in aligned_frames for f in sublist]

        aligned_images = [cv2.imread(p) for p in aligned_frames if isinstance(p, str)]
        aligned_images = [img for img in aligned_images if img is not None]

        if not aligned_images:
            print(f"⚠️ No aligned images loaded for {input_path}")
            return

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if fps <= 0 or fps > 120:
            fps = 25

        height, width = aligned_images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in aligned_images:
            writer.write(frame)
        writer.release()

        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")

def process_all_videos():
    video_files = [f for f in os.listdir(input_root) if f.endswith(".mp4")]
    for fname in tqdm(video_files, desc="Extracting aligned face videos"):
        input_path = os.path.join(input_root, fname)
        output_path = os.path.join(output_root, fname)
        if os.path.exists(output_path):
            continue
        process_video(input_path, output_path)

if __name__ == "__main__":
    import pandas as pd  # Required for DataFrame type checking
    process_all_videos()
'''