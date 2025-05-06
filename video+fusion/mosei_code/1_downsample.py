import os
import pandas as pd
import subprocess
from tqdm import tqdm

# Configuration
csv_path = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/majority_vote_sentiment.csv"
input_dir = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Combined"
output_dir = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Segmented_5fps"
target_fps = 5

os.makedirs(output_dir, exist_ok=True)

# Load filenames from CSV
df = pd.read_csv(csv_path)
video_files = df["mp4_filename"].unique()

# Process each video
for filename in tqdm(video_files, desc="Downsampling videos to 5 FPS"):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    if not os.path.exists(input_path):
        print(f"⚠️ Skipping missing: {input_path}")
        continue

    if os.path.exists(output_path):
        print(f"✅ Already processed: {output_path}")
        continue

    print(f"🔄 Processing: {filename}")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-r", str(target_fps),
        "-c:v", "mpeg4", "-q:v", "5",
        "-c:a", "copy",
        output_path
    ]


    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to downsample: {filename}")
        print(f"🔧 stderr:\n{e.stderr}")
        print(f"🔧 stdout:\n{e.stdout}")
