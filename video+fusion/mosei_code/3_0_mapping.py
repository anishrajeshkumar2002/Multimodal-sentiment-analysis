import pandas as pd
import os

# Directory containing the mp4 files
video_dir = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/FaceCropped_LibreFace_mp4"

# Collect all .mp4 filenames in the directory
existing_mp4_filenames = set(os.listdir(video_dir))

# Input CSV file paths
csv_paths = [
    "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/5000_batch_raw.csv",
    "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/Batch_2980374_batch_results.csv",
    "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/extreme_sentiment_results.csv"
]

columns_to_keep = ['Input.VIDEO_ID', 'Input.CLIP', 'Answer.sentiment']
dfs = []

for path in csv_paths:
    df = pd.read_csv(path, low_memory=False)
    df_filtered = df[columns_to_keep].copy()
    df_filtered['mp4_filename'] = df_filtered.apply(
        lambda row: f"{row['Input.VIDEO_ID']}_{row['Input.CLIP']}.mp4", axis=1
    )
    df_existing = df_filtered[df_filtered['mp4_filename'].isin(existing_mp4_filenames)]
    dfs.append(df_existing)

# Combine all valid data
combined_df = pd.concat(dfs, ignore_index=True)

# Output file
output_csv = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/sentiment_mapped_mp4.csv"
combined_df.to_csv(output_csv, index=False)

print(f"Filtered sentiment annotations saved to: {output_csv}")
