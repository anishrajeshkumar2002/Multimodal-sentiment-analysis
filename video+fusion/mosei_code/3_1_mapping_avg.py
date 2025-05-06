import pandas as pd

# Load the CSV
csv_path = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/sentiment_mapped_mp4.csv"
df = pd.read_csv(csv_path)

# Convert sentiment to numeric (in case it's read as string)
df['Answer.sentiment'] = pd.to_numeric(df['Answer.sentiment'], errors='coerce')

# Group by mp4 filename and compute the mean sentiment
averaged_df = df.groupby('mp4_filename', as_index=False)['Answer.sentiment'].mean()

# Rename column for clarity
averaged_df.rename(columns={'Answer.sentiment': 'AverageSentiment'}, inplace=True)

# Save the result
output_path = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI/Labels/averaged_sentiment.csv"
averaged_df.to_csv(output_path, index=False)

print(f"Averaged sentiment scores saved to: {output_path}")
