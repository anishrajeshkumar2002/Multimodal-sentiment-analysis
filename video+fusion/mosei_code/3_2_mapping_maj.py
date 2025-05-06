import pandas as pd
from collections import Counter

# Load the dataset
csv_path = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Labels/sentiment_mapped_mp4.csv"
df = pd.read_csv(csv_path)

# Ensure sentiment is float
df['Answer.sentiment'] = pd.to_numeric(df['Answer.sentiment'], errors='coerce')

# Function to resolve sentiment by majority vote (with tie-breaking by average)
def resolve_sentiment(sentiments):
    counter = Counter(sentiments)
    most_common = counter.most_common()
    
    # Check if there's a tie
    max_count = most_common[0][1]
    candidates = [val for val, count in most_common if count == max_count]
    
    if len(candidates) == 1:
        return candidates[0]  # clear winner
    else:
        # tie: use the average of all sentiments in this group
        return round(sum(sentiments) / len(sentiments), 2)

# Group by mp4_filename and apply custom resolver
result_df = df.groupby('mp4_filename')['Answer.sentiment'].apply(lambda x: resolve_sentiment(list(x))).reset_index()
result_df.rename(columns={'Answer.sentiment': 'FinalSentiment'}, inplace=True)

# Save the result
output_path = "/data/home/huixian/Documents/Homeworks/535_project/MOSEI-Seg/Labels/majority_vote_sentiment.csv"
result_df.to_csv(output_path, index=False)

print(f"Majority-voted sentiment scores saved to: {output_path}")
