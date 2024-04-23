import pandas as pd
from sklearn.utils import resample

# Load your dataset
df = pd.read_csv('text.csv')

# Find the category with the minimum samples (to balance all categories to this size)
min_count = df['label'].value_counts().min()

# Create a new DataFrame for the balanced data
balanced_df = pd.DataFrame()

# Sample each category to match the smallest size
for label in df['label'].unique():
    # Filter by category
    category_group = df[df['label'] == label]
    # Resample the data
    resampled_group = resample(category_group, 
                               replace=False, 
                               n_samples=min_count, 
                               random_state=42)  # Random state for reproducibility
    # Append to the balanced DataFrame
    balanced_df = pd.concat([balanced_df, resampled_group])

# Shuffle the dataset to mix up rows randomly
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the balanced dataset to a new CSV file, without an index column
balanced_df.to_csv('balanced_text.csv', index=False)

print(f"Balanced dataset saved with each category having {min_count} samples.")
