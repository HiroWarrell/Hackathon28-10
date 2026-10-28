import os
import pandas as pd

# Define the training data directory path
data_dir = os.path.join("Dataset", "Training")

# List of training files provided in the hackathon
files = [
    "la_mmh_training_lms_item_activity_by_session_fact.csv",
    "la_mmh_training_person_dim.csv",
    "la_mmh_training_person_module_instance_fact.csv",
    "la_mmh_training_video_activity_by_view_fact.csv",
]

# Dictionary to store dataframes in memory
dfs = {}

print("=" * 70)
print(" DATA COLAB MINI MONSTER HACK: TRAINING DATA INSPECTION ")
print("=" * 70 + "\n")

for filename in files:
  file_path = os.path.join(data_dir, filename)
  if os.path.exists(file_path):
    print(f"📁 Loading: {filename}")
    try:
      # low_memory=False helps prevent mixed-type warnings on large CSVs
      df = pd.read_csv(file_path, low_memory=False)
      dfs[filename] = df

      print(f"   Shape (Rows, Columns): {df.shape}")
      print(f"   Columns: {list(df.columns)}")
      print("\n" + "-" * 70 + "\n")
    except Exception as e:
      print(f"   ❌ Error reading {filename}: {e}\n")
  else:
    print(f"⚠️ File not found: {file_path}\n")

# Optional: Print info or head for a specific file (e.g., person dimension)
person_file = "la_mmh_training_person_dim.csv"
if person_file in dfs:
  print(f"🔍 Preview of {person_file}:")
  print(dfs[person_file].head(3))
