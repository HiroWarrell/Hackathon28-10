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
  
#%%  
import pandas as pd

# Load person data
person_df = pd.read_csv("Dataset/Training/la_mmh_training_person_dim.csv")

# Filter out students who were never deregistered
dereg_df = person_df.dropna(subset=["deregistration_date"]).copy()
dereg_df["deregistration_date"] = pd.to_datetime(dereg_df["deregistration_date"])

# 1. Check LMS Activity
lms_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_lms_item_activity_by_session_fact.csv",
    usecols=["P_ID", "DATE"],
)
lms_df["DATE"] = pd.to_datetime(lms_df["DATE"])

lms_merged = lms_df.merge(dereg_df, on="P_ID", how="inner")
undead_lms = lms_merged[lms_merged["DATE"] > lms_merged["deregistration_date"]][
    "P_ID"
].unique()

# 2. Check Video Activity
video_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_video_activity_by_view_fact.csv",
    usecols=["P_ID", "DATE"],
)
video_df["DATE"] = pd.to_datetime(video_df["DATE"])

video_merged = video_df.merge(dereg_df, on="P_ID", how="inner")
undead_video = video_merged[
    video_merged["DATE"] > video_merged["deregistration_date"]
]["P_ID"].unique()

# Combine unique P_IDs for The Undead Student
undead_students = sorted(list(set(list(undead_lms) + list(undead_video))))
print(f"👻 The Undead Student - Found {len(undead_students)} students.")
print(undead_students[:10])  # Print first 10

import numpy as np
import pandas as pd

# Load LMS and video timestamps
lms_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_lms_item_activity_by_session_fact.csv",
    usecols=["P_ID", "DATE", "TIMESTAMP"],
)
video_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_video_activity_by_view_fact.csv",
    usecols=["P_ID", "DATE", "TIMESTAMP"],
)

# Combine and convert to full datetime
for df in [lms_df, video_df]:
  df["DATETIME"] = pd.to_datetime(df["DATE"] + " " + df["TIMESTAMP"])

combined_activity = (
    pd.concat(
        [lms_df[["P_ID", "DATETIME"]], video_df[["P_ID", "DATETIME"]]]
    )
    .dropna()
    .sort_values(["P_ID", "DATETIME"])
)

zombie_pids = []

# Group by student to check gaps
for p_id, group in combined_activity.groupby("P_ID"):
  timestamps = group["DATETIME"].values
  if len(timestamps) < 2:
    continue

  # Calculate differences between consecutive timestamps in seconds
  diffs = np.diff(timestamps) / np.timedelta64(1, "s")

  # Find blocks where gap <= 900 seconds (15 minutes)
  # We can track continuous blocks
  current_duration = 0
  max_duration = 0

  for d in diffs:
    if d <= 900:
      current_duration += d
    else:
      current_duration = 0
    if current_duration > max_duration:
      max_duration = current_duration

  # 3 hours = 10,800 seconds
  if max_duration >= 10800:
    zombie_pids.append(p_id)

print(f"🧟 The Zombie - Found {len(zombie_pids)} students.")
print(zombie_pids[:10])
#%%
import numpy as np
import pandas as pd

# Load LMS and video data
lms_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_lms_item_activity_by_session_fact.csv",
    usecols=["P_ID", "TIMESTAMP"],
)
video_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_video_activity_by_view_fact.csv",
    usecols=["P_ID", "TIMESTAMP"],
)

# Convert TIMESTAMP directly to datetime (handling ISO strings cleanly)
for df in [lms_df, video_df]:
  df["DATETIME"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)

# 1. THE ZOMBIE DETECTION
combined_activity = (
    pd.concat(
        [lms_df[["P_ID", "DATETIME"]], video_df[["P_ID", "DATETIME"]]]
    )
    .dropna()
    .sort_values(["P_ID", "DATETIME"])
)

zombie_pids = []
for p_id, group in combined_activity.groupby("P_ID"):
  timestamps = group["DATETIME"].values
  if len(timestamps) < 2:
    continue

  # Calculate differences in seconds
  diffs = np.diff(timestamps).astype("float64") / 1e9  # nanoseconds to seconds

  current_duration = 0
  max_duration = 0
  for d in diffs:
    if d <= 900:  # 15 minutes = 900 seconds
      current_duration += d
    else:
      current_duration = 0
    if current_duration > max_duration:
      max_duration = current_duration

  # 3 hours = 10,800 seconds
  if max_duration >= 10800:
    zombie_pids.append(p_id)

print(f"🧟 The Zombie - Found {len(zombie_pids)} students.")
print(zombie_pids[:10])


# 2. THE WEB CRAWLER DETECTION (> 200 items in any 12-hour block)
lms_crawler = pd.read_csv(
    "Dataset/Training/la_mmh_training_lms_item_activity_by_session_fact.csv",
    usecols=["P_ID", "TIMESTAMP", "LMS_ITEM_ID"],
).rename(columns={"LMS_ITEM_ID": "ITEM_ID"})
video_crawler = pd.read_csv(
    "Dataset/Training/la_mmh_training_video_activity_by_view_fact.csv",
    usecols=["P_ID", "TIMESTAMP", "VIDEO_ID"],
).rename(columns={"VIDEO_ID": "ITEM_ID"})

for df in [lms_crawler, video_crawler]:
  df["DATETIME"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)

crawler_df = (
    pd.concat([lms_crawler, video_crawler])
    .dropna(subset=["DATETIME"])
    .sort_values(["P_ID", "DATETIME"])
)

crawler_pids = []
window_ns = 12 * 3600 * 1e9  # 12 hours in nanoseconds

for p_id, group in crawler_df.groupby("P_ID"):
  times = group["DATETIME"].values.astype("int64")
  found = False
  for i in range(len(times)):
    count = np.sum((times >= times[i]) & (times <= times[i] + window_ns))
    if count > 200:
      found = True
      break
  if found:
    crawler_pids.append(p_id)

print(f"🕷️ The Web Crawler - Found {len(crawler_pids)} students.")
print(crawler_pids[:10])

#%%
import numpy as np
import pandas as pd

# Load timestamps for timestamp-based calculations
lms_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_lms_item_activity_by_session_fact.csv",
    usecols=["P_ID", "TIMESTAMP"],
)
video_df = pd.read_csv(
    "Dataset/Training/la_mmh_training_video_activity_by_view_fact.csv",
    usecols=["P_ID", "TIMESTAMP"],
)

for df in [lms_df, video_df]:
  df["DATETIME"] = pd.to_datetime(df["TIMESTAMP"], errors="coerce", utc=True)
  df["HOUR"] = df["DATETIME"].dt.hour
  df["DATE_ONLY"] = df["DATETIME"].dt.date

combined_activity = pd.concat(
    [lms_df[["P_ID", "DATETIME", "HOUR", "DATE_ONLY"]], video_df[["P_ID", "DATETIME", "HOUR", "DATE_ONLY"]]]
).dropna(subset=["DATETIME"])


# ==========================================
# 3. THE VAMPIRE (More night than day activity, > 21 night days)
# Night: > midnight (00:00) to 5 AM (05:00) -> HOUR in [0, 1, 2, 3, 4]
# Day: > 5 AM to midnight -> HOUR >= 5
# ==========================================
combined_activity["IS_NIGHT"] = combined_activity["HOUR"].between(0, 4, inclusive="left")

# Count unique active days (night vs day) per student
night_days = (
    combined_activity[combined_activity["IS_NIGHT"]]
    .groupby("P_ID")["DATE_ONLY"]
    .nunique()
)
total_night_days = night_days[night_days > 21]

vampire_pids = []
for p_id in total_night_days.index:
  p_acts = combined_activity[combined_activity["P_ID"] == p_id]
  n_days = p_acts[p_acts["IS_NIGHT"]]["DATE_ONLY"].nunique()
  d_days = p_acts[~p_acts["IS_NIGHT"]]["DATE_ONLY"].nunique()
  if n_days > d_days:
    vampire_pids.append(p_id)

print(f"🧛 The Vampire - Found {len(vampire_pids)} students.")


# ==========================================
# 4. THE GHOST (Active in at least 3 of first 4 weeks, then <10% later)
# ==========================================
# Load person registration dates to anchor Autumn term start
person_dim = pd.read_csv("Dataset/Training/la_mmh_training_person_dim.csv")
person_dim["registration_date"] = pd.to_datetime(person_dim["registration_date"])

ghost_pids = []
# Group activity by student and week relative to registration or fixed autumn dates
# Simplified approach: group by student and week number of the year
combined_activity["WEEK"] = combined_activity["DATETIME"].dt.isocalendar().week

for p_id, group in combined_activity.groupby("P_ID"):
  # Get student registration week
  reg_row = person_dim[person_dim["P_ID"] == p_id]
  if reg_row.empty or pd.isna(reg_row["registration_date"].values[0]):
    continue
  reg_date = reg_row["registration_date"].values[0]
  
  # Weeks since registration
  group["DAYS_SINCE_REG"] = (group["DATETIME"] - reg_date).dt.days
  group["WEEK_NUM"] = group["DAYS_SINCE_REG"] // 7

  early_weeks = group[group["WEEK_NUM"].between(0, 3)]
  early_active_weeks = early_weeks["WEEK_NUM"].nunique()

  if early_active_weeks >= 3:
    early_avg = len(early_weeks) / 4.0
    later_weeks = group[group["WEEK_NUM"] > 3]
    total_later_weeks = max(1, (later_weeks["WEEK_NUM"].max() - 3) if not later_weeks.empty else 1)
    later_avg = len(later_weeks) / total_later_weeks if not later_weeks.empty else 0

    if later_avg < (0.1 * early_avg):
      ghost_pids.append(p_id)

print(f"👻 The Ghost - Found {len(ghost_pids)} students.")


# ==========================================
# 5. THE SHAPE-SHIFTER (Switch programme during the year)
# Check module instance fact or programme fields if available in person/module table
# ==========================================
mod_fact = pd.read_csv("Dataset/Training/la_mmh_training_person_module_instance_fact.csv")
# Check if students appear across different modules or instances indicating switches
shape_shifter_pids = []
# If module instances imply different programmes or courses:
p_mods = mod_fact.groupby("P_ID")["MODULE_INSTANCE_ID"].nunique()
# Students with high module variety or explicit flags can be flagged
# Let's inspect unique module counts or check if metadata has program columns
print(f"🔄 Module Fact loaded. Unique students checked for Shape-Shifter.")


# ==========================================
# 6. THE DOPPELGÄNGER (Suspiciously identical activity pairs)
# ==========================================
# Hash student timestamp sequences or activity signatures to find identical pairs
student_activity_signatures = combined_activity.groupby("P_ID")["TIMESTAMP"].apply(lambda x: tuple(sorted(x)))
# Find duplicates in signatures
sig_counts = student_activity_signatures.value_counts()
duplicate_sigs = sig_counts[sig_counts > 1]
doppelganger_pids = student_activity_signatures[student_activity_signatures.isin(duplicate_sigs.index)].index.tolist()

print(f"👥 The Doppelgänger - Found {len(doppelganger_pids)} students in matching pairs.")