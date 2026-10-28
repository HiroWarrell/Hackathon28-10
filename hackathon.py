import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# DATA COLAB MINI MONSTER HACK
# Paste this file into your notebook/script and run it.
# ============================================================

DATA_DIR = Path("Dataset/Training")
OUTPUT_FILE = "monster_hackathon_submission.csv"

# Set this manually if you know the exact Autumn-term start date.
# Leave as None to infer 1 September of the earliest activity year.
AUTUMN_START = None  # Example: "2023-09-25"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def utc_datetime(series):
    return pd.to_datetime(series, errors="coerce", utc=True)


def get_first_existing(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def infer_autumn_start(activity):
    first_date = activity["DATETIME"].min()
    if pd.isna(first_date):
        raise ValueError("No valid activity timestamps were found.")
    return pd.Timestamp(year=first_date.year, month=9, day=1, tz="UTC")


def make_activity(lms, video):
    lms_activity = lms[["P_ID", "DATETIME"]].copy()
    video_activity = video[["P_ID", "DATETIME"]].copy()
    return pd.concat([lms_activity, video_activity], ignore_index=True).dropna()


# ------------------------------------------------------------
# 1. Load datasets
# ------------------------------------------------------------
print("=" * 70)
print("DATA COLAB MINI MONSTER HACK: MONSTER DETECTION")
print("=" * 70)

lms_df = pd.read_csv(
    DATA_DIR / "la_mmh_training_lms_item_activity_by_session_fact.csv"
)
video_df = pd.read_csv(
    DATA_DIR / "la_mmh_training_video_activity_by_view_fact.csv"
)
person_dim = pd.read_csv(DATA_DIR / "la_mmh_training_person_dim.csv")
mod_fact = pd.read_csv(DATA_DIR / "la_mmh_training_person_module_instance_fact.csv")

for df in [lms_df, video_df]:
    df["DATETIME"] = utc_datetime(df["TIMESTAMP"])
    df["DATE_ONLY"] = df["DATETIME"].dt.date
    df["HOUR"] = df["DATETIME"].dt.hour

person_dim["registration_date"] = utc_datetime(person_dim["registration_date"])
person_dim["deregistration_date"] = utc_datetime(person_dim["deregistration_date"])

activity = make_activity(lms_df, video_df).sort_values(["P_ID", "DATETIME"])
autumn_start = (
    pd.Timestamp(AUTUMN_START, tz="UTC")
    if AUTUMN_START is not None
    else infer_autumn_start(activity)
)

print(f"Autumn start used: {autumn_start.date()}")
print(f"LMS rows: {len(lms_df):,}")
print(f"Video rows: {len(video_df):,}")


# ------------------------------------------------------------
# 2. THE PHANTOM: duplicate activity rows
# ------------------------------------------------------------
def duplicate_pids(df, id_column):
    # Exclude the source-system row identifier. It should not prevent
    # otherwise identical activity rows from being detected.
    subset = [c for c in df.columns if c not in {id_column, "DATETIME", "DATE_ONLY", "HOUR"}]
    # Include the normalized timestamp in the duplicate key.
    subset = [c for c in subset if c in df.columns] + ["DATETIME"]
    return set(df.loc[df.duplicated(subset=subset, keep=False), "P_ID"].dropna())


phantom_pids = sorted(
    duplicate_pids(lms_df, "PIT_ID") | duplicate_pids(video_df, "PVT_ID")
)
print(f"1. The Phantom: {len(phantom_pids)} students")


# ------------------------------------------------------------
# 3. THE GHOST: active in at least 3 of first 4 Autumn weeks,
#    then later average weekly activity is below 10% of early average
# ------------------------------------------------------------
activity["TERM_WEEK"] = ((activity["DATETIME"] - autumn_start).dt.days // 7).astype("Int64")

early = activity[activity["TERM_WEEK"].between(0, 3)].copy()
later = activity[activity["TERM_WEEK"] >= 4].copy()

early_counts = (
    early.groupby(["P_ID", "TERM_WEEK"]).size().rename("COUNT").reset_index()
)
later_counts = (
    later.groupby(["P_ID", "TERM_WEEK"]).size().rename("COUNT").reset_index()
)

ghost_pids = []
for p_id, early_group in early_counts.groupby("P_ID"):
    if early_group["TERM_WEEK"].nunique() < 3:
        continue

    early_average = early_group["COUNT"].sum() / 4.0
    later_group = later_counts[later_counts["P_ID"] == p_id]

    if later_group.empty:
        later_average = 0.0
    else:
        last_week = int(later_group["TERM_WEEK"].max())
        number_of_later_weeks = max(1, last_week - 3)
        later_average = later_group["COUNT"].sum() / number_of_later_weeks

    if later_average < 0.10 * early_average:
        ghost_pids.append(p_id)

print(f"2. The Ghost: {len(ghost_pids)} students")


# ------------------------------------------------------------
# 4. THE ZOMBIE: continuous activity for more than 3 hours,
#    with no gap longer than 15 minutes
# ------------------------------------------------------------
zombie_pids = []
for p_id, group in activity.groupby("P_ID"):
    times = group["DATETIME"].sort_values().drop_duplicates().reset_index(drop=True)
    if len(times) < 2:
        continue

    starts_new_run = times.diff().gt(pd.Timedelta(minutes=15))
    run_id = starts_new_run.cumsum()
    runs = pd.DataFrame({"TIME": times, "RUN": run_id}).groupby("RUN")["TIME"].agg(["min", "max"])

    if ((runs["max"] - runs["min"]) > pd.Timedelta(hours=3)).any():
        zombie_pids.append(p_id)

print(f"3. The Zombie: {len(zombie_pids)} students")


# ------------------------------------------------------------
# 5. THE VAMPIRE: more night days than day days and >21 night days
# ------------------------------------------------------------
for df in [lms_df, video_df]:
    df["IS_NIGHT"] = df["HOUR"].between(0, 4, inclusive="both")

vampire_activity = pd.concat(
    [
        lms_df[["P_ID", "DATE_ONLY", "IS_NIGHT"]],
        video_df[["P_ID", "DATE_ONLY", "IS_NIGHT"]],
    ],
    ignore_index=True,
).dropna(subset=["P_ID", "DATE_ONLY"])

night_days = (
    vampire_activity[vampire_activity["IS_NIGHT"]]
    .groupby("P_ID")["DATE_ONLY"]
    .nunique()
)
day_days = (
    vampire_activity[~vampire_activity["IS_NIGHT"]]
    .groupby("P_ID")["DATE_ONLY"]
    .nunique()
)

vampire_pids = sorted(
    p_id for p_id in night_days.index
    if night_days[p_id] > 21 and night_days[p_id] > day_days.get(p_id, 0)
)
print(f"4. The Vampire: {len(vampire_pids)} students")


# ------------------------------------------------------------
# 6. THE UNDEAD STUDENT: activity after deregistration
# ------------------------------------------------------------
registration = person_dim[["P_ID", "deregistration_date"]].dropna(
    subset=["deregistration_date"]
)

undead_activity = activity.merge(registration, on="P_ID", how="inner")
undead_pids = sorted(
    undead_activity.loc[
        undead_activity["DATETIME"] > undead_activity["deregistration_date"], "P_ID"
    ].dropna().unique()
)
print(f"5. The Undead Student: {len(undead_pids)} students")


# ------------------------------------------------------------
# 7. THE DOPPELGANGER: identical LMS/video activity signatures
# ------------------------------------------------------------
lms_signature = pd.DataFrame({
    "P_ID": lms_df["P_ID"],
    "DATETIME": lms_df["DATETIME"].dt.floor("min"),
    "SOURCE": "LMS",
    "ITEM": lms_df["LMS_ITEM_ID"].astype(str),
})
video_signature = pd.DataFrame({
    "P_ID": video_df["P_ID"],
    "DATETIME": video_df["DATETIME"].dt.floor("min"),
    "SOURCE": "VIDEO",
    "ITEM": video_df["VIDEO_ID"].astype(str),
})

signature_activity = pd.concat([lms_signature, video_signature], ignore_index=True).dropna(
    subset=["P_ID", "DATETIME"]
)

signatures = (
    signature_activity.sort_values(["P_ID", "DATETIME", "SOURCE", "ITEM"])
    .groupby("P_ID", sort=False)
    .apply(
        lambda x: tuple(
            map(tuple, x[["DATETIME", "SOURCE", "ITEM"]].astype(str).to_numpy())
        ),
        include_groups=False,
    )
    .rename("SIGNATURE")
    .reset_index()
)

doppelganger_pids = sorted(
    signatures.loc[signatures.duplicated("SIGNATURE", keep=False), "P_ID"].unique()
)
print(f"6. The Doppelganger: {len(doppelganger_pids)} students")


# ------------------------------------------------------------
# 8. THE WEREWOLF: repeated weekly spikes in Autumn and Spring
# ------------------------------------------------------------
weekly = activity.copy()
weekly["WEEK_START"] = weekly["DATETIME"].dt.to_period("W").dt.start_time.dt.tz_localize("UTC")
weekly_counts = (
    weekly.groupby(["P_ID", "WEEK_START"]).size().rename("COUNT").reset_index()
)

# Define Spring as January through April. This can be adjusted if the
# programme's official Spring-term dates are different.
weekly_counts["TERM"] = np.select(
    [
        weekly_counts["WEEK_START"].dt.month.isin([9, 10, 11, 12]),
        weekly_counts["WEEK_START"].dt.month.isin([1, 2, 3, 4]),
    ],
    ["AUTUMN", "SPRING"],
    default="OTHER",
)

werewolf_pids = []
for p_id, group in weekly_counts[weekly_counts["TERM"].isin(["AUTUMN", "SPRING"])].groupby("P_ID"):
    term_spikes = set()
    for term, term_group in group.groupby("TERM"):
        if len(term_group) < 3:
            continue

        mean_count = term_group["COUNT"].mean()
        std_count = term_group["COUNT"].std(ddof=0)
        threshold = max(mean_count + 3 * std_count, term_group["COUNT"].quantile(0.90))

        if (term_group["COUNT"] > threshold).any():
            term_spikes.add(term)

    if {"AUTUMN", "SPRING"}.issubset(term_spikes):
        werewolf_pids.append(p_id)

print(f"7. The Werewolf: {len(werewolf_pids)} students")


# ------------------------------------------------------------
# 9. THE WEB CRAWLER: >200 LMS/video rows in any 12-hour window
# ------------------------------------------------------------
crawler_activity = activity.sort_values(["P_ID", "DATETIME"])
crawler_pids = []
window = pd.Timedelta(hours=12)

for p_id, group in crawler_activity.groupby("P_ID"):
    times = group["DATETIME"].dropna().sort_values().reset_index(drop=True)
    if len(times) <= 200:
        continue

    left = 0
    for right in range(len(times)):
        while times.iloc[right] - times.iloc[left] > window:
            left += 1
        if right - left + 1 > 200:
            crawler_pids.append(p_id)
            break

print(f"8. The Web Crawler: {len(crawler_pids)} students")


# ------------------------------------------------------------
# 10. THE SHAPE-SHIFTER: programme changes over time
# ------------------------------------------------------------
# Locate a likely programme column automatically. Inspect the printed
# columns if this reports that no programme field exists.
programme_candidates = [
    "PROGRAMME", "PROGRAMME_CODE", "PROGRAMME_NAME", "COURSE", "COURSE_CODE",
    "COURSE_NAME", "PATHWAY", "QUALIFICATION", "PROGRAM",
]
programme_col = get_first_existing(mod_fact, programme_candidates)
programme_source = mod_fact

if programme_col is None:
    programme_col = get_first_existing(person_dim, programme_candidates)
    programme_source = person_dim

shape_shifter_pids = []
if programme_col is not None:
    print(f"Programme field used for Shape-Shifter: {programme_col}")

    # Prefer a module/activity date if available; otherwise use the row order.
    date_col = get_first_existing(
        programme_source,
        ["DATE", "MODULE_START_DATE", "START_DATE", "DATE_FROM", "EFFECTIVE_DATE"],
    )

    programme_rows = programme_source[["P_ID", programme_col]].copy()
    programme_rows["PROGRAMME_VALUE"] = programme_rows[programme_col].astype(str)
    programme_rows = programme_rows[
        programme_rows["PROGRAMME_VALUE"].notna()
        & ~programme_rows["PROGRAMME_VALUE"].isin(["nan", "None", ""])
    ]

    if date_col is not None:
        programme_rows["PROGRAMME_DATE"] = utc_datetime(programme_source.loc[programme_rows.index, date_col])
        programme_rows = programme_rows.sort_values(["P_ID", "PROGRAMME_DATE"])

    programme_counts = programme_rows.groupby("P_ID")["PROGRAMME_VALUE"].nunique()
    shape_shifter_pids = sorted(programme_counts[programme_counts > 1].index.tolist())
else:
    print("WARNING: No programme/course column found; Shape-Shifter list is empty.")
    print("mod_fact columns:", list(mod_fact.columns))
    print("person_dim columns:", list(person_dim.columns))

print(f"9. The Shape-Shifter: {len(shape_shifter_pids)} students")


# ------------------------------------------------------------
# 11. Export submission
# ------------------------------------------------------------
monster_results = {
    "The Phantom": phantom_pids,
    "The Ghost": sorted(set(ghost_pids)),
    "The Zombie": sorted(set(zombie_pids)),
    "The Vampire": vampire_pids,
    "The Undead Student": undead_pids,
    "The Doppelgänger": doppelganger_pids,
    "The Werewolf": sorted(set(werewolf_pids)),
    "The Web Crawler": sorted(set(crawler_pids)),
    "The Shape-Shifter": shape_shifter_pids,
}

submission_rows = [
    {"Monster": monster, "P_ID": p_id}
    for monster, pids in monster_results.items()
    for p_id in pids
]

submission_df = (
    pd.DataFrame(submission_rows, columns=["Monster", "P_ID"])
    .drop_duplicates()
    .sort_values(["Monster", "P_ID"])
    .reset_index(drop=True)
)

submission_df.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print(f"Saved submission: {OUTPUT_FILE}")
print(f"Total submission rows: {len(submission_df):,}")
print("\nRows per monster:")
print(submission_df.groupby("Monster").size().reindex(monster_results.keys(), fill_value=0))
print("\nPreview:")
print(submission_df.head(20).to_string(index=False))
