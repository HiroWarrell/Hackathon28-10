import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# DATA COLAB MINI MONSTER HACK — TEST 1 DETECTOR
# ============================================================
# Reads the unseen Test 1 data and writes:
#   monster_hackathon_test_submission.csv
#
# Output columns:
#   Monster,P_ID
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = BASE_DIR / "monster_hackathon_test_submission.csv"

# Use the official Autumn-term start date if your organisers supplied it.
# The event clue says "first 4 weeks of the Autumn term"; this is only
# a fallback because the supplied files do not include calendar_dim.
AUTUMN_START = "2023-09-25"

MONSTER_ORDER = [
    "The Phantom", "The Ghost", "The Zombie", "The Vampire",
    "The Undead Student", "The Doppelgänger", "The Werewolf",
    "The Web Crawler", "The Shape-Shifter",
]


def utc_datetime(values):
    return pd.to_datetime(values, errors="coerce", utc=True)


def find_file(filename):
    candidates = [
        BASE_DIR / "Dataset" / "Test" / filename,
        BASE_DIR / "Datasets" / "Test" / filename,
        BASE_DIR / "Dataset" / "test" / filename,
        BASE_DIR / "Datasets" / "test" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = [
        path for path in BASE_DIR.rglob("*")
        if path.is_file() and path.name.lower() == filename.lower()
    ]
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename}. Searched Dataset/Test and Datasets/Test "
        f"under {BASE_DIR}."
    )


def load_csv(filename):
    path = find_file(filename)
    return pd.read_csv(path), path


def duplicate_pids(df, source_id):
    # Exact duplicate activity rows, ignoring only the technical source ID.
    ignored = {source_id, "DATETIME", "DATE_ONLY", "HOUR", "IS_NIGHT"}
    key = [column for column in df.columns if column not in ignored]
    if not key:
        return set()
    duplicated = df.duplicated(subset=key, keep=False)
    return set(df.loc[duplicated, "P_ID"].dropna())


def make_activity(lms, video):
    return pd.concat(
        [lms[["P_ID", "DATETIME"]], video[["P_ID", "DATETIME"]]],
        ignore_index=True,
    ).dropna(subset=["P_ID", "DATETIME"])


def detect_shape_shifters(mod_fact, module_dim):
    # The four supplied files do not include a programme field. If an
    # optional module_instance_dim is present, use it when possible.
    if module_dim is None or "MODULE_INSTANCE_ID" not in module_dim.columns:
        return []

    programme_names = [
        "PROGRAMME", "PROGRAMME_CODE", "PROGRAMME_NAME",
        "COURSE", "COURSE_CODE", "COURSE_NAME", "PATHWAY",
    ]
    programme_col = next(
        (column for column in programme_names if column in module_dim.columns),
        None,
    )
    if programme_col is None:
        return []

    joined = mod_fact.merge(
        module_dim[["MODULE_INSTANCE_ID", programme_col]],
        on="MODULE_INSTANCE_ID",
        how="left",
    )
    values = joined[programme_col].astype("string")
    joined = joined[values.notna() & values.ne("")]
    counts = joined.groupby("P_ID")[programme_col].nunique()
    return sorted(counts[counts > 1].index.tolist())


# ------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------
print("=" * 70)
print("DATA COLAB MINI MONSTER HACK — TEST 1 DETECTOR")
print("=" * 70)

file_names = {
    "lms": "la_mmh_test_lms_item_activity_by_session_fact.csv",
    "video": "la_mmh_test_video_activity_by_view_fact.csv",
    "person": "la_mmh_test_person_dim.csv",
    "module": "la_mmh_test_person_module_instance_fact.csv",
}

lms_df, lms_path = load_csv(file_names["lms"])
video_df, video_path = load_csv(file_names["video"])
person_dim, person_path = load_csv(file_names["person"])
mod_fact, module_path = load_csv(file_names["module"])

# Optional dimensions, if supplied alongside the four required files.
try:
    module_dim, module_dim_path = load_csv("la_mmh_test_module_instance_dim.csv")
except FileNotFoundError:
    module_dim, module_dim_path = None, None

print("\nFiles loaded:")
for label, path in [
    ("lms", lms_path), ("video", video_path),
    ("person", person_path), ("module", module_path),
]:
    print(f"  {label:7}: {path}")
if module_dim_path:
    print(f"  module_dim: {module_dim_path}")

for df in (lms_df, video_df):
    if "TIMESTAMP" not in df.columns:
        raise KeyError("Both activity files must contain TIMESTAMP.")
    df["DATETIME"] = utc_datetime(df["TIMESTAMP"])
    df["DATE_ONLY"] = df["DATETIME"].dt.date
    df["HOUR"] = df["DATETIME"].dt.hour

person_dim["registration_date"] = utc_datetime(person_dim["registration_date"])
person_dim["deregistration_date"] = utc_datetime(person_dim["deregistration_date"])
activity = make_activity(lms_df, video_df).sort_values(["P_ID", "DATETIME"])
autumn_start = pd.Timestamp(AUTUMN_START, tz="UTC")

print(f"\nAutumn start used: {autumn_start.date()}")
print(f"LMS rows: {len(lms_df):,}")
print(f"Video rows: {len(video_df):,}")


# ------------------------------------------------------------
# 2. Phantom — duplicate LMS/video activity rows
# ------------------------------------------------------------
phantom_pids = sorted(
    duplicate_pids(lms_df, "PIT_ID") | duplicate_pids(video_df, "PVT_ID")
)
print(f"1. The Phantom: {len(phantom_pids)} students")


# ------------------------------------------------------------
# 3. Ghost — active in at least 3 of first 4 Autumn weeks,
#    then later average weekly activity is below 10% of early average
# ------------------------------------------------------------
activity["TERM_WEEK"] = (
    (activity["DATETIME"] - autumn_start).dt.days // 7
).astype("Int64")

early = activity[activity["TERM_WEEK"].between(0, 3)]
later = activity[activity["TERM_WEEK"] >= 4]
early_counts = early.groupby(["P_ID", "TERM_WEEK"]).size().rename("COUNT").reset_index()
later_counts = later.groupby(["P_ID", "TERM_WEEK"]).size().rename("COUNT").reset_index()
last_week = int(activity["TERM_WEEK"].max()) if not activity.empty else 4
later_week_count = max(1, last_week - 3)

ghost_pids = []
for p_id, group in early_counts.groupby("P_ID"):
    if group["TERM_WEEK"].nunique() < 3:
        continue
    early_average = group["COUNT"].sum() / 4.0
    later_total = later_counts.loc[later_counts["P_ID"].eq(p_id), "COUNT"].sum()
    later_average = later_total / later_week_count
    if later_average < 0.10 * early_average:
        ghost_pids.append(p_id)
print(f"2. The Ghost: {len(ghost_pids)} students")


# ------------------------------------------------------------
# 4. Zombie — continuous activity for more than 3 hours,
#    with no gap longer than 15 minutes
# ------------------------------------------------------------
zombie_pids = []
for p_id, group in activity.groupby("P_ID"):
    times = group["DATETIME"].drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) < 2:
        continue
    new_run = times.diff().gt(pd.Timedelta(minutes=15))
    run_id = new_run.cumsum()
    runs = pd.DataFrame({"TIME": times, "RUN": run_id}).groupby("RUN")["TIME"].agg(["min", "max"])
    if ((runs["max"] - runs["min"]) > pd.Timedelta(hours=3)).any():
        zombie_pids.append(p_id)
print(f"3. The Zombie: {len(zombie_pids)} students")


# ------------------------------------------------------------
# 5. Vampire — more than 21 night days and more night than day days
# ------------------------------------------------------------
for df in (lms_df, video_df):
    # Night is midnight through 04:59; 05:00 onward is day.
    df["IS_NIGHT"] = df["HOUR"].lt(5)

vampire_activity = pd.concat(
    [
        lms_df[["P_ID", "DATE_ONLY", "IS_NIGHT"]],
        video_df[["P_ID", "DATE_ONLY", "IS_NIGHT"]],
    ],
    ignore_index=True,
).dropna(subset=["P_ID", "DATE_ONLY"])

night_days = vampire_activity.loc[vampire_activity["IS_NIGHT"]].groupby("P_ID")["DATE_ONLY"].nunique()
day_days = vampire_activity.loc[~vampire_activity["IS_NIGHT"]].groupby("P_ID")["DATE_ONLY"].nunique()
vampire_pids = sorted(
    p_id for p_id in night_days.index
    if night_days[p_id] > 21 and night_days[p_id] > day_days.get(p_id, 0)
)
print(f"4. The Vampire: {len(vampire_pids)} students")


# ------------------------------------------------------------
# 6. Undead Student — activity after deregistration date
# ------------------------------------------------------------
reg = person_dim[["P_ID", "deregistration_date"]].dropna(subset=["deregistration_date"])
undead = activity.merge(reg, on="P_ID", how="inner")
undead_pids = sorted(
    undead.loc[undead["DATETIME"] > undead["deregistration_date"], "P_ID"].unique()
)
print(f"5. The Undead Student: {len(undead_pids)} students")


# ------------------------------------------------------------
# 7. Doppelgänger — matching complete activity signatures
# ------------------------------------------------------------
lms_sig = pd.DataFrame({
    "P_ID": lms_df["P_ID"],
    "DATETIME": lms_df["DATETIME"].dt.floor("min"),
    "SOURCE": "LMS",
    "ITEM": lms_df["LMS_ITEM_ID"].astype("string"),
})
video_sig = pd.DataFrame({
    "P_ID": video_df["P_ID"],
    "DATETIME": video_df["DATETIME"].dt.floor("min"),
    "SOURCE": "VIDEO",
    "ITEM": video_df["VIDEO_ID"].astype("string"),
})
sig = pd.concat([lms_sig, video_sig], ignore_index=True).dropna(subset=["P_ID", "DATETIME"])
sig = sig.sort_values(["P_ID", "DATETIME", "SOURCE", "ITEM"])

signatures = (
    sig.groupby("P_ID", sort=False)
    .apply(
        lambda x: tuple(
            map(tuple, x[["DATETIME", "SOURCE", "ITEM"]].astype(str).to_numpy())
        )
    )
    .rename("SIGNATURE")
    .reset_index()
)
doppelganger_pids = sorted(
    signatures.loc[signatures.duplicated("SIGNATURE", keep=False), "P_ID"].unique()
)
print(f"6. The Doppelgänger: {len(doppelganger_pids)} students")


# ------------------------------------------------------------
# 8. Werewolf — an activity spike in both Autumn and Spring
# ------------------------------------------------------------
weekly = activity.copy()
weekly["WEEK_START"] = (
    weekly["DATETIME"].dt.normalize()
    - pd.to_timedelta(weekly["DATETIME"].dt.weekday, unit="D")
)
weekly_counts = weekly.groupby(["P_ID", "WEEK_START"]).size().rename("COUNT").reset_index()
weekly_counts["TERM"] = np.select(
    [
        weekly_counts["WEEK_START"].dt.month.isin([9, 10, 11, 12]),
        weekly_counts["WEEK_START"].dt.month.isin([1, 2, 3, 4]),
    ],
    ["AUTUMN", "SPRING"],
    default="OTHER",
)

werewolf_pids = []
relevant = weekly_counts[weekly_counts["TERM"].isin(["AUTUMN", "SPRING"])]
for p_id, group in relevant.groupby("P_ID"):
    spike_terms = set()
    for term, term_group in group.groupby("TERM"):
        counts = term_group["COUNT"]
        if len(counts) < 3:
            continue
        threshold = max(
            counts.mean() + 3 * counts.std(ddof=0),
            counts.quantile(0.90),
        )
        if (counts > threshold).any():
            spike_terms.add(term)
    if spike_terms == {"AUTUMN", "SPRING"}:
        werewolf_pids.append(p_id)
print(f"7. The Werewolf: {len(werewolf_pids)} students")


# ------------------------------------------------------------
# 9. Web Crawler — more than 200 LMS/video rows in 12 hours
# ------------------------------------------------------------
crawler_pids = []
for p_id, group in activity.groupby("P_ID"):
    times = group["DATETIME"].sort_values().reset_index(drop=True)
    left = 0
    for right in range(len(times)):
        while times.iloc[right] - times.iloc[left] > pd.Timedelta(hours=12):
            left += 1
        if right - left + 1 > 200:
            crawler_pids.append(p_id)
            break
print(f"8. The Web Crawler: {len(crawler_pids)} students")


# ------------------------------------------------------------
# 10. Shape-Shifter — only when programme metadata is available
# ------------------------------------------------------------
shape_shifter_pids = detect_shape_shifters(mod_fact, module_dim)
if module_dim is None:
    print("WARNING: Shape-Shifter cannot be tested from the four supplied files; module_instance_dim is missing.")
print(f"9. The Shape-Shifter: {len(shape_shifter_pids)} students")


# ------------------------------------------------------------
# 11. Export submission
# ------------------------------------------------------------
results = {
    "The Phantom": sorted(set(phantom_pids)),
    "The Ghost": sorted(set(ghost_pids)),
    "The Zombie": sorted(set(zombie_pids)),
    "The Vampire": sorted(set(vampire_pids)),
    "The Undead Student": sorted(set(undead_pids)),
    "The Doppelgänger": sorted(set(doppelganger_pids)),
    "The Werewolf": sorted(set(werewolf_pids)),
    "The Web Crawler": sorted(set(crawler_pids)),
    "The Shape-Shifter": sorted(set(shape_shifter_pids)),
}

rows = [
    {"Monster": monster, "P_ID": p_id}
    for monster in MONSTER_ORDER
    for p_id in results[monster]
]
submission = pd.DataFrame(rows, columns=["Monster", "P_ID"])
if not submission.empty:
    submission = submission.drop_duplicates().sort_values(["Monster", "P_ID"]).reset_index(drop=True)
submission.to_csv(OUTPUT_FILE, index=False)

print("\n" + "=" * 70)
print(f"Saved submission: {OUTPUT_FILE}")
print(f"Total submission rows: {len(submission):,}")
print("\nRows per monster:")
print(pd.Series({monster: len(results[monster]) for monster in MONSTER_ORDER}))
print("\nPreview:")
print(submission.head(20).to_string(index=False))
