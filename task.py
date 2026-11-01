# ============================================================
# ELECTRIC VEHICLE POPULATION DATA
# DATA CLEANING & ANALYSIS PRACTICAL
# ============================================================

import pandas as pd
import numpy as np
from scipy.stats import zscore
import os

# ============================================================
# SETTINGS
# ============================================================

FILE_PATH = "Electric_Vehicle_Population_Data.csv"

# ============================================================
# PART A — IMPORTING & UNDERSTANDING THE DATA
# ============================================================

print("\n" + "=" * 70)
print("PART A — IMPORTING & UNDERSTANDING THE DATA")
print("=" * 70)

# Task 1 — Import dataset
print("\nTASK 1 — IMPORT DATASET")

df = pd.read_csv(FILE_PATH)

print("\nFirst 5 records:")
print(df.head())

print("\nNumber of rows and columns:")
print(df.shape)

print("\nRows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nOne row represents one vehicle registration record in the dataset.")

# Task 2 — Inspect dataset structure
print("\nTASK 2 — DATASET STRUCTURE")

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nDataset information:")
df.info()

# Automatically classify columns
categorical_columns = df.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical_columns = df.select_dtypes(
    include=["int64", "int32", "float64", "float32"]
).columns.tolist()

datetime_columns = df.select_dtypes(
    include=["datetime64[ns]", "datetime64[ns, UTC]"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

print("\nDate/Time columns:")
print(datetime_columns)

# Identifier/code columns
identifier_columns = []

for col in df.columns:
    col_lower = col.lower()

    if any(word in col_lower for word in [
        "id", "vin", "code", "number"
    ]):
        identifier_columns.append(col)

print("\nPossible identifier/code columns:")
print(identifier_columns)

# Task 3 — Understand important columns
print("\nTASK 3 — UNDERSTAND IMPORTANT COLUMNS")

for col in df.columns:
    print(f"- {col}")

print("\nIMPORTANT:")
print("Inspect the above column names and identify:")
print("- Manufacturer / Make")
print("- Vehicle Model")
print("- Electric Vehicle Type")
print("- Location columns")
print("- Electric Range")
print("- Vehicle Price / MSRP if available")
print("- Model Year")


# ============================================================
# PART B — MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("PART B — MISSING VALUES")
print("=" * 70)

# Task 4 — Missing values
print("\nTASK 4 — FIND MISSING VALUES")

missing_count_before = df.isnull().sum()

print("\nMissing values in every column:")
print(missing_count_before)

print("\nColumns containing missing values:")
print(missing_count_before[missing_count_before > 0])

if missing_count_before.sum() > 0:
    highest_missing_column = missing_count_before.idxmax()

    print(
        "\nColumn with highest number of missing values:",
        highest_missing_column
    )

    print(
        "Number of missing values:",
        missing_count_before.max()
    )
else:
    print("\nNo missing values found.")

# Task 5 — Missing percentages
print("\nTASK 5 — MISSING VALUE PERCENTAGES")

missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_summary = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage.round(2)
})

print(missing_summary)

if missing_percentage.max() > 0:
    highest_percentage_column = missing_percentage.idxmax()
    highest_percentage = missing_percentage.max()

    print(
        f"\nThe column with the highest percentage of missing "
        f"data is '{highest_percentage_column}' "
        f"with {highest_percentage:.2f}%."
    )

    if highest_percentage < 5:
        level = "Low"
    elif highest_percentage < 30:
        level = "Moderate"
    else:
        level = "High"

    print("Missing-data level:", level)

# ============================================================
# TASK 6 — DECIDE HOW TO HANDLE MISSING VALUES
# ============================================================

print("\nTASK 6 — MISSING VALUE HANDLING DECISIONS")

print("""
General decision rules:

1. Numerical columns:
   - Median can be used when missing values need to be filled.
   - Median is useful when extreme values may exist.

2. Categorical columns:
   - Mode can be used when filling is appropriate.

3. Columns with extremely high missing percentages:
   - Consider dropping the column if it is not important.

4. Rows:
   - Drop rows only when the missing record is not useful
     and the number of missing rows is small.

The actual decision should depend on the meaning of each column.
""")

# ============================================================
# TASK 7 & 8 — HANDLE MISSING VALUES
# ============================================================

print("\nTASK 7 & 8 — HANDLE NUMERICAL AND CATEGORICAL MISSING VALUES")

# Keep a copy so we can compare before/after
df_before_missing_cleaning = df.copy()

# Numerical columns
for col in numerical_columns:

    missing = df[col].isnull().sum()

    if missing > 0:

        # Use median for numerical columns
        median_value = df[col].median()

        if not pd.isna(median_value):
            df[col] = df[col].fillna(median_value)

            print(
                f"\nNumerical column '{col}':"
                f"\nMissing values before = {missing}"
                f"\nFilled using median = {median_value}"
            )

# Categorical columns
for col in categorical_columns:

    missing = df[col].isnull().sum()

    if missing > 0:

        mode_values = df[col].mode()

        if len(mode_values) > 0:

            mode_value = mode_values[0]

            df[col] = df[col].fillna(mode_value)

            print(
                f"\nCategorical column '{col}':"
                f"\nMissing values before = {missing}"
                f"\nFilled using mode = {mode_value}"
            )

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())

print("\nComparison:")
comparison = pd.DataFrame({
    "Before": df_before_missing_cleaning.isnull().sum(),
    "After": df.isnull().sum()
})

print(comparison)


# ============================================================
# PART C — DATA INCONSISTENCY
# ============================================================

print("\n" + "=" * 70)
print("PART C — DATA INCONSISTENCY")
print("=" * 70)

# Task 9 — Text consistency
print("\nTASK 9 — TEXT CONSISTENCY")

print("""
Checking text columns for:
- Leading spaces
- Trailing spaces
- Inconsistent capitalization
""")

# Standardize all object/string columns
for col in categorical_columns:

    before_unique = df[col].nunique()

    df[col] = df[col].astype(str).str.strip()

    after_unique = df[col].nunique()

    print(
        f"\nColumn: {col}"
        f"\nUnique values before standardization: {before_unique}"
        f"\nUnique values after stripping spaces: {after_unique}"
    )

# Task 10 — Investigate categorical values
print("\nTASK 10 — INVESTIGATE CATEGORICAL VALUES")

for col in categorical_columns:

    print("\n" + "-" * 50)
    print("COLUMN:", col)

    print("\nNumber of unique values:")
    print(df[col].nunique())

    print("\nFirst 20 unique values:")
    print(df[col].unique()[:20])

    print("\nMost common values:")
    print(df[col].value_counts().head(10))


# ============================================================
# TASK 11 — TYPE CONVERSION
# ============================================================

print("\nTASK 11 — TYPE CONVERSION")

print("\nData types after import:")
print(df.dtypes)

# Try to identify object columns that may contain numbers
for col in df.columns:

    if df[col].dtype == "object":

        converted = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        numeric_percentage = converted.notna().mean()

        # If more than 90% of values can be numeric,
        # convert the column.
        if numeric_percentage > 0.90:

            print(
                f"\n'{col}' appears to contain numerical data."
            )

            print("Before:", df[col].dtype)

            df[col] = converted

            print("After:", df[col].dtype)


# ============================================================
# PART D — DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("PART D — DUPLICATES")
print("=" * 70)

# Task 12
print("\nTASK 12 — IDENTIFY DUPLICATE ROWS")

duplicate_count = df.duplicated().sum()

duplicate_percentage = (
    duplicate_count / len(df)
) * 100

print("Duplicate rows:", duplicate_count)
print(
    f"Duplicate percentage: {duplicate_percentage:.4f}%"
)

# Task 13
print("\nTASK 13 — REMOVE EXACT DUPLICATES")

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        f"Removed {duplicate_count} exact duplicate rows."
    )

else:

    print(
        "No exact duplicate records were found."
        " Therefore, no records were removed."
    )

# Task 14 — Possible identifiers
print("\nTASK 14 — INVESTIGATE POSSIBLE VEHICLE IDENTIFIERS")

for col in identifier_columns:

    print("\nIdentifier column:", col)

    repeated = df[col].duplicated().sum()

    print("Repeated values:", repeated)

    print(
        "A repeated identifier does not automatically mean "
        "that the record is an error. The dataset structure "
        "and meaning of the identifier must be investigated."
    )


# ============================================================
# PART E — OUTLIERS
# ============================================================

print("\n" + "=" * 70)
print("PART E — OUTLIERS")
print("=" * 70)

print("\nTASK 15 — NUMERICAL COLUMNS")

print("Available numerical columns:")
print(numerical_columns)

# Select numerical columns with enough valid values
usable_numeric_columns = []

for col in numerical_columns:

    if df[col].notna().sum() > 0:

        usable_numeric_columns.append(col)

print("\nSuitable numerical columns:")
print(usable_numeric_columns)


# ============================================================
# TASK 16 — IQR OUTLIER DETECTION
# ============================================================

print("\nTASK 16 — IQR OUTLIER DETECTION")

# Try to find Electric Range column automatically
range_column = None

for col in df.columns:

    if "electric range" in col.lower():

        range_column = col
        break

if range_column is None:

    print(
        "\nElectric Range column was not automatically found."
    )

    if len(usable_numeric_columns) > 0:

        range_column = usable_numeric_columns[0]

        print(
            "Using this numerical column instead:",
            range_column
        )

else:

    print(
        "\nUsing Electric Range column:",
        range_column
    )


if range_column is not None:

    series = pd.to_numeric(
        df[range_column],
        errors="coerce"
    ).dropna()

    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    print("\nQ1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower Bound:", lower_bound)
    print("Upper Bound:", upper_bound)

    iqr_outliers = df[
        (df[range_column] < lower_bound) |
        (df[range_column] > upper_bound)
    ]

    print(
        "\nNumber of IQR outliers:",
        len(iqr_outliers)
    )

    print("\nSome outlier records:")

    columns_to_display = [
        col for col in [
            "Make",
            "Model",
            range_column
        ]
        if col in df.columns
    ]

    print(
        iqr_outliers[
            columns_to_display
        ].head(20)
    )


# ============================================================
# TASK 17 — INVESTIGATE OUTLIERS
# ============================================================

print("\nTASK 17 — INVESTIGATE OUTLIERS")

if range_column is not None:

    print("""
Remember:

An outlier is NOT automatically an error.

An unusually high or low value may represent:
- A legitimate vehicle
- A rare vehicle
- An unusual but valid observation
- Or an actual data-entry error

The records should be investigated before deleting anything.
""")

    print(
        "Extreme values:"
    )

    print(
        iqr_outliers[
            columns_to_display
        ].head(20)
    )


# ============================================================
# TASK 18 — Z-SCORE METHOD
# ============================================================

print("\nTASK 18 — Z-SCORE OUTLIER DETECTION")

if range_column is not None:

    valid_values = df[range_column].notna()

    df["Z_Score"] = np.nan

    df.loc[valid_values, "Z_Score"] = zscore(
        df.loc[valid_values, range_column]
    )

    z_outliers = df[
        df["Z_Score"].abs() > 3
    ]

    print(
        "\nMean:",
        df[range_column].mean()
    )

    print(
        "Standard deviation:",
        df[range_column].std()
    )

    print(
        "\nNumber of Z-score outliers:",
        len(z_outliers)
    )


# ============================================================
# TASK 19 — COMPARE IQR AND Z-SCORE
# ============================================================

print("\nTASK 19 — IQR VS Z-SCORE")

if range_column is not None:

    print(
        "IQR outliers:",
        len(iqr_outliers)
    )

    print(
        "Z-score outliers:",
        len(z_outliers)
    )

    print("""
The two methods may identify different records because:

IQR:
- Uses Q1 and Q3
- Uses the interquartile range
- Is less influenced by extreme values

Z-score:
- Uses mean and standard deviation
- Measures distance from the mean
- Can behave differently for skewed data
""")


# ============================================================
# PART F — OUTLIER HANDLING
# ============================================================

print("\n" + "=" * 70)
print("PART F — OUTLIER HANDLING")
print("=" * 70)

# Task 20
print("\nTASK 20 — OUTLIER HANDLING DECISION")

print("""
For this practical, outliers are NOT automatically deleted.

The decision should depend on whether the values are:
- Valid but unusual
- Invalid
- Data-entry errors

Because vehicle data can contain legitimate extreme values,
investigate the records before removing them.
""")


# ============================================================
# TASK 21 — OUTLIER CAPPING
# ============================================================

print("\nTASK 21 — OUTLIER CAPPING")

if range_column is not None:

    capped_column = range_column + "_Capped"

    df[capped_column] = df[range_column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    print(
        f"\nCreated new column: {capped_column}"
    )

    print("\nOriginal vs capped values:")

    print(
        df[
            [range_column, capped_column]
        ].head(20)
    )


# ============================================================
# TASK 22 — LOG TRANSFORMATION
# ============================================================

print("\nTASK 22 — LOG TRANSFORMATION")

if range_column is not None:

    # Only use log1p for non-negative values
    if (df[range_column].dropna() >= 0).all():

        log_column = range_column + "_Log"

        df[log_column] = np.log1p(
            df[range_column]
        )

        print(
            f"\nCreated log-transformed column: {log_column}"
        )

        print("""
Purpose of log transformation:

A log transformation can reduce the effect
of strong right-skewness and make a numerical
distribution more manageable for analysis.
""")

    else:

        print(
            "Log transformation skipped because "
            "the selected column contains negative values."
        )


# ============================================================
# PART G — BASIC DATA ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PART G — BASIC DATA ANALYSIS")
print("=" * 70)


# ============================================================
# TASK 23 — MOST COMMON MANUFACTURERS
# ============================================================

print("\nTASK 23 — MOST COMMON MANUFACTURERS")

make_column = None

for col in df.columns:

    if col.lower() == "make":

        make_column = col
        break

if make_column:

    print(
        df[make_column]
        .value_counts()
        .head(10)
    )

    print(
        "\nMost common manufacturer:",
        df[make_column].value_counts().idxmax()
    )


# ============================================================
# TASK 24 — MOST COMMON MODELS
# ============================================================

print("\nTASK 24 — MOST COMMON VEHICLE MODELS")

model_column = None

for col in df.columns:

    if col.lower() == "model":

        model_column = col
        break

if model_column:

    print(
        df[model_column]
        .value_counts()
        .head(10)
    )


# ============================================================
# TASK 25 — BEV VS PHEV
# ============================================================

print("\nTASK 25 — BEV VS PHEV")

ev_type_column = None

for col in df.columns:

    if "electric vehicle type" in col.lower():

        ev_type_column = col
        break

if ev_type_column:

    ev_counts = df[
        ev_type_column
    ].value_counts()

    print(ev_counts)

    # Search for BEV and PHEV
    for vehicle_type in ev_counts.index:

        if "BEV" in str(vehicle_type).upper():

            print(
                "BEV count:",
                ev_counts[vehicle_type]
            )

        if "PHEV" in str(vehicle_type).upper():

            print(
                "PHEV count:",
                ev_counts[vehicle_type]
            )


# ============================================================
# TASK 26 — MODEL YEAR ANALYSIS
# ============================================================

print("\nTASK 26 — MODEL YEAR ANALYSIS")

year_column = None

for col in df.columns:

    if "model year" in col.lower():

        year_column = col
        break

if year_column:

    print(
        "Earliest model year:",
        df[year_column].min()
    )

    print(
        "Latest model year:",
        df[year_column].max()
    )

    print("\nMost frequent model years:")

    print(
        df[year_column]
        .value_counts()
        .head(10)
    )


# ============================================================
# TASK 27 — ELECTRIC RANGE ANALYSIS
# ============================================================

print("\nTASK 27 — ELECTRIC RANGE ANALYSIS")

if range_column:

    print(
        "Average:",
        df[range_column].mean()
    )

    print(
        "Minimum:",
        df[range_column].min()
    )

    print(
        "Maximum:",
        df[range_column].max()
    )

    print(
        "Potential IQR outliers:",
        len(iqr_outliers)
    )


# ============================================================
# TASK 28 — GEOGRAPHIC ANALYSIS
# ============================================================

print("\nTASK 28 — GEOGRAPHIC ANALYSIS")

for location_name in [
    "County",
    "City",
    "State"
]:

    matching_column = None

    for col in df.columns:

        if col.lower() == location_name.lower():

            matching_column = col
            break

    if matching_column:

        print(
            f"\nTop values for {matching_column}:"
        )

        print(
            df[matching_column]
            .value_counts()
            .head(10)
        )


# ============================================================
# PART H — FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("PART H — FINAL VALIDATION")
print("=" * 70)

# Task 29
print("\nTASK 29 — VALIDATE CLEANED DATASET")

print("\n1. Remaining missing values:")
print(df.isnull().sum())

print(
    "\nTotal remaining missing values:",
    df.isnull().sum().sum()
)

print(
    "\n2. Remaining duplicate rows:",
    df.duplicated().sum()
)

print("\n3. Numerical data types:")
print(df.dtypes)

print("""
4. Text consistency:
Important categorical columns were inspected
and leading/trailing spaces were removed.
""")

print("""
5. Outlier treatment:
Outliers were investigated using IQR and Z-score.
A capped column was created without destroying
the original numerical column.
""")

if (
    df.isnull().sum().sum() == 0
    and df.duplicated().sum() == 0
):

    print(
        "\nThe dataset is ready for analysis because "
        "remaining missing values and exact duplicates "
        "have been checked and the important data-cleaning "
        "operations have been completed."
    )

else:

    print(
        "\nThe dataset still requires additional review."
    )


# ============================================================
# TASK 30 — EXPORT CLEANED DATASET
# ============================================================

print("\nTASK 30 — EXPORT CLEANED DATASET")

# Remove helper Z-score column before final export
if "Z_Score" in df.columns:

    df_export = df.drop(
        columns=["Z_Score"]
    )

else:

    df_export = df.copy()

output_file = "cleaned_ev_data.csv"

df_export.to_csv(
    output_file,
    index=False
)

print(
    "\nCleaned dataset saved as:",
    output_file
)

# Create compressed CSV
compressed_file = "cleaned_ev_data.csv.gz"

df_export.to_csv(
    compressed_file,
    index=False,
    compression="gzip"
)

print(
    "Compressed dataset saved as:",
    compressed_file
)


# ============================================================
# TASK 31 — CLEANING REPORT INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("TASK 31 — CLEANING REPORT")
print("=" * 70)

print("""
Use the following points to write your 5–8 sentence report:

1. Describe the problems found in the raw dataset.
2. Explain which missing-value techniques were used.
3. Explain which text inconsistencies were corrected.
4. State whether exact duplicate rows existed.
5. Explain how outliers were investigated and handled.
6. Explain what changed after cleaning.
7. State whether the final dataset is ready for analysis.
""")

print("\nSuggested report structure:")

print("""
The Electric Vehicle Population dataset was inspected
for missing values, inconsistent text, incorrect data
types, duplicate records, and numerical outliers.
Missing numerical values were handled using the median
where appropriate, while categorical missing values were
handled using the mode. Text columns were standardized
by removing unnecessary leading and trailing spaces.
Duplicate records were identified and removed only when
exact duplicates were present. Outliers were investigated
using both the IQR and Z-score methods, and a capped column
was created while preserving the original data. After
validation, the cleaned dataset was exported for further
analysis.
""")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print("\nOriginal dataset shape:")
print(df_before_missing_cleaning.shape)

print("\nFinal dataset shape:")
print(df_export.shape)

print("\nFinal missing values:")
print(
    df_export.isnull().sum().sum()
)

print("\nFinal duplicate rows:")
print(
    df_export.duplicated().sum()
)

print("\nOutput files:")
print("-", output_file)
print("-", compressed_file)

print("\nDATA CLEANING AND ANALYSIS COMPLETED!")
print("=" * 70)