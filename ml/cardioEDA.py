# Databricks notebook source
# MAGIC %md
# MAGIC # Cardiovascular Disease — Exploratory Data Analysis
# MAGIC
# MAGIC EDA on `pf1.gold.cardiofeatures`. Covers data quality, univariate and bivariate analysis,
# MAGIC correlations and feature importance ranking by mutual information.

# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Setup and configuration

# COMMAND ----------

# DBTITLE 1,Catalog
spark.sql("USE CATALOG `databricks_service_pf`")

# COMMAND ----------

# DBTITLE 1,Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"]  = 110
plt.rcParams["axes.grid"]   = True
plt.rcParams["grid.alpha"]  = 0.3

# COMMAND ----------

# DBTITLE 1,Parameters
dbutils.widgets.text("source_schema", "gold")
dbutils.widgets.text("source_table",  "cardiofeatures")

# COMMAND ----------

# DBTITLE 1,Variables
SOURCE_SCHEMA = dbutils.widgets.get("source_schema")
SOURCE_TABLE  = dbutils.widgets.get("source_table")
FULL_SOURCE   = f"{SOURCE_SCHEMA}.{SOURCE_TABLE}"

TARGET = "cardio"

CONTINUOUS_FEATURES = [
    "age_years", "height_cm", "weight_kg", "bmi",
    "systolic_bp", "diastolic_bp", "pulse_pressure",
]

CATEGORICAL_FEATURES = [
    "age_group_id", "gender", "cholesterol", "gluc",
    "hypertension", "is_smoker", "drinks_alcohol", "is_physically_active",
]

# COMMAND ----------

# DBTITLE 1,Variable labels
VARIABLE_LABELS = {
    "age_years":            "Age (years)",
    "age_group_id":         "Age group",
    "gender":               "Gender",
    "height_cm":            "Height (cm)",
    "weight_kg":            "Weight (kg)",
    "bmi":                  "BMI",
    "systolic_bp":          "Systolic BP (mmHg)",
    "diastolic_bp":         "Diastolic BP (mmHg)",
    "pulse_pressure":       "Pulse pressure (mmHg)",
    "hypertension":         "Hypertension",
    "cholesterol":          "Cholesterol level",
    "gluc":                 "Glucose level",
    "is_smoker":            "Smoker",
    "drinks_alcohol":       "Drinks alcohol",
    "is_physically_active": "Physically active",
    "cardio":               "Cardiovascular disease",
}

# COMMAND ----------

# DBTITLE 1,Color palette
COLORS = {
    "no_cvd":    "#9DD9BB",   # verde menta pastel
    "cvd":       "#EE9695",   # rojo coral pastel
    "primary":   "#9FC1E8",   # azul cielo pastel
    "secondary": "#C3BCE5",   # lila pastel
    "accent":    "#F4B58D",   # durazno pastel
    "neutral":   "#B5B5B0",   # gris claro
    "grid":      "#6B6B68",   # gris medio
    "box_face":  "#CFDEEC",   # azul muy claro
    "box_edge":  "#7B9FC4",   # azul medio
    "box_med":   "#4A6F94",   # azul oscuro
}

CLASS_LEGEND = [
    (0, COLORS["no_cvd"], "No CVD"),
    (1, COLORS["cvd"],    "CVD"),
]

# COMMAND ----------

# DBTITLE 1,Category labels (loaded from Delta dims)
try:
    DIM_TABLE_MAP = {
        "age_group_id": ("gold.dimagegroup",    "IdAgeGroup",        "AgeGroupDescription"),
        "gender":       ("gold.dimgender",      "IdGender",          "GenderDescription"),
        "cholesterol":  ("gold.dimcholesterol", "IdCholesterolType", "CholesterolTypeDescription"),
        "gluc":         ("gold.dimglucose",     "IdGlucoseType",     "GlucoseTypeDescription"),
    }

    BOOLEAN_FEATURES = ["hypertension", "is_smoker", "drinks_alcohol", "is_physically_active"]
    BOOLEAN_LABELS   = {0: "No", 1: "Yes"}

    CATEGORY_LABELS = {}

    for col_name, (table, id_col, desc_col) in DIM_TABLE_MAP.items():
        dim_df = (
            spark.table(table)
                 .select(id_col, desc_col)
                 .toPandas()
        )
        CATEGORY_LABELS[col_name] = {
            int(row[id_col]): f"{int(row[id_col])} - {row[desc_col]}"
            for _, row in dim_df.iterrows()
        }

    for col_name in BOOLEAN_FEATURES:
        CATEGORY_LABELS[col_name] = {
            value: f"{value} - {label}"
            for value, label in BOOLEAN_LABELS.items()
        }

    for col, mapping in CATEGORY_LABELS.items():
        print(f"  {col}: {list(mapping.values())}")

except Exception as e:
    raise Exception(f"[Category-Labels] Failed to build category labels: {e}")

# COMMAND ----------

# DBTITLE 1,Helper functions
def attach_label(df: pd.DataFrame, var_col: str = "variable") -> pd.DataFrame:
    """Add a human-readable description column based on VARIABLE_LABELS."""
    df = df.copy()
    df.insert(1, "description", df[var_col].map(VARIABLE_LABELS).fillna(df[var_col]))
    return df


def grid_layout(n_items: int, n_cols: int = 3, fig_w_per_col: float = 5,
                fig_h_per_row: float = 3.5):
    """Build a (fig, flat_axes) for a grid of n_items subplots."""
    n_rows    = int(np.ceil(n_items / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(fig_w_per_col * n_cols, fig_h_per_row * n_rows))
    axes = np.array(axes).flatten()
    for ax in axes[n_items:]:
        ax.axis("off")
    return fig, axes[:n_items]


def plot_histogram_grid(df, columns, color, suptitle, n_cols=3):
    fig, axes = grid_layout(len(columns), n_cols=n_cols)
    for ax, col in zip(axes, columns):
        ax.hist(df[col], bins=40, color=color, edgecolor="white", alpha=0.85)
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
        ax.set_ylabel("Frequency", fontsize=9)
    plt.suptitle(suptitle, fontsize=13, y=1.00)
    plt.tight_layout(); plt.show()


def plot_boxplot_grid(df, columns, suptitle, n_cols=4):
    fig, axes = grid_layout(len(columns), n_cols=n_cols)
    for ax, col in zip(axes, columns):
        ax.boxplot(df[col], patch_artist=True, widths=0.5,
                   boxprops=dict(facecolor=COLORS["box_face"], edgecolor=COLORS["box_edge"]),
                   medianprops=dict(color=COLORS["box_med"], linewidth=2),
                   flierprops=dict(marker="o", markersize=3,
                                   markerfacecolor=COLORS["cvd"], alpha=0.4))
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
    plt.suptitle(suptitle, fontsize=13, y=1.00)
    plt.tight_layout(); plt.show()


def plot_kde_by_class_grid(df, columns, target_col, suptitle, n_cols=3):
    """KDE plots overlaid by target class — one per continuous variable."""
    fig, axes = grid_layout(len(columns), n_cols=n_cols)
    for ax, col in zip(axes, columns):
        for cls, color, label in CLASS_LEGEND:
            sns.kdeplot(
                data=df[df[target_col] == cls],
                x=col, ax=ax,
                color=color, label=label,
                fill=True, alpha=0.35,
                linewidth=1.8,
                common_norm=False,
            )
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_xlabel("")
        ax.legend(fontsize=8)
    plt.suptitle(suptitle, fontsize=13, y=1.00)
    plt.tight_layout(); plt.show()


def plot_population_pyramid(df, age_col, gender_col,
                            left_gender_code, right_gender_code,
                            suptitle):
    """Horizontal back-to-back bar chart: left gender vs right gender by age group."""
    pyramid = (
        df.groupby([age_col, gender_col], observed=True)
          .size()
          .unstack(fill_value=0)
    )

    age_codes    = pyramid.index.tolist()
    left_counts  = pyramid.get(left_gender_code,  pd.Series(0, index=age_codes)).values
    right_counts = pyramid.get(right_gender_code, pd.Series(0, index=age_codes)).values

    # Build readable labels from CATEGORY_LABELS
    age_label_map    = CATEGORY_LABELS.get(age_col, {})
    gender_label_map = CATEGORY_LABELS.get(gender_col, {})

    age_labels  = [age_label_map.get(int(c), str(c)) for c in age_codes]
    left_label  = gender_label_map.get(left_gender_code, str(left_gender_code))
    right_label = gender_label_map.get(right_gender_code, str(right_gender_code))

    fig, ax = plt.subplots(figsize=(11, 5))
    y_pos   = np.arange(len(age_codes))

    ax.barh(y_pos, -left_counts,  color=COLORS["primary"],
            edgecolor="white", label=left_label)
    ax.barh(y_pos,  right_counts, color=COLORS["accent"],
            edgecolor="white", label=right_label)

    for i, (l_val, r_val) in enumerate(zip(left_counts, right_counts)):
        ax.text(-l_val - max(left_counts) * 0.02, i, f"{l_val:,}",
                va="center", ha="right", fontsize=9)
        ax.text(r_val + max(right_counts) * 0.02, i, f"{r_val:,}",
                va="center", ha="left", fontsize=9)

    max_abs = max(left_counts.max(), right_counts.max())
    ax.set_xlim(-max_abs * 1.15, max_abs * 1.15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(age_labels)
    ax.set_xlabel("Patients")
    ax.set_ylabel(VARIABLE_LABELS.get(age_col, age_col))
    ax.set_title(suptitle, fontsize=13, pad=12)
    ax.axvline(0, color=COLORS["grid"], linewidth=0.8)
    ax.legend(loc="lower right")

    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(abs(v)):,}")
    )

    plt.tight_layout(); plt.show()


def plot_pair_top_features(df, columns, target_col, sample_size, suptitle):
    """Pair plot of the top features colored by target class, on a random sample."""
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    palette = {cls: color for cls, color, _ in CLASS_LEGEND}

    grid = sns.pairplot(
        sample[columns + [target_col]],
        hue=target_col,
        palette=palette,
        diag_kind="kde",
        plot_kws=dict(alpha=0.5, s=14, edgecolor="none"),
        diag_kws=dict(fill=True, alpha=0.4, linewidth=1.5),
        height=2.4,
        corner=False,
    )

    # Replace technical names with readable labels
    for i, col in enumerate(columns):
        label = VARIABLE_LABELS.get(col, col)
        grid.axes[-1][i].set_xlabel(label, fontsize=10)
        grid.axes[i][0].set_ylabel(label, fontsize=10)

    # Rename legend entries from class codes to "No CVD" / "CVD"
    handles = grid._legend.legend_handles if hasattr(grid._legend, "legend_handles") \
              else grid._legend.legendHandles
    class_labels = {cls: label for cls, _, label in CLASS_LEGEND}
    for handle, text in zip(handles, grid._legend.texts):
        try:
            text.set_text(class_labels[int(text.get_text())])
        except (ValueError, KeyError):
            pass

    grid.fig.suptitle(suptitle, fontsize=13, y=1.02)
    plt.show()


def plot_categorical_grid(df, columns, suptitle, mode="count",
                          target_col=None, n_cols=4):
    """mode: 'count' shows frequency bars; 'target_rate' shows mean target by category."""
    fig, axes = grid_layout(len(columns), n_cols=n_cols)
    overall_rate = df[target_col].mean() if mode == "target_rate" else None
    total_rows   = len(df) if mode == "count" else None

    for ax, col in zip(axes, columns):
        if mode == "count":
            series   = df[col].value_counts().sort_index()
            color    = COLORS["secondary"]
            fmt      = lambda v: f"{v:,}\n({v / total_rows:.1%})"
            ylim_top = max(series.values) * 1.20
        else:  # target_rate
            series   = df.groupby(col)[target_col].mean().sort_index()
            color    = COLORS["accent"]
            fmt      = lambda v: f"{v:.1%}"
            ylim_top = max(series.values) * 1.15
            ax.axhline(overall_rate, color=COLORS["grid"], linestyle="--",
                       linewidth=1, label=f"Global mean ({overall_rate:.1%})")
            ax.legend(fontsize=7)

        # Build readable labels from CATEGORY_LABELS
        label_map = CATEGORY_LABELS.get(col, {})
        labels    = [label_map.get(int(v), str(v)) for v in series.index]

        bars = ax.bar(labels, series.values, color=color, edgecolor="white")
        for bar, value in zip(bars, series.values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + ylim_top * 0.01,
                    fmt(value), ha="center", fontsize=8)
        ax.set_title(VARIABLE_LABELS.get(col, col), fontsize=11)
        ax.set_ylim(0, ylim_top)

        # Rotate long labels for legibility
        max_label_len = max((len(l) for l in labels), default=0)
        if max_label_len > 10:
            ax.tick_params(axis="x", rotation=20, labelsize=8)
        else:
            ax.tick_params(axis="x", labelsize=9)

    plt.suptitle(suptitle, fontsize=13, y=1.00)
    plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Data loading and quality assessment

# COMMAND ----------

# DBTITLE 1,Step 1 — Initial load
try:
    cardio_spark   = spark.table(FULL_SOURCE)
    cardio_df      = cardio_spark.toPandas()
    n_rows, n_cols = cardio_df.shape

    print(f"Source:  {FULL_SOURCE}")
    print(f"Rows:    {n_rows:,}")
    print(f"Columns: {n_cols}")

except Exception as e:
    raise Exception(f"[Load] Failed to read {FULL_SOURCE}: {e}")

# COMMAND ----------

# DBTITLE 1,Step 2 — Dataset structure
try:
    dataset_structure = pd.DataFrame({
        "variable": cardio_df.columns,
        "dtype":    cardio_df.dtypes.astype(str).values,
        "role":     [
            "target"      if c == TARGET else
            "continuous"  if c in CONTINUOUS_FEATURES else
            "categorical" if c in CATEGORICAL_FEATURES else
            "other"
            for c in cardio_df.columns
        ],
    })
    dataset_structure = attach_label(dataset_structure)

    role_summary = (
        dataset_structure["role"]
            .value_counts()
            .to_frame()
            .T
            .rename(index={"role": "count"})
    )

    display(dataset_structure)
    display(role_summary)

    print(f"Continuous variables:  {len(CONTINUOUS_FEATURES)}")
    print(f"Categorical variables: {len(CATEGORICAL_FEATURES)}")
    print(f"Target variable:       {TARGET}")

except Exception as e:
    raise Exception(f"[Structure] Failed to build dataset structure: {e}")

# COMMAND ----------

# DBTITLE 1,Step 3 — Data quality
try:
    quality_report = pd.DataFrame({
        "variable":  cardio_df.columns,
        "dtype":     cardio_df.dtypes.astype(str).values,
        "nulls":     cardio_df.isnull().sum().values,
        "null_pct":  (cardio_df.isnull().sum().values / len(cardio_df) * 100).round(2),
        "n_unique":  [cardio_df[c].nunique() for c in cardio_df.columns],
        "constant":  [cardio_df[c].nunique(dropna=False) <= 1 for c in cardio_df.columns],
    })
    quality_report = attach_label(quality_report)

    duplicate_rows = cardio_df.duplicated().sum()
    constant_cols  = quality_report.loc[quality_report["constant"], "variable"].tolist()

    display(quality_report)

    print(f"Duplicate rows:    {duplicate_rows:,}")
    print(f"Constant columns:  {len(constant_cols)} → {constant_cols if constant_cols else 'none'}")

except Exception as e:
    raise Exception(f"[Quality] Failed to build data quality report: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 3. Target variable analysis

# COMMAND ----------

# DBTITLE 1,Step 4 — Target distribution
try:
    target_distribution = (
        cardio_df[TARGET].value_counts()
                         .rename("count")
                         .to_frame()
                         .assign(pct=lambda d: (d["count"] / d["count"].sum()).round(4))
                         .sort_index()
    )
    display(target_distribution)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    class_counts = target_distribution["count"]
    bars = ax.bar(
        [label for _, _, label in CLASS_LEGEND],
        class_counts.values,
        color=[c for _, c, _ in CLASS_LEGEND],
        edgecolor="white",
    )
    for bar, value in zip(bars, class_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 200,
                f"{value:,}", ha="center", fontsize=10)
    ax.set_title("Target variable distribution (cardio)")
    ax.set_ylabel("Patients")
    plt.tight_layout(); plt.show()

except Exception as e:
    raise Exception(f"[Target] Failed to plot target distribution: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 4. Univariate analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 Continuous variables

# COMMAND ----------

# DBTITLE 1,Step 5 — Continuous variables (stats)
try:
    continuous_stats = (
        cardio_df[CONTINUOUS_FEATURES]
            .describe()
            .T
            .round(2)
            .rename_axis("variable")
            .reset_index()
    )
    continuous_stats = attach_label(continuous_stats)
    display(continuous_stats)

except Exception as e:
    raise Exception(f"[Continuous-Stats] Failed to compute describe: {e}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Continuous variables (histograms)
try:
    plot_histogram_grid(
        cardio_df, CONTINUOUS_FEATURES,
        color=COLORS["primary"],
        suptitle="Continuous variable distributions",
    )

except Exception as e:
    raise Exception(f"[Continuous-Histograms] Failed to plot histograms: {e}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Continuous variables (boxplots)
try:
    plot_boxplot_grid(
        cardio_df, CONTINUOUS_FEATURES,
        suptitle="Boxplots — outlier detection",
    )

except Exception as e:
    raise Exception(f"[Continuous-Boxplots] Failed to plot boxplots: {e}")

# COMMAND ----------

# DBTITLE 1,Step 5 — Continuous variables (KDE by class)
try:
    plot_kde_by_class_grid(
        cardio_df, CONTINUOUS_FEATURES, TARGET,
        suptitle="KDE distributions by cardio class",
    )

except Exception as e:
    raise Exception(f"[Continuous-KDE] Failed to plot KDE by class: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 Categorical variables

# COMMAND ----------

# DBTITLE 1,Step 6 — Categorical variables (cardinality and mode)
try:
    categorical_summary_rows = []
    for col in CATEGORICAL_FEATURES:
        mode_value  = cardio_df[col].mode().iloc[0]
        mode_count  = int((cardio_df[col] == mode_value).sum())
        label_map   = CATEGORY_LABELS.get(col, {})
        mode_label  = label_map.get(int(mode_value), str(mode_value))

        categorical_summary_rows.append({
            "variable":    col,
            "cardinality": cardio_df[col].nunique(),
            "mode":        mode_label,
            "mode_count":  mode_count,
            "mode_pct":    round(mode_count / len(cardio_df) * 100, 2),
        })

    categorical_summary = pd.DataFrame(categorical_summary_rows)
    categorical_summary = attach_label(categorical_summary)
    display(categorical_summary)

except Exception as e:
    raise Exception(f"[Categorical-Summary] Failed to build cardinality/mode table: {e}")

# COMMAND ----------

# DBTITLE 1,Step 6 — Categorical variables (frequencies)
try:
    plot_categorical_grid(
        cardio_df, CATEGORICAL_FEATURES,
        suptitle="Categorical variable frequencies",
        mode="count",
    )

except Exception as e:
    raise Exception(f"[Categorical-Frequencies] Failed to plot frequencies: {e}")

# COMMAND ----------

# DBTITLE 1,Step 6 — Population pyramid (age × gender)
try:
    GENDER_FEMALE_CODE = 1
    GENDER_MALE_CODE   = 2

    plot_population_pyramid(
        cardio_df,
        age_col=           "age_group_id",
        gender_col=        "gender",
        left_gender_code=  GENDER_MALE_CODE,
        right_gender_code= GENDER_FEMALE_CODE,
        suptitle=          "Population pyramid — age group by gender",
    )

except Exception as e:
    raise Exception(f"[Population-Pyramid] Failed to plot population pyramid: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 5. Bivariate analysis

# COMMAND ----------

# DBTITLE 1,Step 7 — CVD rate by category
try:
    plot_categorical_grid(
        cardio_df, CATEGORICAL_FEATURES,
        suptitle="CVD rate by category",
        mode="target_rate", target_col=TARGET,
    )

except Exception as e:
    raise Exception(f"[Target-Rate] Failed to plot CVD rate by category: {e}")

# COMMAND ----------

# DBTITLE 1,Step 8 — Correlation matrix (Spearman)
try:
    correlation_features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    correlation_matrix   = cardio_df[correlation_features].corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.75}, ax=ax,
        annot_kws={"fontsize": 8},
    )
    ax.set_title("Spearman correlation matrix", fontsize=13, pad=12)
    plt.tight_layout(); plt.show()

except Exception as e:
    raise Exception(f"[Correlation] Failed to plot correlation matrix: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC # 6. Feature importance

# COMMAND ----------

# DBTITLE 1,Step 9 — Mutual information ranking
try:
    feature_columns = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    X_for_mi        = cardio_df[feature_columns]
    y_for_mi        = cardio_df[TARGET]

    mi_scores = mutual_info_classif(X_for_mi, y_for_mi, random_state=42)

    mutual_info_ranking = (
        pd.DataFrame({
            "variable":    feature_columns,
            "mutual_info": mi_scores.round(4),
        })
        .sort_values("mutual_info", ascending=False)
        .reset_index(drop=True)
    )
    mutual_info_ranking = attach_label(mutual_info_ranking)
    display(mutual_info_ranking)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        mutual_info_ranking["description"][::-1],
        mutual_info_ranking["mutual_info"][::-1],
        color=COLORS["primary"], edgecolor="white",
    )
    for i, value in enumerate(mutual_info_ranking["mutual_info"][::-1]):
        ax.text(value + 0.001, i, f"{value:.4f}",
                va="center", fontsize=9)
    ax.set_xlabel("Mutual information with cardio")
    ax.set_title("Feature ranking by mutual information", fontsize=13, pad=12)
    ax.set_xlim(0, mutual_info_ranking["mutual_info"].max() * 1.15)
    plt.tight_layout(); plt.show()

except Exception as e:
    raise Exception(f"[Mutual-Info] Failed to compute mutual information ranking: {e}")

# COMMAND ----------

# DBTITLE 1,Step 10 — Pair plot of top features
try:
    TOP_N_FEATURES   = 5
    PAIRPLOT_SAMPLE  = 5_000

    top_features = (
        mutual_info_ranking
            .head(TOP_N_FEATURES)["variable"]
            .tolist()
    )

    print(f"Top {TOP_N_FEATURES} features (by MI): {top_features}")
    print(f"Pair plot sample size:           {PAIRPLOT_SAMPLE:,} of {len(cardio_df):,}")

    plot_pair_top_features(
        cardio_df,
        columns=     top_features,
        target_col=  TARGET,
        sample_size= PAIRPLOT_SAMPLE,
        suptitle=    f"Pair plot — top {TOP_N_FEATURES} features by MI",
    )

except Exception as e:
    raise Exception(f"[Pair-Plot] Failed to plot pair plot of top features: {e}")
