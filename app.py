import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    from db_handler import HouseRentDatabase
    import pandas as pd
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.preprocessing import PowerTransformer
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    from llm_service import LLMService

    db = HouseRentDatabase()
    df = db.get_data()

    return (
        IsolationForest,
        LLMService,
        LocalOutlierFactor,
        PowerTransformer,
        add_constant,
        df,
        mo,
        np,
        pd,
        plt,
        sns,
        stats,
        variance_inflation_factor,
    )


@app.cell
def _(df, mo):
    mo.vstack([
        mo.md("# 🏠 Scientific House Rent Dashboard"),
        mo.md("### What is BHK?"),
        mo.md("BHK stands for **Bedroom, Hall, and Kitchen**...")
    ])

    city_select = mo.ui.dropdown(
        options=df["City"].unique().tolist(),
        label="Select City",
        value=df["City"].unique()[0],
    )
    bhk_slider = mo.ui.slider(
        start=df["BHK"].min(), stop=df["BHK"].max(), label="BHK", value=df["BHK"].min()
    )

    # Define sidebar
    sidebar = mo.sidebar([mo.md("## Filters"), city_select, bhk_slider])

    sidebar
    return bhk_slider, city_select


@app.cell
def _(bhk_slider, city_select, df, mo, plt, sns, stats):
    filtered_df = df[
        (df["City"] == city_select.value) & (df["BHK"] == bhk_slider.value)
    ]

    # KPIs
    kpi_row = mo.hstack(
        [
            mo.stat(label="Avg Rent", value=f"₹{filtered_df['Rent'].mean():,.0f}"),
            mo.stat(
                label="Median Size", value=f"{filtered_df['Size'].median():.0f} sqft"
            ),
            mo.stat(label="Count", value=f"{len(filtered_df)}"),
        ]
    )

    # Plots
    fig1, ax1 = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(filtered_df["Rent"], kde=True, ax=ax1[0])
    ax1[0].set_title("Rent Distribution")

    sns.regplot(data=filtered_df, x="Size", y="Rent", ax=ax1[1])
    ax1[1].set_title("Rent vs Size (Regression)")

    # Regression Stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        filtered_df["Size"], filtered_df["Rent"]
    )

    mo.vstack(
        [
            kpi_row,
            mo.as_html(fig1),
            mo.md(
                f"### Regression Analysis\nSlope: {slope:.2f}, R-squared: {r_value**2:.4f}, P-value: {p_value:.4f}"
            ),
        ]
    )
    return p_value, r_value


@app.cell
def _(LLMService, mo, p_value, r_value):
    llm = LLMService()
    conclusion = llm.generate_conclusion(r_value, p_value)
    mo.md(f"## 🧠 Research Conclusion\n{conclusion}")
    return


@app.cell
def _(df, mo):
    mo.md("## 📊 Data Preview")
    mo.ui.table(df)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    # 🔬 Statistical Integrity Module
    *By Mario — covering: (1) abnormal points, (2) deeper feature engineering, (3) deeper EDA*

    This module audits the dataset before any inference: it identifies outliers with both
    univariate and multivariate methods, engineers new features grounded in domain knowledge,
    and runs formal statistical tests so the regression in the dashboard above rests on solid ground.
    """)
    return


@app.cell
def _(df, mo, np):
    # --- Data integrity snapshot --------------------------------------------------
    _num_cols = df.select_dtypes(include=np.number).columns.tolist()
    _desc = df[_num_cols].describe().T
    _desc["skew"] = df[_num_cols].skew()
    _desc["kurtosis"] = df[_num_cols].kurtosis()
    _desc["missing_%"] = (df[_num_cols].isna().sum() / len(df) * 100)
    _desc = _desc.round(3)

    _n_missing = int(df.isna().sum().sum())
    _n_dupes = int(df.duplicated().sum())

    mo.vstack([
        mo.md("## 📋 Data Integrity Snapshot"),
        mo.hstack([
            mo.stat(label="Rows", value=f"{len(df):,}"),
            mo.stat(label="Columns", value=f"{df.shape[1]}"),
            mo.stat(label="Missing cells", value=f"{_n_missing}"),
            mo.stat(label="Duplicate rows", value=f"{_n_dupes}"),
        ]),
        mo.md(
            "**Skewness** measures asymmetry (0 = symmetric, >1 strong positive skew)."
            "High **kurtosis** indicates heavy tails, meaning outliers are more likely."
    "If a target variable (Rent) has high skewness, linear regression without transformation violates its assumptions."
        ),
        mo.ui.table(_desc),
    ])
    return


@app.cell
def _(mo):
    # --- Section 1 · Outlier detection controls ----------------------------------
    outlier_target = mo.ui.dropdown(
        options=["Rent", "Size", "BHK", "Bathroom"],
        label="Variable to inspect",
        value="Rent",
    )
    outlier_method = mo.ui.dropdown(
        options=["IQR", "Z-score", "Isolation Forest", "Local Outlier Factor (LOF)"],
        label="Detection method",
        value="IQR",
    )
    iqr_factor = mo.ui.slider(start=1.0, stop=3.0, step=0.1, value=1.5, label="IQR factor k")
    z_thresh = mo.ui.slider(start=2.0, stop=5.0, step=0.1, value=3.0, label="Z-score threshold")
    contamination = mo.ui.slider(start=0.01, stop=0.20, step=0.01, value=0.05, label="Contamination (IF/LOF)")
    scope_city = mo.ui.checkbox(value=True, label="Restrict to selected city only")

    mo.vstack([
        mo.md("## 🚨 Section 1 — Abnormal Points"),
        mo.md(
            "Compare four detection strategies. **IQR** and **Z-score** are univariate "
            "(flag rows extreme in one variable). **Isolation Forest** and **LOF** are "
            "multivariate — they flag suspicious *combinations* (e.g. small flat with huge rent)."
        ),
        mo.hstack([outlier_target, outlier_method]),
        mo.hstack([iqr_factor, z_thresh, contamination]),
        scope_city,
    ])
    return (
        contamination,
        iqr_factor,
        outlier_method,
        outlier_target,
        scope_city,
        z_thresh,
    )


@app.cell
def _(
    IsolationForest,
    LocalOutlierFactor,
    city_select,
    contamination,
    df,
    iqr_factor,
    mo,
    np,
    outlier_method,
    outlier_target,
    plt,
    scope_city,
    sns,
    stats,
    z_thresh,
):
    # --- Section 1 · Outlier computation -----------------------------------------
    _base = df[df["City"] == city_select.value].copy() if scope_city.value else df.copy()
    _feat_cols = [c for c in ["Rent", "Size", "BHK", "Bathroom"] if c in _base.columns]
    _work = _base.dropna(subset=_feat_cols).copy()
    _var = outlier_target.value
    _method = outlier_method.value

    if _method == "IQR":
        _q1, _q3 = _work[_var].quantile([0.25, 0.75])
        _iqr = _q3 - _q1
        _k = iqr_factor.value
        _lo, _hi = _q1 - _k * _iqr, _q3 + _k * _iqr
        _work["is_outlier"] = (_work[_var] < _lo) | (_work[_var] > _hi)
        _desc_method = f"IQR bounds [{_lo:,.0f} — {_hi:,.0f}] with k={_k}"
    elif _method == "Z-score":
        _z = np.abs(stats.zscore(_work[_var]))
        _work["is_outlier"] = _z > z_thresh.value
        _desc_method = f"|Z| > {z_thresh.value}"
    elif _method == "Isolation Forest":
        _iso = IsolationForest(contamination=contamination.value, random_state=42, n_estimators=200)
        _work["is_outlier"] = _iso.fit_predict(_work[_feat_cols]) == -1
        _desc_method = f"contamination={contamination.value}, features={_feat_cols}"
    else:  # LOF
        _n_neighbors = min(20, max(5, len(_work) // 50))
        _lof = LocalOutlierFactor(n_neighbors=_n_neighbors, contamination=contamination.value)
        _work["is_outlier"] = _lof.fit_predict(_work[_feat_cols]) == -1
        _desc_method = f"n_neighbors={_n_neighbors}, contamination={contamination.value}"

    _n_out = int(_work["is_outlier"].sum())
    _pct_out = _n_out / len(_work) * 100 if len(_work) else 0
    _mean_full = _work[_var].mean()
    _mean_clean = _work.loc[~_work["is_outlier"], _var].mean()
    _median_full = _work[_var].median()
    _median_clean = _work.loc[~_work["is_outlier"], _var].median()
    _shift_mean = (_mean_full - _mean_clean) / _mean_full * 100 if _mean_full else 0

    _fig_out, _ax_out = plt.subplots(1, 2, figsize=(13, 5))
    sns.boxplot(data=_work, x="is_outlier", y=_var, ax=_ax_out[0])
    _ax_out[0].set_title(f"{_var} — boxplot by outlier flag")
    sns.scatterplot(
        data=_work, x="Size", y="Rent",
        hue="is_outlier", palette={False: "#3b82f6", True: "#ef4444"},
        alpha=0.65, ax=_ax_out[1],
    )
    _ax_out[1].set_title("Rent vs Size — outliers highlighted in red")
    plt.tight_layout()

    _top_out = (
        _work.loc[_work["is_outlier"], _feat_cols + (["City"] if "City" in _work.columns else [])]
        .sort_values(_var, ascending=False)
        .head(10)
    )

    mo.vstack([
        mo.md(f"### Result — **{_method}** · {_desc_method}"),
        mo.hstack([
            mo.stat(label="Subset size", value=f"{len(_work):,}"),
            mo.stat(label="Outliers", value=f"{_n_out}"),
            mo.stat(label="% outliers", value=f"{_pct_out:.2f}%"),
            mo.stat(label=f"Mean {_var} drop after cleaning", value=f"{_shift_mean:+.1f}%"),
        ]),
        mo.md(
            f"Mean {_var}: **{_mean_full:,.0f}** (all) → **{_mean_clean:,.0f}** (clean).  "
            f"Median {_var}: **{_median_full:,.0f}** (all) → **{_median_clean:,.0f}** (clean).  "
            "_Large difference between the full and cleaned mean = outliers are biasing the average; the median is robust and barely changes._"
        ),
        mo.as_html(_fig_out),
        mo.md("### Top-10 most extreme flagged rows"),
        mo.ui.table(_top_out),
    ])
    return


@app.cell
def _(PowerTransformer, df, mo, np, pd, plt, sns):
    # --- Section 2 · Feature engineering -----------------------------------------
    df_fe = df.copy()

    # 1) Price normalized by area
    df_fe["Rent_per_sqft"] = df_fe["Rent"] / df_fe["Size"].replace(0, np.nan)

    # 2) Luxury / quality proxy
    df_fe["Bath_per_BHK"] = df_fe["Bathroom"] / df_fe["BHK"].replace(0, np.nan)

    # 3) Parse free-text Floor column → numeric features
    def _parse_floor(s):
        try:
            parts = str(s).lower().split("out of")
            cur_raw = parts[0].strip()
            tot = int(parts[1].strip()) if len(parts) > 1 else np.nan
            if "ground" in cur_raw or "lower" in cur_raw:
                cur = 0
            elif "upper basement" in cur_raw or "basement" in cur_raw:
                cur = -1
            else:
                cur = int(cur_raw)
            return pd.Series([cur, tot])
        except Exception:
            return pd.Series([np.nan, np.nan])

    if "Floor" in df_fe.columns:
        df_fe[["Current_floor", "Total_floors"]] = df_fe["Floor"].apply(_parse_floor)
        df_fe["Floor_ratio"] = df_fe["Current_floor"] / df_fe["Total_floors"].replace(0, np.nan)
        df_fe["Is_high_rise"] = (df_fe["Total_floors"] >= 10).astype("Int64")

    # 4) Variance-stabilizing transforms on Rent (target)
    df_fe["log_Rent"] = np.log1p(df_fe["Rent"])
    df_fe["log_Size"] = np.log1p(df_fe["Size"])

    _pt = PowerTransformer(method="yeo-johnson", standardize=False)
    df_fe["yj_Rent"] = _pt.fit_transform(df_fe[["Rent"]])
    yj_lambda = float(_pt.lambdas_[0])

    # 5) Size buckets (discretized for grouped analysis)
    df_fe["Size_bucket"] = pd.cut(
        df_fe["Size"],
        bins=[0, 500, 1000, 1500, 2500, np.inf],
        labels=["XS (<500)", "S (500-1k)", "M (1k-1.5k)", "L (1.5k-2.5k)", "XL (>2.5k)"],
    )

    # Before/after skewness & kurtosis
    _cmp_cols = ["Rent", "log_Rent", "yj_Rent", "Size", "log_Size", "Rent_per_sqft"]
    skew_table = pd.DataFrame({
        "feature": _cmp_cols,
        "skewness": [df_fe[c].skew() for c in _cmp_cols],
        "kurtosis": [df_fe[c].kurtosis() for c in _cmp_cols],
    }).round(3)

    _fig_fe, _ax_fe = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df_fe["Rent"], kde=True, ax=_ax_fe[0])
    _ax_fe[0].set_title(f"Rent (skew={df_fe['Rent'].skew():.2f})")
    sns.histplot(df_fe["log_Rent"], kde=True, ax=_ax_fe[1])
    _ax_fe[1].set_title(f"log(Rent) (skew={df_fe['log_Rent'].skew():.2f})")
    sns.histplot(df_fe["yj_Rent"], kde=True, ax=_ax_fe[2])
    _ax_fe[2].set_title(f"Yeo-Johnson Rent (λ={yj_lambda:.2f}, skew={df_fe['yj_Rent'].skew():.2f})")
    plt.tight_layout()

    mo.vstack([
        mo.md("## 🛠️ Section 2 — Deeper Feature Engineering"),
        mo.md(
            f"""
    **New variables generated:**

    - `Rent_per_sqft` — rent per square foot; normalizes price by size.
    - `Bath_per_BHK` — bathrooms/bedrooms ratio; proxy for quality/luxury.
    - `Current_floor`, `Total_floors`, `Floor_ratio`, `Is_high_rise` — parsing of the Floor field.
    - `log_Rent`, `log_Size` — stabilize variance (Rent is strongly right-skewed).
    - `yj_Rent` — Yeo-Johnson with λ={yj_lambda:.3f}; generalizes Box-Cox and handles zeros/negative values.
    - `Size_bucket` — discretization of Size for categorical analysis."""
        ),
        mo.md("### Comparison of skewness/kurtosis"),
        mo.ui.table(skew_table),
        mo.as_html(_fig_fe),
        mo.md(
            "**Interpretation**. The closer `skew` is to 0, the more symmetric the distribution is."
    "If `Rent` has high skewness (>2), linear regression violates the normality of residuals;"
    "after applying `log` or Yeo-Johnson, the distribution becomes closer to normal and the inferences are valid."
        ),
    ])
    return (df_fe,)


@app.cell
def _(add_constant, df_fe, mo, np, pd, plt, sns, variance_inflation_factor):
    # --- Section 2 · VIF + correlation -------------------------------------------
    _vif_features = ["BHK", "Size", "Bathroom", "Rent_per_sqft", "Bath_per_BHK"]
    _vif_features = [c for c in _vif_features if c in df_fe.columns]
    _vif_df = df_fe[_vif_features].replace([np.inf, -np.inf], np.nan).dropna()
    _X_vif = add_constant(_vif_df)
    vif_table = pd.DataFrame({
        "feature": _X_vif.columns,
        "VIF": [variance_inflation_factor(_X_vif.values, i) for i in range(_X_vif.shape[1])],
    })
    vif_table = vif_table[vif_table["feature"] != "const"].round(2)

    _corr_cols = ["Rent", "log_Rent", "Size", "BHK", "Bathroom", "Rent_per_sqft", "Bath_per_BHK"]
    _corr_cols = [c for c in _corr_cols if c in df_fe.columns]
    _fig_corr, _ax_corr = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        df_fe[_corr_cols].corr(method="spearman"),
        annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=_ax_corr,
    )
    _ax_corr.set_title("Spearman correlation — engineered features")
    plt.tight_layout()

    mo.vstack([
        mo.md("### 🧪 Multicollinearity check — Variance Inflation Factor"),
        mo.ui.table(vif_table),
        mo.md(
            "Practical rule: **VIF > 5** suggests relevant multicollinearity; **VIF > 10** is severe."
    "Two highly correlated features inflate the variance of the coefficients and make the regression unstable."
        ),
        mo.as_html(_fig_corr),
        mo.md(
            "We use **Spearman** correlation (rank-based) because it is robust to outliers"
    " and captures monotonic relationships that are not necessarily linear — more reliable than Pearson when there is skewness."
        ),
    ])
    return


@app.cell
def _(df_fe, mo):
    # --- Section 3 · Deeper EDA controls -----------------------------------------
    eda_var = mo.ui.dropdown(
        options=[c for c in ["Rent", "log_Rent", "yj_Rent", "Size", "log_Size", "Rent_per_sqft", "BHK", "Bathroom"]
                if c in df_fe.columns],
        label="Variable",
        value="Rent",
    )
    eda_group = mo.ui.dropdown(
        options=[c for c in ["City", "Furnishing Status", "Area Type", "Tenant Preferred", "Size_bucket"]
                if c in df_fe.columns],
        label="Group by",
        value="City",
    )

    mo.vstack([
        mo.md("## 🔍 Section 3 — Deeper EDA"),
        mo.md(
            "Change the variable and grouping factor to explore distribution, normality, and differences between groups."
    "We use non-parametric tests where appropriate because the data are rarely normally distributed."
        ),
        mo.hstack([eda_var, eda_group]),
    ])
    return eda_group, eda_var


@app.cell
def _(df_fe, eda_group, eda_var, mo, pd, plt, sns, stats):
    # --- Section 3 · EDA computation ---------------------------------------------
    _v = eda_var.value
    _g = eda_group.value
    _eda = df_fe.dropna(subset=[_v, _g]).copy()

    # Sample for normality tests (Shapiro is unreliable above n≈5000)
    _sample = _eda[_v].sample(min(5000, len(_eda)), random_state=42)

    _sh = stats.shapiro(_sample)
    _ad = stats.anderson(_sample, dist="norm")
    _jb = stats.jarque_bera(_sample)

    _norm_table = pd.DataFrame({
        "Test": ["Shapiro-Wilk", "Anderson-Darling", "Jarque-Bera"],
        "Statistic": [_sh.statistic, _ad.statistic, _jb.statistic],
        "p-value / crit(5%)": [_sh.pvalue, _ad.critical_values[2], _jb.pvalue],
        "Normal at α=0.05?": [
            "❌ No" if _sh.pvalue < 0.05 else "✅ Yes",
            "❌ No" if _ad.statistic > _ad.critical_values[2] else "✅ Yes",
            "❌ No" if _jb.pvalue < 0.05 else "✅ Yes",
        ],
    }).round(4)

    # Non-parametric ANOVA across groups
    _groups = [grp_df[_v].dropna().values for _, grp_df in _eda.groupby(_g) if len(grp_df) > 5]
    if len(_groups) >= 2:
        _kw_h, _kw_p = stats.kruskal(*_groups)
    else:
        _kw_h, _kw_p = float("nan"), float("nan")

    _fig_eda, _ax_eda = plt.subplots(1, 3, figsize=(16, 5))
    sns.histplot(_eda[_v], kde=True, ax=_ax_eda[0])
    _ax_eda[0].set_title(f"Distribution of {_v}")
    stats.probplot(_sample, dist="norm", plot=_ax_eda[1])
    _ax_eda[1].set_title(f"Q-Q plot vs Normal — {_v}")
    sns.boxplot(data=_eda, x=_g, y=_v, ax=_ax_eda[2])
    _ax_eda[2].tick_params(axis="x", rotation=30)
    _ax_eda[2].set_title(f"{_v} by {_g}")
    plt.tight_layout()

    _kw_verdict = (
        f"H = {_kw_h:.2f}, p = {_kw_p:.2e} → "
        + ("**Significant differences between groups** (p<0.05). The grouper explains variance."
           if _kw_p < 0.05 else
           "Without significant differences — the grouper doesn't separate variables")
    )

    mo.vstack([
        mo.md(f"### Normality tests for `{_v}` (n={len(_sample)})"),
        mo.ui.table(_norm_table),
        mo.md(f"### Kruskal-Wallis — `{_v}` across `{_g}`"),
        mo.md(_kw_verdict),
        mo.as_html(_fig_eda),
        mo.md(
            "**How to interpret it.** In the Q-Q plot, if the points follow the diagonal line, the variable is approximately ~normal;"
    " if they deviate in the tails, there is skewness or heavy tails.When all three tests reject normality"
    " and the variable is the target, it is advisable to use regression on `log_Rent` or `yj_Rent` (already created)."
        ),
    ])
    return


@app.cell
def _(df_fe, mo, plt, sns):
    # --- Section 3 · Segmented summary by city -----------------------------------
    city_summary = (
        df_fe.groupby("City")
        .agg(
            n=("Rent", "size"),
            median_rent=("Rent", "median"),
            mean_rent=("Rent", "mean"),
            median_size=("Size", "median"),
            median_rent_per_sqft=("Rent_per_sqft", "median"),
            skew_rent=("Rent", "skew"),
        )
        .round(2)
        .sort_values("median_rent_per_sqft", ascending=False)
    )

    _fig_seg, _ax_seg = plt.subplots(1, 2, figsize=(14, 5))
    sns.barplot(data=city_summary.reset_index(), x="City", y="median_rent_per_sqft",
                ax=_ax_seg[0])
    _ax_seg[0].set_title("Median ₹/sqft by city (size-normalized)")
    _ax_seg[0].tick_params(axis="x", rotation=30)
    sns.violinplot(data=df_fe, x="City", y="log_Rent", ax=_ax_seg[1])
    _ax_seg[1].set_title("log(Rent) distribution by city")
    _ax_seg[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()

    mo.vstack([
        mo.md("### 🌆 Segmented view — by city"),
        mo.md(
            "The ranking by **median ₹/sqft** corrects for the size effect: a city with apartments"
            "large apartments will appear expensive in total `Rent` but may be cheap per square foot. The violin plot on"
            "`log_Rent` shows how dispersion varies by city."
        ),
        mo.ui.table(city_summary),
        mo.as_html(_fig_seg),
    ])
    return


if __name__ == "__main__":
    app.run()
