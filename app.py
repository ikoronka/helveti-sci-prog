import marimo

__generated_with = "0.10.19"
app = marimo.App()


@app.cell
def __(mo):
    import marimo as mo
    from db_handler import HouseRentDatabase
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy import stats
    from llm_service import LLMService

    db = HouseRentDatabase()
    df = db.get_data()
    return LLMService, db, df, mo, pd, plt, sns, stats


@app.cell
def __(df, mo):
    mo.md("# 🏠 Scientific House Rent Dashboard")

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

    return bhk_slider, city_select, sidebar


@app.cell
def __(bhk_slider, city_select, df, mo, plt, sns, stats):
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
    return ax1, fig1, intercept, kpi_row, p_value, r_value, slope, std_err


@app.cell
def __(LLMService, mo, p_value, r_value, slope):
    llm = LLMService()
    conclusion = llm.generate_conclusion(r_value, p_value)
    mo.md(f"## 🧠 Research Conclusion\n{conclusion}")
    return conclusion, llm


@app.cell
def __(df, mo):
    mo.md("## 📊 Data Preview")
    mo.ui.table(df)
    return


if __name__ == "__main__":
    app.run()
