import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Titanic EDA Dashboard",
    page_icon="🚢",
    layout="wide"
)

df = pd.read_csv("data/preprocessed_titanic.csv")

st.title("🚢 Titanic Exploratory Data Analysis Dashboard")
st.markdown(
    """
    This interactive dashboard provides an exploratory data analysis (EDA) of the Titanic passenger manifest. 
    Analyze demographic profiles, ticket fare distributions, and key determinants of survival 
    across passenger cohorts. Use the filter below to segment findings by ticket class.
    """
)
st.divider()

kpi_section = st.container()
st.divider()

st.header("🎛 Interactive Filter")

selected_class = st.selectbox(
    "Passenger Class",
    options=["All", 1, 2, 3],
    index=0
)

if selected_class == "All":
    filtered_df = df.copy()
    selected_class_text = "All Passenger Classes"
else:
    filtered_df = df[df["Pclass"] == selected_class]
    selected_class_text = f"Passenger Class {selected_class}"

st.caption(f"Displaying data for {len(filtered_df):,} passengers")
st.divider()

total_passengers = len(filtered_df)
survival_rate = filtered_df["Survived"].mean() * 100 if total_passengers > 0 else 0.0
average_age = filtered_df["Age"].mean() if total_passengers > 0 else 0.0
average_fare = filtered_df["Fare"].mean() if total_passengers > 0 else 0.0

with kpi_section:
    st.header("📊 KPI Overview")
    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
    kpi_col1.metric("Total Passengers", f"{total_passengers:,}")
    kpi_col2.metric("Survival Rate", f"{survival_rate:.1f}%")
    kpi_col3.metric("Average Age", f"{average_age:.1f} Years")
    kpi_col4.metric("Average Fare", f"${average_fare:,.2f}")

chart_df = filtered_df.copy()
chart_df["Survival Status"] = chart_df["Survived"].map({
    0: "Did Not Survive",
    1: "Survived"
})

survival_colors = {
    "Survived": "#10b981",
    "Did Not Survive": "#ef4444"
}

st.header("📈 Survival Analysis")

survival_by_sex = (
    chart_df.groupby(["Sex", "Survival Status"])
    .size()
    .reset_index(name="Passenger Count")
)
survival_by_sex["Sex"] = survival_by_sex["Sex"].str.capitalize()

fig_survival_by_sex = px.bar(
    survival_by_sex,
    x="Sex",
    y="Passenger Count",
    color="Survival Status",
    barmode="group",
    title="Survival Distribution by Gender",
    color_discrete_map=survival_colors,
    labels={
        "Sex": "Gender",
        "Passenger Count": "Passenger Count",
        "Survival Status": "Survival Status"
    }
)

fig_survival_by_sex.update_layout(
    template="plotly_white",
    title_font=dict(size=16, family="sans-serif", color="#1e293b"),
    font=dict(family="sans-serif", size=12, color="#475569"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title=None
    ),
    margin=dict(l=40, r=40, t=80, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    bargap=0.25,
    bargroupgap=0.05,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif")
)
fig_survival_by_sex.update_xaxes(showgrid=False, linecolor="#cbd5e1")
fig_survival_by_sex.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
fig_survival_by_sex.update_traces(hovertemplate="%{y:,} Passengers<extra></extra>")

survival_by_class = (
    chart_df.groupby(["Pclass", "Survival Status"])
    .size()
    .reset_index(name="Passenger Count")
)
survival_by_class["Passenger Class"] = survival_by_class["Pclass"].map({
    1: "1st Class",
    2: "2nd Class",
    3: "3rd Class"
})

fig_survival_by_class = px.bar(
    survival_by_class,
    x="Passenger Class",
    y="Passenger Count",
    color="Survival Status",
    barmode="group",
    title="Survival Distribution by Passenger Class",
    color_discrete_map=survival_colors,
    labels={
        "Passenger Class": "Passenger Class",
        "Passenger Count": "Passenger Count",
        "Survival Status": "Survival Status"
    }
)

fig_survival_by_class.update_layout(
    template="plotly_white",
    title_font=dict(size=16, family="sans-serif", color="#1e293b"),
    font=dict(family="sans-serif", size=12, color="#475569"),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        title=None
    ),
    margin=dict(l=40, r=40, t=80, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    bargap=0.25,
    bargroupgap=0.05,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif")
)
fig_survival_by_class.update_xaxes(showgrid=False, linecolor="#cbd5e1")
fig_survival_by_class.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
fig_survival_by_class.update_traces(hovertemplate="%{y:,} Passengers<extra></extra>")

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.plotly_chart(fig_survival_by_sex, use_container_width=True)
    if selected_class == "All":
        st.info(
            "**Key Insight:** Female passengers achieved a survival rate of ~74% compared to ~19% for males, "
            "reflecting the maritime 'women and children first' evacuation priority."
        )
    else:
        st.info(
            f"**Key Insight:** Within **{selected_class_text}**, female passengers consistently show higher "
            "survival rates, confirming gender as a primary determinant across all passenger classes."
        )

with chart_col2:
    st.plotly_chart(fig_survival_by_class, use_container_width=True)
    if selected_class == "All":
        st.info(
            "**Key Insight:** Socioeconomic class strongly influenced survival. Over 62% of 1st Class passengers "
            "survived, whereas 3rd Class passengers suffered the highest mortality (~76%)."
        )
    else:
        st.info(
            f"**Key Insight:** Within **{selected_class_text}**, the distribution shows the direct impact "
            "of ticket class tiering on survival outcomes and access to lifeboats."
        )

st.divider()

st.header("💰 Fare Analysis")

fig_fare_distribution = px.histogram(
    filtered_df,
    x="Fare",
    nbins=30,
    title="Passenger Ticket Fare Distribution",
    labels={
        "Fare": "Fare Paid (USD)",
        "count": "Passenger Count"
    },
    color_discrete_sequence=["#4f46e5"]
)

fig_fare_distribution.update_layout(
    template="plotly_white",
    title_font=dict(size=16, family="sans-serif", color="#1e293b"),
    font=dict(family="sans-serif", size=12, color="#475569"),
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    bargap=0.05,
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="sans-serif")
)
fig_fare_distribution.update_xaxes(showgrid=False, linecolor="#cbd5e1")
fig_fare_distribution.update_yaxes(showgrid=True, gridcolor="#f1f5f9", linecolor="#cbd5e1")
fig_fare_distribution.update_traces(
    hovertemplate="Fare Paid: $%{x:.2f}<br>Passenger Count: %{y:,}<extra></extra>"
)

st.plotly_chart(fig_fare_distribution, use_container_width=True)

if selected_class == "All":
    st.info(
        "**Key Insight:** Fares are heavily right-skewed, showing that the majority of passengers "
        "paid under $50. A small number of premium 1st Class passengers paid fares up to $512."
    )
else:
    st.info(
        f"**Key Insight:** Fare distributions within **{selected_class_text}** highlight the internal "
        "pricing tiers and variations for this specific passenger class."
    )

st.divider()

st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.9em; line-height: 1.6; padding-top: 10px;'>"
    "Titanic Dataset<br>"
    "Interactive EDA Dashboard<br>"
    "Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
