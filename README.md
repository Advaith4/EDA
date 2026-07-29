# Titanic Exploratory Data Analysis (EDA) Dashboard

An interactive, single-file Streamlit web application that provides exploratory data analysis (EDA) of the Titanic passenger manifest. This dashboard is designed for analyzing passenger demographics, ticket pricing structures, and key determinants of survival using dynamic filters and interactive Plotly visualizations.

---

## 🚀 Features

- **📊 KPI Overview**: Real-time summary metrics presenting Total Passengers, Survival Rate (%), Average Age, and Average Ticket Fare.
- **🎛️ Interactive Filters**: Dynamic segmentation of all metrics, charts, and insights by ticket class (All, 1st, 2nd, or 3rd Class).
- **📈 Survival Analysis**: 
  - **Gender Breakdown**: Side-by-side grouped bar chart illustrating survival count distributions across male and female passengers.
  - **Class Breakdown**: Side-by-side grouped bar chart illustrating survival distributions by socioeconomic class.
  - **Unified Hover Tooltips**: Refined hover tooltips providing immediate counts for improved readability.
- **💰 Fare Distribution**: A detailed histogram displaying ticket pricing ranges to highlight fare skewness and class-based ticket pricing structures.
- **💡 Actionable Insights**: Context-aware insight callouts positioned directly underneath each visualization, updating dynamically depending on selected filter criteria.

---

## 🛠️ Tech Stack

- **Python** (Core Logic)
- **Streamlit** (Dashboard UI/UX and Layout Flow)
- **Pandas** (Data Transformation and Aggregation)
- **Plotly Express** (Interactive Visualization Charts)

---

## ⚙️ Setup & Installation

Follow these steps to run the dashboard locally on your system:

### 1. Navigate to Workspace
```bash
cd NUMPY
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
python -m venv .venv
```

**Activate the virtual environment:**
- **Windows (PowerShell/CMD):**
  ```powershell
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### 3. Install Dependencies
```bash
pip install streamlit pandas plotly
```

### 4. Run the Application
Start the Streamlit development server:
```bash
streamlit run app.py
```

Once started, the dashboard will open automatically in your default browser at `http://localhost:8501`.

---

## 📂 Project Structure
- `app.py`: The single-file entrypoint containing the dashboard layout, KPI metrics, interactive filters, and Plotly charts.
- `data/preprocessed_titanic.csv`: Cleaned Titanic passenger manifest dataset used for visual mapping.
