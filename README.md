# week-2

## Overview

This project contains several dashboards for analyzing various aspects of telecom data, including user satisfaction, experience, and engagement. The dashboards are built using Streamlit and leverage data analysis and visualization libraries such as Pandas, Seaborn, and Plotly.

## Dashboards

### 1. Satisfaction Dashboard

The `SatisfactionDashboard` class provides an analysis of user satisfaction based on engagement and experience scores.

#### Key Methods:
- `load_data()`: Loads the data using a provided SQL query and performs various analyses including clustering and scoring.
- `display_dashboard()`: Displays the dashboard with various plots and data tables.

#### Example Usage:
```python
from Dashboard.SatisfactionDashboard import SatisfactionDashboard

query = "SELECT * FROM telecom_data"
dashboard = SatisfactionDashboard(query)
dashboard.load_data()
dashboard.display_dashboard()