# Battery Analysis Tool

This repository contains the **BatteryAnalysisTool.ipynb** notebook, which is used for analyzing and visualizing battery voltage data, detecting anomalies, and generating reports. Follow the instructions below to get the notebook up and running on your local machine.

## Prerequisites

Make sure you have the following installed on your system:

- Python 3.7 or higher (Python 3.11 is recommended)
- `pip` (Python package installer)
- Jupyter Notebook or Jupyter Lab

# Battery Storage Voltage Analysis Detection

This tool processes cell voltage data from CSV files to identify potential sensor failures and analyze battery performance.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You need Python 3.12.4 installed on your system. The project uses the following Python libraries:
- pandas
- matplotlib
- numpy

### Installing

1. Clone the repository:

```bash
git clone https://github.com/lennox55555/Battery-Analysis-Tool.git
```

2. Change Directory:

```bash
cd Battery-Storage-Analysis-Tool
```

3. (Optional) Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

4. Install the required packages:

```bash
pip install -r requirements.txt
```

5. Update the script with correlating path:

- Place your CSV file in the `Battery-Storage-Analysis-Tool` directory.

- Open `main.py` and update the file path in the `load_battery_data`.

6. Run the script

```bash
python main.py
```


