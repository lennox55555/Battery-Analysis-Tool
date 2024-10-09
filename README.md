# Battery Analysis Tool

This repository contains the **BatteryAnalysisTool.ipynb** notebook, which is used for analyzing and visualizing battery voltage data, detecting anomalies, and generating reports. Follow the instructions below to get the notebook up and running on your local machine.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.


### Installing

1. Clone the repository:

```bash
git clone https://github.com/lennox55555/Battery-Analysis-Tool.git
```

2. Change Directory:

```bash
cd Battery-Analysis-Tool/
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

- Place your all your CSV files in the `Battery-Storage-Analysis-Tool` directory.

- Open `main.py` and update the input and output file path. Input should be the directory `.../Battery-Storage-Analysis-Tool` and the Output path should be `.../Battery-Storage-Analysis-Tool/output`. Refer to the existing path as an example.

6. Run the script

```bash
python main.py
```


