# DSS: Decision Support System for Pavement Management

The DSS (Decision Support System) for Pavement Management is a specialized tool designed to assist road agencies in making informed maintenance decisions. It leverages data from multiple organizations involved in the Pavement Management System (PMS) to provide comprehensive insights.

# Setting Up and Running DSS

## Prerequisites

- **Python**: DSS uses Python for its operations. The recommended version is Python 3.6.1 (64-bit), which can be downloaded for free from [Python's official website](https://www.python.org/).

- **Anaconda**: Anaconda facilitates the management of the Python environment. Please download the version of Anaconda that matches your installed Python version. Anaconda is available for free at [Anaconda's official website](https://www.anaconda.com/).

## Environment Setup

### Step 1: Install Dependencies

Open your command prompt and execute the following commands:

```bash
py -m pip install pandas sklearn matplotlib seaborn pydotplus IPython graphviz xlrd openpyxl PyQt5
```

### Step 2: Install Graphviz

Open Anaconda Prompt and execute the following command:

```bash
conda install -c anaconda graphviz
```

### Step 3: Set up Graphviz Environment Variable

1. Open Control Panel
2. Click System and Security
3. Click System
4. Click Advanced system settings
5. Click Environment Variables
6. Select Path, and click Edit
7. Click New, and add the following code, replacing "location of Anaconda" with your Anaconda installation path: 

```bash
<location of Anaconda>\Library\bin\graphviz
```
For example, if you installed Anaconda in C:\Users, the new path for the environment variable would be C:\Users\Anaconda3\Library\bin\graphviz.

After setting up Graphviz as the environment variable, you may need to restart Python to ensure the changes take effect.

## Data Preparation (Only for Software Developers)

Ensure your input data is an .xlsx file named 'DTC.xlsx', located in the same directory as Main.py.

## Running the Software

1. Open IDLE (python 3.6 64-bit) or the version that corresponds to your installed Python.
2. Click File > Open...
3. Navigate to the 'SimulationTool' directory and select 'Main.py' to open.
4. Run 'Main.py'.

## Important Notes

- Running the software will overwrite the original output files. To preserve existing outputs, rename them or save them in a different directory.
- Ensure that the 'RESULT.Report.xlsx' output file is not open when running the software.

# DSS UI

1. Select a target road for analysis.
2. Select a specific date to evaluate road performance.
3. Choose a performance indicator to assess whether the road meets maintenance requirements.
4. Select a performance model to predict road maintenance needs.
5. Run the software - it only takes a few seconds to yield results.
6. User settings will be displayed.
7. Partial results will be displayed. For complete results, refer to 'RESULT_Decision Tree.png' and 'RESULT_Report.xlsx'.
