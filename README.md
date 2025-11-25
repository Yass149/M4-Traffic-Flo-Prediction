# 🚦 CSMAD CW2: M4 Traffic Flow Prediction (Group 08)

**Team:** Yassine, Amir, Ahmed  
**Deadline:** Tuesday, 16 December 2025 (12:00 Noon)  
**Goal:** Predict traffic flow using Regression & Time Series models.

---

## 🛑 CRITICAL RULES (Read Before Coding)

**1. DO NOT Push Directly to `main`**
* **Risk:** If you break `main`, you break the project for everyone.
* **Rule:** ALWAYS work on your own branch.
    * `git checkout -b amir-weather-dev`
    * `git checkout -b ahmed-dates-dev`
* **Merge:** Only merge to `main` when your code is working and tested.

**2. DO NOT Write Analysis Code in the Notebook**
* **Risk:** We will lose marks. The brief requires code to be in **external modules**.
* **Rule:** Write your logic in the `modules/` folder (`.py` files). The Notebook is ONLY for importing modules and displaying results.

**3. DO NOT Edit the Master Notebook Simultaneously**
* **Risk:** Git cannot merge Jupyter Notebooks. If two people save `Group_08_CSMAD_CW2.ipynb` at the same time, the file will corrupt.
* **Rule:** Use a temporary "scratchpad" notebook (e.g., `test_amir.ipynb`) to test your code. **Yassine** will manage the final master notebook updates.

---

## 🍏 Mac Setup Instructions (3 Steps)

### Step 1: Clone the Repository
1.  Open **VS Code**.
2.  Press **Cmd + Shift + P** (Command Palette).
3.  Type `Git: Clone` and press Enter.
4.  Paste our GitLab URL: `[INSERT URL HERE]`
5.  Select a folder to save it.
6.  Enter your **University Username** and **Password** if prompted.

### Step 2: Verify the Data
*The data is already included in this repository.*
1.  Open the file explorer in VS Code.
2.  Check that you see the `data/` folder with these contents:
    * `data/traffic/` (8 CSV files)
    * `data/weather/` (4 CSV files)
    * `data/dates/` (PDF calendars)

### Step 3: Check Your Python Environment
1.  Open any `.py` file in the `modules/` folder.
2.  Look at the bottom-right corner of VS Code.
3.  Ensure it says **3.11.7** (or `base (Anaconda)`).
    * *If not:* Click it -> Select Interpreter -> Choose the correct Conda environment.

---

## 📂 Project Structure (Strict Requirement)

We must follow this structure for the "Technical Implementation" marks.

    ├── modules/                  <-- ALL LOGIC GOES HERE (The Python Package)
    │   ├── __init__.py           <-- Makes this importable
    │   ├── data_loader.py        <-- Functions to load/merge Traffic CSVs
    │   ├── preprocessing.py      <-- Functions to decode Weather strings
    │   ├── features.py           <-- Functions to identify School Holidays
    │   ├── modelling.py          <-- Classes for Regression & Time Series
    │   └── visualization.py      <-- Plotting functions
    │
    ├── data/                     <-- Raw Data (Do not edit these files)
    │
    ├── Group_08_CSMAD_CW2.ipynb  <-- FINAL SUBMISSION NOTEBOOK (Presentation Only)
    └── .gitignore                <-- Keeps the repo clean

---

## 📅 Work Division (To Be Agreed)

Please update this table once we agree on roles.

| Role | Responsibility | Assigned To |
| :--- | :--- | :--- |
| **Traffic Handler** | Write `data_loader.py` to merge 8 traffic CSVs & fix timestamps. | *Pending* |
| **Weather Decoder** | Write `preprocessing.py` to decode FM-12 strings (read `CSV_HELP.pdf`). | *Pending* |
| **Date Engineer** | Parse PDFs in `data/dates/` to create a "Day Type" feature (School vs Holiday). | *Pending* |
| **Modeller** | Build Regression & Time Series models in `modelling.py`. | *Joint Effort* |

---

## 📝 Coding Standards (The Contract)

**IMPORTANT:** To ensure our code works together when we merge, everyone must follow these naming rules:

1.  **Time Column:** Always name the datetime column `timestamp`.
2.  **Target Column:** Always name the traffic count column `total_volume`.
3.  **Time Interval:** All data must be resampled to **15-minute intervals**.
4.  **Docstrings:** Every function **MUST** have a docstring (Required for marks).

**Docstring Example:**

    def my_function(df):
        """
        Brief description of what the function does.
        
        Args: 
            df (pd.DataFrame): Input dataframe.
            
        Returns: 
            pd.DataFrame: Processed dataframe.
        """
        pass

---

## 🚀 How to Start Working

1.  **Switch Branch:** `git checkout -b [your-name]-dev`
2.  **Code:** Write your function in the `modules/` folder.
3.  **Test:** Create a file named `test_[name].ipynb` (e.g. `test_amir.ipynb`) to run your code.
4.  **Push:**
    * Click Source Control (Graph icon).
    * Stage your changes (+).
    * Message: "Implemented weather decoder".
    * Commit & Sync (Push).