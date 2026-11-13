# Reinforced-concrete wall design quantities with a DFNN

**Author:** GIAP Tran Hoang

Machine Learning Mini-Project. A deep feed-forward neural network (DFNN) estimates three reinforced-concrete wall design quantities from three inputs for cases from one building. The data and initial modelling workflow were adapted from the author's capstone project. The saved model and five example design cases are included so the prediction script can be checked without retraining.

## Problem and data

| Input | Meaning |
| --- | --- |
| $P$ | Axial force |
| $M_3$ | Bending moment |
| $L_w$ | Wall length |

| Output | Meaning |
| --- | --- |
| $B$ | Edge-region length |
| $t$ | Wall thickness |
| $A_{s,\mathrm{prov}}$ | Provided reinforcement area |

The training inputs and design targets were prepared from the author's finite-element building analysis and capstone reinforced-concrete design workflow. They are provided in `Code/Input/CSV/Input_Python.csv` and `Code/Input/CSV/Output_Python.csv`; supporting Excel files are in `Code/Input/Excel/`. Each CSV has 155 data rows and four columns. The first column is an index-like column that the training script discards; the remaining three columns are read in the order shown above.

The CSV and case files contain already-prepared numerical values. The supplied scripts do not implement the original normalization procedure, convert predictions back to physical units, or perform Eurocode design checks. Use inputs on the same scale and in the same column order as the supplied files. Do not interpret the three predicted numbers as a completed structural design.

The first 139 rows (90%) are used for training and the last 16 rows (10%) for validation. The split follows file order; the script does not shuffle before splitting. Five additional design cases from the same building are supplied in `Code/Detection/Case 1` through `Case 5`. Their design targets are used only for comparison with predictions and are not used in fitting the model.

## Model and evaluation

`Code/Code/DevelopDNNsModel.py` trains a Keras Sequential network with three hidden layers of 72 neurons each, ELU activation ($\alpha=0.5$), and a linear three-neuron output layer. It uses Adamax with learning rate 0.002, mean-squared-error (MSE) loss, batch size 1, a maximum of 1,500 epochs, and early stopping monitored on validation loss with patience 1,000 and restoration of the best weights.

The saved `Code/Model/B_SS.keras` model gives validation MAE **0.00917** and validation MSE **1.82 × 10⁻⁴** on the 16 held-out rows, on the scale of the supplied targets. `Code/Code/History/B_SS.csv` and `Code/Converging_Figure/` contain the recorded training curves.

`Code/Code/Detection_Presentation.py` runs **one case per invocation**. It reads `X_Cn.xlsx` and `Y_Cn.xlsx` from `Code/Detection/Case n/`, uses the first data row of each, and compares the saved model prediction with the supplied design target. The outputs are `Code/Result/Case n/Cn.xlsx` (actual and predicted numerical values) and `Cn.jpeg` (comparison chart). For each output, the absolute relative difference can be calculated as

$$
100\,\frac{|y_{\mathrm{design}}-\hat y_{\mathrm{DFNN}}|}{|y_{\mathrm{design}}|}\%.
$$

These five comparisons are separate from the 10% validation-set MAE and MSE. They test cases within the same building and do not establish performance for other buildings or guarantee Eurocode compliance. The report in `Latex/ml_mini_project_report_thg.pdf` discusses the results and limitations.

## Repository layout

```text
README.md
requirements.txt
Picture1.png
Code/
  Code/
    DevelopDNNsModel.py
    Detection_Presentation.py
    History/B_SS.csv
  Input/CSV/                     # 155 paired training/validation records
  Input/Excel/                   # supporting source workbooks
  Detection/Case 1/ ... Case 5/ # X_Cn and design targets Y_Cn
  Model/B_SS.keras               # saved Keras model
  Result/Case 1/ ... Case 5/    # supplied case outputs
  Converging_Figure/             # recorded training curves
Latex/
  ml_mini_project_report_thg.tex
  ml_mini_project_report_thg.pdf
```

The local Python environment (`.venv`) and LaTeX auxiliary files are excluded by `.gitignore`. Keep the repository structure shown above: both Python scripts resolve paths from their own location, so no machine-specific drive path needs editing.

## Software setup

The project was run on Windows with **Python 3.12.10**, TensorFlow **2.21.0**, Keras **3.15.1**, and the package versions pinned in `requirements.txt`. From a PowerShell terminal opened at the repository root (the folder containing this README), run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is not available, install Python 3.12 first. Do not copy the `PS C:\...>` prompt into PowerShell; type only the commands shown.

## Check the supplied model on an example case

From the repository root:

```powershell
.\.venv\Scripts\python.exe Code\Code\Detection_Presentation.py 1
```

Replace `1` with `2`, `3`, `4`, or `5` to run another case. If no case number is supplied, the script asks for one. The range is restricted to 1–5. **Close the chart window** after inspecting it: the script exports the Excel result after `plt.show()` returns. Each run overwrites the corresponding `Cn.xlsx` and `Cn.jpeg` in `Code/Result/Case n/`; copy supplied results elsewhere first if you want to preserve them.

The script requires both `X_Cn.xlsx` and `Y_Cn.xlsx` because it is a prediction-and-comparison demonstration. It is not an unlabeled-input design tool.

## Retrain

From the repository root:

```powershell
.\.venv\Scripts\python.exe Code\Code\DevelopDNNsModel.py
```

This reads the two CSV files, applies the ordered 139/16 split, trains the model, prints training and validation evaluations, and writes `Code/Model/B_SS.keras`, `Code/Code/History/B_SS.csv`, and two PDF convergence plots. Close the plot windows to finish the process. **Retraining overwrites these saved files.** The script does not set a random seed, so a new run can produce different weights, metrics, and case predictions. For checking the submitted case results, use the supplied model rather than retraining first.

## Provenance and disclosure

The dataset, finite-element inputs, design targets, and initial model code were adapted from the author's capstone project; they are not a public benchmark dataset. This repository's README and path organization were prepared with generative-AI assistance. TensorFlow/Keras, pandas, NumPy, Matplotlib, scikit-learn, and openpyxl are third-party software. No externally pretrained model was used; `B_SS.keras` contains weights trained for this project.

Before making the repository public, the author should confirm that the capstone data and supporting Excel workbooks may be shared. The exact normalization steps and applicable Eurocode edition/National Annex are not encoded in the scripts and should be documented separately if the project is extended to physical-unit design.
