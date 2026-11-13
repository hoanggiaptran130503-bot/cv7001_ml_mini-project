#==DETECTION FROM TESTING DATA==
import sys
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error

#=INITIAL INFORMATION==========================================================
ROOT = Path(__file__).resolve().parent.parent
case_no = int(sys.argv[1]) if len(sys.argv) > 1 else int(input("Which case you want to calculate (1–5): "))
if case_no not in range(1, 6):
    raise ValueError("Case must be from 1 to 5")

configs = ('B',)
bcs = ('SS',)

cases = []
for config in configs:
        for bc in bcs:
            cases.append(config + '_' + bc)

case_dir = ROOT / "Detection" / f"Case {case_no}"
result_dir = ROOT / "Result" / f"Case {case_no}"
result_dir.mkdir(exist_ok=True)

# model_folder = 'Model'
# data_folder = 'Detection'
nodmele = '1'
samples = np.array([0]) ###
damage_element = np.array([1, 2, 3]) ###

showfigure = 'y'
savefigure = 'y'
if savefigure == 'y':
        save_fig_folder = 'Converging_Figure'

#=LOAD THE TRAINED MODEL=======================================================
Models = {}
X = {} ; x = {}
y_pred = {}
for case in cases:
        Models[case] = load_model(ROOT / "Model" / "B_SS.keras")

        X_frame = pandas.read_excel(case_dir / f"X_C{case_no}.xlsx")
        X[case] = X_frame.values ; X[case] = np.delete(X[case], 0, 1)
        x[case] = np.atleast_2d(X[case][samples,:])
    
        y_pred[case] = Models[case](x[case])

#=LOAD SCENARIOS DATA==========================================================
#-Input data-------------------------------------------------------------------
Y_frame = pandas.read_excel(case_dir / f"Y_C{case_no}.xlsx")
    
#Y_frame = pandas.read_excel(r"D:\NCKH_DATN\ANNs\Detection-Euclid\Case 5\Y_C5.xlsx")
Y = Y_frame.values
Y = np.delete(Y,0,1)
y = np.atleast_2d(Y[samples,:])

#=RESULTS PRESENTING & RECORDING===============================================
# Showing the results
if showfigure == 'y':
        plt.close()
        no_element = Y.shape[1]

        mpl.rc('font',family='Times New Roman')
        axis_x = np.arange(no_element) + 1.25
        axis_y = np.arange(0, 1.1, 0.1)
        width = 0.5
        no_subplts = int(len(bcs))
     
        for config in configs:
            fig, axs = plt.subplots(no_subplts, 1, num=config, figsize=(12, 6), dpi=600) #, sharex='col', sharey='row')
            plt.subplots_adjust(hspace=0.4, top=0.95, bottom=0.1, left=0.07, right=0.37)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            for bc in bcs:
                rects1 = axs.bar(axis_x - width, y[0], width, color='darkorange', label='Actual')
                rects2 = axs.bar(axis_x, y_pred[config + '_' + bc][0], width, color='lightskyblue', label='Prediction')
                for rect in rects1:
                       height = rect.get_height()
                       axs.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                                '{:.3f}'.format(height), ha='center', va='bottom')
           
                for rect in rects2:
                       height = rect.get_height()
                       axs.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                                '{:.3f}'.format(height), ha='center', va='bottom')
                
                axs.set_xlim(auto=True)
                axs.set_xlabel('Feature', fontsize=12, fontweight='bold', labelpad=10)
                axs.set_ylabel('Normalise output value', fontsize=12, fontweight='bold', labelpad=10)
                axs.set_xticks(damage_element)
                axs.set_xticklabels([str(j) for j in damage_element], fontsize=14)
                axs.set_yticks(axis_y)
                axs.set_yticklabels(['{:.1f}'.format(j) for j in axis_y], fontsize=12)
                mse_value = mean_squared_error(y[0], y_pred[config + '_' + bc][0])
                axs.set_title('MSE = {:.3e}'.format(mse_value), fontsize=12, fontweight='bold', pad=15)
                axs.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
                axs.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
                if samples in [0, 85, 99, 121]:
                    axs.legend(fontsize=14, loc='upper right')
                elif samples == 106:
                    axs.legend(fontsize=14, loc='upper center')
                else:
                    axs.legend(fontsize=14, loc='upper right')

            if savefigure == 'y':
                fig.savefig(result_dir / f"C{case_no}.jpeg", dpi=600)

        plt.show()

import pandas as pd
result_df = pd.DataFrame({
    'Element': range(1, len(y[0]) + 1),  
    'Actual': y[0].flatten(),             
    'Predicted': y_pred[config + '_' + bc][0].numpy().flatten()
    })

result_df.to_excel(result_dir / f"C{case_no}.xlsx")
print("Results have been exported")

print('Value predict',y_pred[config + '_' + bc][0])