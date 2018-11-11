###################################################
# Author: Jingyuan Gao				  #
# Created on 24 October 2018	 		  #
#						  #
# Created for plotting the Jackknife analysis for #
# max_depth of the BDT.				  #
###################################################


import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

x = range(6, 12)  # max depth of the BDT
# Training fraction 50%
y_5 = [0.782, 0.777, 0.787, 0.788, 0.783, 0.782]  # mean area under the roc curve (test)
y_5_err = [0.00324, 0.00617, 0.00382, 0.00681, 0.00667, 0.00492]  # std of the area

x = range(4,12)
#Training fraction 70%
y_7_train = [.695,.715,.712,.714,.708,.702,.693,.]
y_7_test = [.779,.792,.78,.733,.769,.771,.794]
y_7_train_err = [.0106,.0177,.0126,.0111,.00664,.0136,.0179]
y_7_test_err = [.0436,.0355,.0359,.0603,.0267,.0445,.0297]

plt.errorbar(x,y_5,y_5_err,fmt='o')
plt.set_title("Jackknife Analysis (Training:50%,Testing:40%), Jackknife Cut: 0.5")
plt.xlabel("Max Depth")
plt.ylabel("Area Under the Roc Curve")
plt.savefig("Jackknife Analysis")
