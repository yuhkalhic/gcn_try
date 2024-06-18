# gcn_try
try to solve the problem in https://github.com/fzb0316/iDC-MlSys_interview


原始
prepross_time: 1.15633200
edgeNorm_time: 0.92946500
XW1_time: 4.63032800
AX1_time: 0.45136800
ReLU_time: 0.12797500
XW2_time: 0.42807400
AX2_time: 0.23738700
LogSoftmax_time: 0.11557700
max_sum_time: 0.01300600
-16.68968964
total time: 8.15652200

openmp
prepross_time: 0.85309200
edgeNorm_time: 0.80738600
XW1_time: 3.09611700
AX1_time: 0.25296600
ReLU_time: 0.08009900
XW2_time: 0.27652500
AX2_time: 0.16179000
LogSoftmax_time: 0.08197900
max_sum_time: 0.01334200
-16.68968964
total time: 5.66738300

cuda
prepross_time: 119.51983200
edgeNorm_time: 0.00924000
XW1_time: 29.44687200
AX1_time: 0.00599300
ReLU_time: 0.00067700
XW2_time: 0.00226600
AX2_time: 0.00103000
LogSoftmax_time: 0.00057200
max_sum_time: 0.01171300
0.00000000
total time: 149.02683800

cuda并且使用cuBLAS 库和 cuSPARSE 库
prepross_time: 141.08498300
edgeNorm_time: 0.01064200
XW1_time: 56.09808100
AX1_time: 0.11928300
ReLU_time: 0.04123400
XW2_time: 0.30044200
AX2_time: 0.06697900
LogSoftmax_time: 0.04444900
max_sum_time: 0.01383100
-16.68649483
total time: 197.81729500
