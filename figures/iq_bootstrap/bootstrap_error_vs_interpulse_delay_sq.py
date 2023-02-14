
import utils.tool_belt as tool_belt
import utils.kplotlib as kpl
from utils.kplotlib import KplColors
import numpy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils.tool_belt import NormStyle

kpl.init_kplotlib()

t = [50, 75, 100, 125, 150, 175]

phi_list = [-0.02941, -0.13213, -0.21603, -0.20053, 
            -0.19402, -0.14746]
phi_err_list = [0.05292, 0.05069, 0.05,  0.04856,
                0.04973,  0.05162]
e_y_list = [0.00820, -0.04418, -0.15186, -0.1535,
            -0.17161,  -0.15443]
e_y_err_list = [0.04433, 0.03245,  0.01299, 0.030,
                0.00962, 0.02486]
e_z_list = [0.02848, 0.02355, -0.08316, -0.153,
            -0.16528, -0.20896]
e_z_err_list = [0.06720, 0.0724, 0.06847, 0.0667,
                0.0689, 0.06556]

chi_list = [-0.05295, -0.09111, -0.18256, -0.2341,
            -0.16257, -0.16571]
chi_err_list = [0.06758, 0.06888, 0.07112, 0.0642,
                0.0675 ,0.0673]
v_x_list = [0.26535,  0.26172, 0.08055,  0.095,
            0.04680,  -0.030 ]
v_x_err_list = [0.04565,  0.0266, 0.0127, 0.026,
                0.01,  0.0196]
v_z_list = [-0.00824,  0.19023, 0.20458, 0.2339,
            0.22326, 0.1620]
v_z_err_list = [0.07209, 0.06775, 0.067, 0.0654,
                0.06746, 0.071]

phi_p_list = [-0.0275, 0.05707, 0.05410, 0.0729,
              0.00905,  0.05]
phi_p_err_list = [0.0529, 0.05, 0.05,  0.0485,
                  0.04973, 0.051]
# e_y_p_list = []
# e_y_p_err_list = []
e_z_p_list = [-0.18904, -0.06775, -0.03927,  -0.01644,
              -0.02615, 0.07694]
e_z_p_err_list = [0.04927, 0.054,  0.057, 0.052,
                  0.05562,  0.0526 ]

chi_p_list = [-0.03795, -0.03691, 0.00712, 0.05740,
              -0.02920,  0.00979 ]
chi_p_err_list = [0.05055, 0.054,  0.057, 0.050,
                  0.05562,  0.0526 ]
v_x_p_list = [ 0.33757, 0.26061, 0.21926, 0.20082,
              0.18538, 0.12225]
v_x_p_err_list = [ 0.04927, 0.054, 0.057, 0.052,
                  0.05562,  0.0526 ]
v_z_p_list = [0.19110, 0.03411, 0.00576,  -0.01820,
              0.03535, -0.05402]
v_z_p_err_list = [0.04927, 0.054, 0.057, 0.052,
                  0.05562, 0.0526 ]

fig, ax = plt.subplots()
ax.set_xlabel('Inter-Pulse duration (ns)')
ax.set_ylabel("Error")
ax.set_title('Pi_x pulse errors')

kpl.plot_points(ax, t, phi_list, yerr=phi_err_list, label = 'phi')
kpl.plot_points(ax, t, e_y_list, yerr=e_y_err_list, label = 'e_y')
kpl.plot_points(ax, t, e_z_list, yerr=e_z_err_list, label = 'e_z')

ax.legend()
fig, ax = plt.subplots()
ax.set_xlabel('Inter-Pulse duration (ns)')
ax.set_ylabel("Error")
ax.set_title('Pi_y pulse errors')
kpl.plot_points(ax, t, chi_list, yerr=chi_err_list, label = 'chi')
kpl.plot_points(ax, t, v_x_list, yerr=v_x_err_list, label = 'v_x')
kpl.plot_points(ax, t, v_z_list, yerr=v_z_err_list, label = 'v_z')

ax.legend()
fig, ax = plt.subplots()
ax.set_xlabel('Inter-Pulse duration (ns)')
ax.set_ylabel("Error")
ax.set_title('Pi/2_x pulse errors')
kpl.plot_points(ax, t, phi_p_list, yerr=phi_p_err_list, label = 'phi_p')
kpl.plot_points(ax, t, e_z_p_list, yerr=e_z_p_err_list, label = 'e_z_p')

ax.legend()
fig, ax = plt.subplots()
ax.set_xlabel('Inter-Pulse duration (ns)')
ax.set_ylabel("Error")
ax.set_title('Pi/2_y pulse errors')
kpl.plot_points(ax, t, chi_p_list, yerr=chi_p_err_list, label = 'chi_p')
kpl.plot_points(ax, t, v_x_p_list, yerr=v_x_p_err_list, label = 'v_x_p')
kpl.plot_points(ax, t, v_z_p_list, yerr=v_z_p_err_list, label = 'v_z_p')

ax.legend()