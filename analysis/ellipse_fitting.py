# -*- coding: utf-8 -*-
"""
Trial ellipse fitting

Created October 3rd, 2022

@author: mccambria
"""

### Imports
import sys
import numpy as np
from numpy.core.shape_base import block
from scipy.optimize import root_scalar, minimize_scalar, minimize, brute
import utils.tool_belt as tool_belt
import time
import matplotlib.pyplot as plt
from utils import common
from utils import kplotlib as kpl
from utils.kplotlib import KplColors
from scipy.optimize import curve_fit
import csv
import sys
import random
from pathos.multiprocessing import ProcessingPool as Pool

cent = 0.5
amp = 0.65 / 2
num_atoms = 1000

num_ellipse_samples = 1000
theta_linspace = np.linspace(0, 2 * np.pi, num_ellipse_samples, endpoint=False)
cos_theta_linspace = np.cos(theta_linspace)
sin_theta_linspace = np.sin(theta_linspace)

# region Functions


def corr_gaussian(data_point, ellipse_sample):
    data_point_x, data_point_y = data_point
    ellipse_sample_x, ellipse_sample_y = ellipse_sample
    varx = ellipse_sample_x * (1 - ellipse_sample_x) / num_atoms
    vary = ellipse_sample_y * (1 - ellipse_sample_y) / num_atoms
    sdx = np.sqrt(varx)
    sdy = np.sqrt(vary)
    z = (((data_point_x - ellipse_sample_x) / sdx) ** 2) + (
        ((data_point_y - ellipse_sample_y) / sdy) ** 2
    )
    return (1 / (2 * np.pi * sdx * sdy)) * np.exp(-z / 2)


def image_cost(phi, image):
    """ """

    image = np.flipud(image)
    ellipse_samples = ellipse_point(theta_linspace, phi)
    ellipse_samples_x = ellipse_samples[0]
    ellipse_samples_y = ellipse_samples[1]

    integrand = 0
    for sample in zip(ellipse_samples_x, ellipse_samples_y):
        sample_x = round(sample[0] * 100)
        sample_y = round(sample[1] * 100)
        integrand += image[sample_x, sample_y]

    # integrand = np.sum(
    #     image[circle_samples_x[valid_samples], circle_samples_y[valid_samples]]
    # )

    cost = integrand / num_ellipse_samples
    cost = 1 - cost  # Best should be minimum

    return cost


def corr_cost(phi, points):
    """ """

    ellipse_samples = ellipse_point(theta_linspace, phi)
    ellipse_samples = np.column_stack(ellipse_samples)

    integrand = 1
    for point in points:
        point_theta_probs = [
            corr_gaussian(point, el) for el in ellipse_samples
        ]
        point_prob = np.sum(point_theta_probs) / num_ellipse_samples
        integrand *= point_prob

    # cost = integrand / (len(ellipse_samples) * len(points))
    cost = integrand
    cost = 1 - cost  # Best should be minimum

    return cost


def ellipse_point(theta, phi):
    return (cent + amp * np.cos(theta + phi), cent + amp * np.cos(theta - phi))


def theta_cost(phi, ellipse_point, test_point):

    theta_cost = lambda theta: sum(
        [(el[0] - el[1]) ** 2 for el in zip(ellipse_point(theta), test_point)]
    )

    # Guess theta by assuming theta and the ellipse amplitude are free and
    # solving for the position of the point on the ellipse
    # x = cent + amp * np.cos(theta + phi)
    # y = cent + amp * np.cos(theta - phi)
    # x-y = amp * (np.cos(theta + phi) - np.cos(theta - phi))
    #     = -2 amp sin(theta) sin(phi)
    # amp = (x-y) / (-2 sin(theta) sin(phi))
    # x+y = 2cent + 2 amp cos(theta) cos(phi)
    #     = 2cent - (x-y) cot(theta) cot(phi)
    x, y = test_point
    # Avoid divide by zero
    guess_denom = np.tan(phi) * (2 * cent - (x + y))
    if guess_denom < 0.01:
        base_guess = np.pi / 2
        # Check the guess and its compliment
        guesses = [
            base_guess % (2 * np.pi),
            (2 * np.pi - base_guess) % (2 * np.pi),
        ]
    else:
        guess_arg = (x - y) / guess_denom
        base_guess = np.arctan(abs(guess_arg)) % np.pi
        # Check the guess and its compliments
        guesses = [
            base_guess % (2 * np.pi),
            (np.pi - base_guess) % (2 * np.pi),
            (base_guess + np.pi) % (2 * np.pi),
            (2 * np.pi - base_guess) % (2 * np.pi),
        ]

    best_cost = None
    for guess in guesses:
        res = minimize(theta_cost, guess)
        opti_theta = res.x
        opti_cost = res.fun
        if (best_cost is None) or (opti_cost < best_cost):
            best_cost = opti_cost
    return best_cost


def ellipse_cost(phi, points, debug=False):

    if debug:
        test = 1

    ellipse_lambda = lambda theta: ellipse_point(theta, phi)

    # The cost is the rms distance between the point and the ellipse
    # Finding the closest distance between an arbitary point and an ellipse,
    # of course, turns out to be a hard problem, so let's just run another
    # minimization for it
    theta_cost_lambda = lambda point: theta_cost(phi, ellipse_lambda, point)
    with Pool() as p:
        theta_costs = p.map(theta_cost_lambda, points)

    cost = sum(theta_costs)
    num_points = len(points)
    cost = np.sqrt(cost / num_points)

    return cost


def gen_ellipses():
    # phis = [0, np.pi / 2, np.pi / 4]
    phis = [0.01]
    num_points = 1000
    ellipses = []
    for phi in phis:
        theta_vals = 2 * np.pi * np.random.random(size=num_points)
        ellipse = [[phi]]
        ellipse_lambda = lambda theta: ellipse_point(theta, phi)
        points = []
        for ind in range(num_points):
            theta = theta_vals[ind]
            point = ellipse_lambda(theta)
            noisy_point = (
                np.random.binomial(num_atoms, point[0]) / num_atoms,
                np.random.binomial(num_atoms, point[1]) / num_atoms,
            )
            points.append(noisy_point)
        ellipse.extend(points)
        ellipses.append(ellipse)
    return ellipses


def populate_imported_phis(path, files):

    phis_by_file = []

    for el in files:
        sub_phis = []
        with open(path / el, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                float_row = [float(el) for el in row]
                sub_phis.append(float_row[0])
        phis_by_file.append(sub_phis)

    # Re-sort by ellipse
    num_files = len(files)
    num_ellipses = len(phis_by_file[0])
    phis = []
    for ellipse_ind in range(num_ellipses):
        sub_phis = []
        for sub_ind in range(num_files):
            sub_phis.append(phis_by_file[sub_ind][ellipse_ind])
        phis.append(sub_phis)

    return phis


def import_ellipses(path):

    x_vals = []
    y_vals = []
    phis = []
    file_x = "testingX.csv"
    file_y = "testingY.csv"
    phi_files = ["testingPhi_d.csv", "phi_LS.csv", "phi_NN.csv"]

    trans_x = zip(*csv.reader(open(path / file_x, newline="")))
    for row in trans_x:
        float_row = [float(el) for el in row]
        x_vals.append(float_row)

    trans_y = zip(*csv.reader(open(path / file_y, newline="")))
    for row in trans_y:
        float_row = [float(el) for el in row]
        y_vals.append(float_row)

    phis = populate_imported_phis(path, phi_files)

    ellipses = []
    for ellipse_x, ellipse_y, ellipse_phi in zip(x_vals, y_vals, phis):
        points = list(zip(ellipse_x, ellipse_y))
        ellipse = [ellipse_phi]
        ellipse.extend(points)
        ellipses.append(ellipse)

    return ellipses


# endregion

# region Main


def main(path):

    ellipses = import_ellipses(path)
    # ellipses = gen_ellipses()
    theta_linspace = np.linspace(0, 2 * np.pi, 100)
    phi_errors = []
    covariances = []
    cov_phis = []
    true_phis = []
    image_phis = []

    do_plot = False

    # ellipses = ellipses[::10]
    # ellipses = [ellipses[-1]]
    num_points = 100
    x_locs = np.linspace(0, 1, num_points)
    y_locs = np.linspace(0, 1, num_points)
    for ind in range(len(ellipses)):

        print(f"Ellipse index: {ind}")
        ellipse = ellipses[ind]
        phi_errors_sub = []
        # for ellipse in ellipses[0]:
        ellipse_phis = ellipse[0]
        true_phi = ellipse_phis.pop(0)
        true_phis.append(true_phi)

        algo_phis = ellipse_phis
        points = ellipse[1:]
        x_vals = np.array([point[0] for point in points])
        y_vals = np.array([point[1] for point in points])

        # corr = np.corrcoef(x_vals, y_vals)[0, 1]
        # covariances.append(corr)
        # cov_phi = np.arccos(corr) / 2
        # cov_phis.append(cov_phi)

        # image = np.zeros((num_points, num_points))
        # for point in points:
        #     gaussian_lambda = lambda loc_x, loc_y: np.exp(
        #         -(
        #             ((loc_x - point[0]) ** 2)
        #             * np.sqrt(num_atoms)
        #             / (2 * (loc_x * (1 - loc_x))) ** 2
        #         )
        #         - (
        #             ((loc_y - point[1]) ** 2)
        #             * np.sqrt(num_atoms)
        #             / (2 * (loc_y * (1 - loc_y))) ** 2
        #         )
        #     )
        #     gaussian_matrix = [
        #         [gaussian_lambda(x_loc, y_loc) for x_loc in x_locs]
        #         for y_loc in y_locs
        #     ]
        #     gaussian_matrix = np.array(gaussian_matrix)
        #     image += gaussian_matrix
        # image = np.flipud(image)
        # fig, axes_pack = plt.subplots(1, 2)
        # ax0, ax1 = axes_pack
        # ax0.imshow(image)
        # kpl.plot_points(ax1, x_vals, y_vals)

        # plt.show(block=True)
        # continue

        # thresh = 0.1
        # if thresh < true_phi < (np.pi / 2) - thresh:
        #     continue

        # phi_lin = np.linspace(0, np.pi / 2, 50)
        # costs = [corr_cost((phi,), points) for phi in phi_lin]
        # plt.plot(phi_lin, costs)
        # plt.show(block=True)

        bounds = (0, np.pi / 2)
        opti_phi = brute(
            # image_cost,
            corr_cost,
            (bounds,),
            Ns=300,
            # args=(image,),
            args=(points,),
            finish=None,
            # finish=minimize,
            workers=-1,  # Multiprocessing: -1 means use as many cores as available
        )
        # best_cost = None
        # while True:
        #     opti_phi = brute(
        #         # image_cost,
        #         corr_cost,
        #         (bounds,),
        #         Ns=20,
        #         # args=(image,),
        #         args=(points,),
        #         finish=minimize,
        #         workers=-1,  # Multiprocessing: -1 means use as many cores as available
        #     )
        #     new_best_cost = corr_cost(opti_phi, points)

        #     if best_cost is None:
        #         best_cost = new_best_cost
        #     else:
        #         # threshold = 0.0001 * best_cost
        #         threshold = 0.001 * abs(best_cost)
        #         if best_cost - new_best_cost < threshold:
        #             break
        #         best_cost = new_best_cost

        #     bounds_span = bounds[1] - bounds[0]
        #     bounds = (
        #         opti_phi - 0.1 * bounds_span,
        #         opti_phi + 0.1 * bounds_span,
        #     )

        # img = np.zeros((num_points, num_points))
        # ellipse_samples = ellipse_point(theta_linspace, opti_phi)
        # ellipse_samples_column = np.column_stack(ellipse_samples)
        # for ellipse_sample in ellipse_samples_column:
        #     gaussian_matrix = [
        #         [
        #             corr_gaussian((x_loc, y_loc), ellipse_sample)
        #             for x_loc in x_locs
        #         ]
        #         for y_loc in y_locs
        #     ]
        #     img += gaussian_matrix

        # plt.imshow(img)
        # for point in points:
        #     plt.scatter(100 * point[0], 100 * point[1])
        # plt.show(block=True)
        # continue

        # Remove degeneracies
        opti_phi = opti_phi % np.pi
        if opti_phi > np.pi / 2:
            opti_phi = np.pi - opti_phi

        image_phis.append(opti_phi)

        # opti_phi = 0
        algo_phis.insert(0, opti_phi)

        # ellipse_lambda = lambda theta: ellipse_point(theta, opti_phi)
        # x_vals, y_vals = zip(ellipse_lambda(theta_linspace))
        # x_vals = x_vals[0]
        # y_vals = y_vals[0]
        # kpl.plot_line(ax1, x_vals, y_vals)
        # plt.show(block=True)
        # continue

        if do_plot:
            fig, ax = plt.subplots()
            for test_phi in [opti_phi]:
                # Plot the data points
                for point in points:
                    color = KplColors.BLUE.value
                    kpl.plot_points(ax, *point, color=color)
                # Plot the fit
                ellipse_lambda = lambda theta: ellipse_point(theta, test_phi)
                x_vals, y_vals = zip(ellipse_lambda(theta_linspace))
                x_vals = x_vals[0]
                y_vals = y_vals[0]
                kpl.plot_line(ax, x_vals, y_vals)
                kpl.tight_layout(fig)
                cost = corr_cost(test_phi, points)
                # cost = ellipse_cost(test_phi, points, True)
                print(f"Phi: {round(test_phi, 3)}; cost: {round(cost, 6)}")
            plt.show(block=True)
        continue

        # Get the costs
        # test_phis = [true_phi]
        # test_phis.extend(algo_phis)
        # for phi in test_phis:
        #     cost = ellipse_cost(phi, points, True)
        #     print(f"{round(phi, 3)}: {round(cost, 6)}")
        # print()
        phi = opti_phi
        cost = ellipse_cost(phi, points, True)
        if cost > 0.1:
            print("Algorithm did poorly...")
            print(f"Phi: {round(phi, 3)}; cost: {round(cost, 6)}")

        # Get the phi errors
        for phi in algo_phis:
            phi_errors_sub.append(phi - true_phi)
        phi_errors.append(phi_errors_sub)

    fig, ax = plt.subplots()
    kpl.plot_points(ax, true_phis, image_phis)
    kpl.tight_layout(fig)

    plt.show(block=True)

    phi_errors = [ela - elb for ela, elb in zip(image_phis, true_phis)]

    print(phi_errors)

    return

    print(f"Summary for ellipse with true phi {true_phi}")
    # for row in phi_errors:
    #     rounded_errs = [round(el, 6) for el in row]
    #     print(
    #         f"Algo: {rounded_errs[0]}; LS: {rounded_errs[1]}; NN:"
    #         f" {rounded_errs[2]}"
    #     )
    # print()
    # print("RMS phase errors for algorithm, least squares, neural net: ")
    print("RMS phase errors for algorithm")
    phi_errors = np.array(phi_errors)
    rms_phi_errors = np.sqrt(np.mean(phi_errors**2, axis=0))
    print([round(el, 6) for el in rms_phi_errors])


# endregion

if __name__ == "__main__":

    kpl.init_kplotlib()

    home = common.get_nvdata_dir()
    path = home / "ellipse_data"

    # main(path)

    # plt.show(block=True)

    # sys.exit()

    ellipses = import_ellipses(path)

    true_phis = [el[0][0] for el in ellipses]

    # fmt: off
    algo_errs = [-0.002696, -0.005236, -0.002495, -0.010397, -0.005612, 0.004582, -0.001166, -0.003142, -0.00672, 0.019494, 0.006595, 0.00969, -0.001045, -0.005577, 0.002375, -0.020186, -0.016039, -0.007915, 0.011466, -0.007573, -0.012851, -0.00625, 0.010959, 0.007106, -0.005491, 0.013471, 0.00267, 0.002729, -0.003578, -0.006198, 0.014313, -0.007534, 0.003608, 0.002803, -0.029468, -0.002701, 0.004912, -0.004386, 0.003009, -0.002204, 0.001627, -0.003329, -0.006071, 0.001658, -0.005253, -0.004552, -0.018404, -0.000391, 0.009777, -0.008527, -0.000372, -0.009247, 0.003607, 0.004043, -0.002131, 0.004303, -0.000208, -0.006015, 0.014381, -0.009208, 0.011965, -0.002119, 0.000371, 0.013484, -0.003873, -0.006543, 0.001001, 0.004758, 0.008269, -0.011544, 0.005727, 0.021964, -0.008503, -0.008904, 0.005893, 0.00064, -0.005962, -0.009019, -0.004009, 0.006963, 0.000145, -0.008003, 0.010323, 0.011814, 0.001115, -0.010788, 0.003803, -0.013374, 0.007496, 0.003659, 0.007829, 0.005227, 0.005001, -0.00057, 0.024445, 0.001972, 0.010003, 0.011258, 0.000721, 0.007917]
    ls_errs = [-0.006651, -0.025661, -0.003856, 0.01137, 0.011135, 0.02866, -0.010305, 0.014189, -0.041953, 0.028655, 0.006517, 0.020179, -0.006231, -0.010907, 0.010187, -0.014631, -0.014665, -0.007319, 0.027088, -0.006274, -0.027916, 0.017605, -0.007834, 0.014636, -0.002544, 0.019428, -0.003474, 0.019037, -0.006241, -0.025533, -0.006932, -0.007198, -0.00379, 0.006552, -0.057791, 0.034609, 0.007318, 0.00847, -0.035114, -0.001717, -0.022486, -0.023689, 0.011426, -0.002936, 0.006608, 0.017152, -0.019594, 0.013142, 0.012604, -0.007891, -0.00768, -0.011221, -0.005237, 0.008858, 0.015301, 0.00297, -0.038235, -0.02749, 0.008799, 0.013698, -0.013355, 0.014874, 0.010341, 0.019017, -0.019166, 0.001746, -0.007538, 0.017759, 0.008184, -0.005907, 0.006894, 0.029796, -0.043181, 0.016486, -0.014571, 0.012811, -0.025845, -0.025957, -0.008738, 0.038078, -0.003798, -0.002053, 0.037769, 0.017541, -0.029644, -0.01479, 0.035284, -0.017504, 0.006683, 0.001623, -0.023922, 0.015435, 0.023689, -0.003199, 0.048901, 0.002686, -0.009473, 0.012474, -0.018711, -0.007447]
    nn_errs = [-0.014066, 0.013953, -0.172563, 0.013603, 0.01627, -0.023009, -0.085988, -0.039819, -0.007913, 0.018008, -0.004063, -0.00584, -0.188581, -0.15544, 0.03117, 0.014923, -0.050193, -0.022615, -0.002615, -0.021856, -0.001479, -0.020141, -0.031364, -0.060448, -0.039629, -0.105871, 0.008865, 0.003506, 0.081942, 0.002916, 0.022175, -0.034657, 0.026548, 0.004461, -0.025558, -0.004031, 0.024522, 0.016392, -0.006209, 0.034737, 0.022379, 0.011969, -0.021362, -0.026672, 0.02229, -0.016773, -0.032711, 0.008744, -0.224467, -0.028047, -0.032582, 0.011764, 0.01255, -0.008026, 0.024754, 0.039537, 0.015747, 0.005508, 0.063291, -0.020051, 0.018704, -0.006485, -0.12445, -0.020867, -0.088132, -0.142866, 0.064274, -0.040068, -0.115205, 0.069389, -0.097154, -0.090545, 0.006695, -0.011963, 0.011854, -0.003998, 0.018181, -0.140684, -0.00384, -0.001536, -0.036581, -0.009636, -0.003184, -0.030662, 0.019089, -0.049583, -0.010491, -0.099624, -0.001683, -0.015948, -0.029186, -0.056644, 0.00774, 0.027986, 0.015959, -0.106259, -0.024237, -0.018905, -0.053612, 0.035453]
    corr_errs = [0.025123958742006236, -0.014869564576668992, -0.034285634271992826, 0.0012830802773810546, -0.011119612184689376, 0.01341739247644605, -0.039752726230330104, -0.0013671677486478684, -0.00746382736564577, -0.00484106136879453, 0.02832041028889154, -0.06375076231478857, -0.08557881382382082, -0.06117408602601393, 0.1120558998725526, 0.06845208311342321, -0.061114972164303616, 0.04973095143537398, 0.010578632913241959, -0.06497858533377188, -0.004782830294254525, -0.01563794100660186, -0.04733573419323678, -0.06258971127051971, -0.03131546645306493, -0.006963848250139737, 0.003424439527336176, 0.01171670142630818, 0.05762413521338994, -0.013187150629633937, 0.0037907747218515198, 0.033423931537272966, 0.030941368697900984, 0.005685296955907304, -0.03841905964313419, 0.022388038445002374, 0.003801580858107223, 0.02917019248669933, -0.027185271911346343, 0.055429510718068675, 0.005300080256147366, 0.014183004126155474, -0.004936784005852318, -0.02713496477876043, 0.017314548074129466, -0.035081139275089074, 0.005924533470652138, 0.02713890015707951, -0.15280740659650327, 0.006325651359583961, 0.03525771032806557, 0.051819752692171095, 0.07176171696645806, 0.04307597825284659, 0.016657393723439717, 0.07239620589293683, -0.021115135908077898, -0.009232759241348365, 0.07923182688724739, -0.01514975969636563, 0.02159736114663291, 0.018385914865301667, -0.1192219291627184, 0.07993637695121558, -0.09346198830345509, -0.037918606028478186, 0.0961599822942596, 0.0053951292663565464, -0.057666178669276646, 0.09355654689332371, 0.04108339525029803, -0.03916218195323701, -0.020798369812875306, -0.0020074630047057784, 0.005111936961794239, -0.017590898072012917, 0.009416729514774236, -0.15942550527794874, 0.11139493469343775, 0.00532015339125827, -0.021138001272778872, 0.027444357156846233, 0.0197185305049686, 0.032679176715255, -0.017356657103021478, 0.004855450414589346, -0.035271105091023436, -0.11342516109354794, -0.036995524516857226, -0.01267403781053078, -0.04868194689561256, -0.030833601058370408, 0.027836465226693613, 0.0055433895091344665, 0.029923348715277943, 0.0018076366414709888, -0.006199321767995425, 0.058419539019826994, -0.09275353655292218, 0.0709073062290646]
    image_errs = [0.0071748023003264105, 0.06726295767556012, 0.0017847463537432606, -0.045689813547463626, -0.041794497031525435, -0.10461077449878407, 0.005590792136189071, -0.021348543830269073, 0.11776812648480739, -0.01691172417169623, 0.0008413352946567976, -0.03359153630476425, -0.008275275585405306, 0.0013714405133193885, -0.005126708543271463, 0.01981681729153273, -0.02398736382205735, -0.0041451274061689025, -0.03225698764465823, -0.02831111379847795, 0.1365809018091304, -0.12897466821489295, 0.034244323760272755, -0.020504968710820692, -0.005767326226283842, 0.017498091379794745, 0.07056661525587726, -0.15808812917458637, -0.0032148713557651476, 0.1509112835386428, 0.1431455447416019, -0.021945707564240524, 0.03815409716384277, -0.009555035860292105, 0.0025323862241084516, -0.0619877324312127, -0.01288471935453972, -0.021877072169955658, 0.11094427282961794, -0.00022568642744102974, 0.10160442994183772, 0.09426944598571874, -0.11464876537605329, -0.001980418086396263, -0.025779306900936882, -0.06684791154877906, 0.02647864218871998, -0.08552802400449953, 0.010350617606127988, -0.006540369505939614, 0.030625517370290956, 0.01419234015197457, 0.016378136857716385, 0.005467952075993154, -0.08939438255066649, 0.03419094685569046, 0.08072921760477758, 0.03605748892762595, 0.07396576319263559, -0.12754970274354893, 0.08517121085516566, -0.0879414236730781, -0.018526968146884426, 0.014699911236986996, 0.0029363542801617015, -0.03475487584566883, 0.034592352290159445, -0.04472282929309501, -0.00996065731072382, -0.011109709565560122, 0.014627689998938043, 0.009424742494993033, 0.0393571178719232, -0.17423711394258598, 0.10609868082227236, -0.14839718728402396, 0.08089574410407985, 0.004280831027939858, 0.024892326512852447, -0.14069671161336414, 0.005757102604354358, -0.02688032250661787, -0.027187046455985645, 0.007784251976486911, 0.07834456656937094, -0.0073283703372989395, -0.08674475235347415, 0.0027881500591995234, -0.022547985394327175, 0.03401220921697434, 0.0266631315894863, -0.0011263413437027636, -0.05380255614412549, 0.005755857773659723, -0.01184717792858247, 0.0016947685955887026, 0.013709962146361665, 0.023204296002524827, 0.002196747186773873, 0.10518906600799349]
    corr2_errs = [0.012285527191333578, 0.058995608587165904, 0.0021521840910052603, -0.024445231647590104, -0.00401521695486351, -0.009344190255954843, 0.022282486437080307, -0.020864194085696453, -0.00789557965878429, 0.004048928567565835, 0.010818940032851998, -0.005375658426111429, 0.006953348452570118, -0.03027831003719994, 0.0116418136481381, -0.01468892658043175, -0.025787808734641038, -0.012699745998240752, 0.007559901702274607, -0.013276229658331584, -0.018845261052680362, -0.006617901706658907, 0.012014340655923883, 0.004146763116391061, -0.008038759511176163, 0.01706384678121242, 0.018924911455241222, -0.02217291016138584, -0.0012407286037487975, -0.0008772457242747222, 0.01516698085325996, 0.0005180995547295897, 0.017544180446512403, -0.007754590947708473, -0.005734962864285764, -0.018997517171562894, -0.004356823506996177, -0.01611498038107484, 0.005122204498172245, -0.011432537413931021, 0.04207951650539954, 0.01820983437249213, -0.025361395221395994, -0.010541717364600078, -0.004968969599645656, -0.023122820814605344, -0.023583079341689794, -0.0077815391429339464, 0.004474954153002475, -0.013955931112499487, 0.00203218072517819, -0.005215053061589092, 0.005859396542826079, 0.015405472697396227, -0.009994427871422074, -0.0047140288342195635, 0.07246186851638337, 0.02779013983923173, 0.0080774962154333, -0.026688043865139788, 0.03187603723685073, -0.01019493881151251, -0.0011237807729314042, 0.02207204838368826, -0.003811138713194495, -0.013045986259618836, 0.022817642982446484, 0.004313406916047091, 0.008110598676432712, -0.007224890579781373, -0.010825724164118089, 0.01856726146668386, 0.03108976878352898, 0.0010306867313709112, 0.04029058207865455, -0.002561149364750409, 0.07262839501568563, -0.009451319316459594, -0.012953760425129746, -0.02263896663109506, -0.004607981929499294, -0.004900865132220378, -0.01891969736759143, 0.009354213217515284, 0.025433532403648096, -0.0063930742788140815, -0.01691488055837087, -0.01889401678232261, 0.0020703430022244174, 0.011114157499421795, 0.016959434982704247, 0.010965700555281788, -0.0011687704528821063, -0.007455199052440964, -0.003579828840188254, 0.0046008670630242055, -0.004578416140086139, 0.012004125702166801, 0.006639403464577365, 0.287321271682617]
    ml_errs = [-0.00046665140253709936, 0.009474464047654285, 0.001201024195885636, -0.010188893219576556, -0.006293576703638365, 0.00528818813059434, -0.002156792868103219, -0.008029065500972099, -0.008315859612441745, 0.01784627704619024, 0.005192718653231898, 0.01199185765884564, 0.0019663950024604304, -0.0054209627781149905, 0.0016656947481628048, -0.020406945949929867, -0.01639897576990812, -0.010088480286173929, 0.007648381692518302, -0.009738136048462243, -0.01577058139171239, -0.008144181538362438, 0.009144270972394564, 0.008150482674917747, -0.006457179685570136, 0.013889627131220217, 0.0031201731979633163, 0.0047703528676938944, -0.0030556744036221994, -0.006693699082918725, 0.01706155864435277, -0.0058137497470840405, 0.0030777020416703493, 0.0038705737704335386, 0.0025323862241084516, -0.004199238803306804, 0.003990157572617337, -0.0030918318170826686, 0.0006207849945250032, -0.0008094085852986543, 0.0017879409481820208, -0.0002935435872182257, -0.0043252775409602096, 0.002689359176464845, -0.004393849663061666, -0.004973362815869714, -0.018786357870603787, -0.0020026897801433408, 0.00913010763969846, -0.009565111596656628, 0.0015455407788378395, -0.008466692703044565, 0.004862890652706731, 0.005202623822421426, -0.0006155489055915953, 0.006066151977095258, 0.0019267262939968166, 0.009789991824032285, 0.012197345761154832, -0.011972715487737137, 0.012471269376532979, 0.0008374099719967998, -0.0008030408082980722, 0.013161007366271349, -0.005448018532702603, -0.006630080967073737, 0.006255294808707124, 0.006326326694090889, 0.007657138726433921, -0.009783068297701925, 0.00396149420535763, 0.016535539690713308, -0.00267087749382644, -0.006125132479587014, 0.006282191828616668, -0.001299203503899865, -0.0031602466274196495, -0.008720253397070943, -0.004293781380029227, 0.006401272166759953, -0.0029456641127958427, -0.008307344756602164, -0.006173048773110774, 0.014364392665063841, 0.004795574679308912, -0.0064262542751553076, 0.0017156873873147571, -0.015254171183672893, 0.0066381224985546106, 0.0037647883098059776, 0.007453366030898634, 0.006408981057732133, 0.003136887072351102, 0.000237030099369262, -0.006593678507863752, 0.0028091472605895618, 0.010048432247072858, 0.012007443701801068, -8.507579394234632e-05, 0.01189965205220056]

    # fmt: on
    algo_errs = np.array(algo_errs)
    ls_errs = np.array(ls_errs)
    nn_errs = np.array(nn_errs)
    corr_errs = np.array(corr_errs)
    image_errs = np.array(image_errs)
    corr2_errs = np.array(corr2_errs)
    corr2_errs *= -1
    ml_errs = np.array(ml_errs)
    ml_errs *= -1

    # num_fails = 0
    # for ind in range(len(phi_errs)):
    #     row = phi_errs[ind]
    #     abs_errs = [abs(el) for el in row]
    #     best_ind = abs_errs.index(min(abs_errs))
    #     if best_ind != 0:
    #         print(f"{ind}: {best_ind}")
    #         num_fails += 1
    # print(num_fails)

    fig, ax = plt.subplots()
    colors = kpl.data_color_cycler
    kpl.plot_points(ax, true_phis, algo_errs, label="Algo", color=colors[0])
    # kpl.plot_points(ax, true_phis, ls_errs, label="LS", color=colors[1])
    # kpl.plot_points(ax, true_phis, nn_errs, label="NN", color=colors[2])
    # kpl.plot_points(ax, true_phis, corr_errs, label="Corr", color=colors[3])
    # kpl.plot_points(ax, true_phis, image_errs, label="Img", color=colors[4])
    # kpl.plot_points(ax, true_phis, corr2_errs, label="Corr", color=colors[5])
    kpl.plot_points(ax, true_phis, ml_errs, label="ML", color=colors[6])
    # kpl.plot_points(ax, true_phis, abs(ml_errs) - abs(algo_errs), label="ML_diff", color=colors[7])
    ax.set_xlabel("True phase")
    ax.set_ylabel("Error")
    ax.legend()
    kpl.tight_layout(fig)

    # inds = []
    # for ind in range(len(true_phis)):
    #     val = true_phis[ind]
    #     thresh = 0.5
    #     if (np.pi / 4) - thresh < val < (np.pi / 4) + thresh:
    #         inds.append(ind)
    # print(len(inds))
    inds = range(len(algo_errs))
    algo_rms = np.sqrt(np.mean(algo_errs[inds] ** 2))
    ls_rms = np.sqrt(np.mean(ls_errs[inds] ** 2))
    nn_rms = np.sqrt(np.mean(nn_errs[inds] ** 2))
    corr_rms = np.sqrt(np.mean(corr_errs[inds] ** 2))
    image_rms = np.sqrt(np.mean(image_errs[inds] ** 2))
    corr2_rms = np.sqrt(np.mean(corr2_errs[inds] ** 2))
    ml_rms = np.sqrt(np.mean(ml_errs[inds] ** 2))
    print(f"Alorithm rms error: {round(algo_rms, 6)}")
    print(f"Least squares rms error: {round(ls_rms, 6)}")
    print(f"Neural net rms error: {round(nn_rms, 6)}")
    print(f"Correlation rms error: {round(corr_rms, 6)}")
    print(f"Image rms error: {round(image_rms, 6)}")
    print(f"Correlation2 rms error: {round(corr2_rms, 6)}")
    print(f"ML rms error: {round(ml_rms, 6)}")

    plt.show(block=True)
