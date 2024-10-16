#!/usr/bin/env python3
"""
    This code generates a 2D matrix giving the MCC, Sparsity Rate (or other metric)
    of an aTTQ experiment studying the influence of t_min and t_max.

    Options:
    --------
    --exp_results_folder: str
        Path to the results folder of the experiment
"""
import os
import argparse
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
from src.utils.plot_results_classification import load_results_data, plotMeanLoss, plotMeanMetrics

# Global variables
DPI_VAL=50

def get_results_and_sparsity_rates_influence_tmin_tmax(exp_results_folder):
    """
        This function gets the results metrics of an experiment studying the
        influence of tmin and tmax.

        Arguments:
        ----------
        exp_results_folder: str
            Path to the results folder of the experiment

        Returns:
        --------
        metrics_dict: dict
            Dictionary containig the metric that is used to evaluate the performances
            of the models.
        sparsity_rate_dict: dict
            Dict containing the sparsity rates of the models
    """
    # Checking if the sumarized results already exist
    if os.path.isfile(exp_results_folder + '/sumarized_results.pth'):
        # Loading the data into a dict
        with open(exp_results_folder + '/sumarized_results.pth', "rb") as fp:   # Unpickling
            summary_dict = pickle.load(fp)
        metrics_dict = summary_dict['metrics_dict']
        sparsity_rate_dict = summary_dict['sparsity_rate_dict']
    else:
        t_min_vals = []
        t_max_vals = []
        metrics_dict = {}
        sparsity_rate_dict = {}
        for folder in os.listdir(exp_results_folder):
            if ('params_exp' not in folder):
                print("\n\n\n===================== Plotting the results of {} =====================".format(folder))
                # Loading the results data
                results_dict = load_results_data(results_folder=exp_results_folder + '/' + folder + '/metrics/')

                # Getting the values of t_min and t_max
                t_min = float(folder.split('_')[1].split('TMIN-')[-1])
                t_max = float(folder.split('_')[2].split('TMAX-')[-1])
                if (t_min not in t_min_vals):
                    t_min_vals.append(t_min)
                if (t_max not in t_max_vals):
                    t_max_vals.append(t_max)

                # Plotting MCC
                print("\t\t=======> MCC <=======")
                max_mcc, last_mcc, sparsity_rates = plotMeanMetrics(results_dict, last_epochs_use=1, metric_type='mcc', selected_epoch=0, plot_curves=False, plot_val_metrics=False)

                # Adding the metric values to the metrics dict
                metrics_dict[(t_min, t_max)] = last_mcc

                # Plotting the sparsity rate
                # Sparsity rate
                if ('SparsityRate' in results_dict[0]):
                    sparsity_rates = [results_dict[i]['SparsityRate'] for i in results_dict]
                    print("\n\n==========> Sparsity rate of: {} +- {}".format(np.mean(sparsity_rates)*100, np.std(sparsity_rates)*100))

                # Adding the sparsity rate to the sparsity rate dict
                sparsity_rate_dict[(t_min, t_max)] = np.mean(sparsity_rates)*100

        # Saving the results
        summary_dict = {'metrics_dict': metrics_dict, 'sparsity_rate_dict': sparsity_rate_dict}
        with open(exp_results_folder + '/sumarized_results.pth', "wb") as fp:   #Pickling
            pickle.dump(summary_dict, fp)

    print("\n\n==========================================================================================")
    for key_val in metrics_dict:
        print("\nFor (t_min, t_max) = ({}, {}), we have: ".format(key_val[0], key_val[1]))
        print("\tMetric: {}".format(metrics_dict[key_val]))
        print("\tSparsity rate: {}".format(sparsity_rate_dict[key_val]))

    return metrics_dict, sparsity_rate_dict

def plotMetricsAndSparsityMatrices(exp_results_folder, metrics_dict, sparsity_rate_dict):
    """
        This function puts the metrics_dict and sparsity_rate_dict into a numpy
        array format to plot it in a 2D image.

        Arguments:
        ----------
        exp_results_folder: str
            Path to the results folder of the experiment
        metrics_dict: dict
            Dictionary containig the metric that is used to evaluate the performances
            of the models.
        sparsity_rate_dict: dict
            Dict containing the sparsity rates of the models
    """
    # Constructing the matrices to plot
    t_vals_min = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    t_vals_max = [2, 1.5, 1, 0.5, 0, -0.5, -1, -1.5, -2]
    metric_matrix = np.zeros((len(t_vals_max), len(t_vals_min)))
    sparsity_rate_matrix = np.zeros((len(t_vals_max), len(t_vals_min)))
    for t_val_max_idx in range(len(t_vals_max)):
        for t_val_min_idx in range(len(t_vals_min)):
            # WARNING  VERY IMPORTANT: THE KEYS IN metrics_dict ARE UNDER THE FORM
            # (t_min, t_max) AND NOT under the form (t_max, t_min) !
            if ((t_vals_min[t_val_min_idx], t_vals_max[t_val_max_idx]) not in metrics_dict):
                metric_matrix[t_val_max_idx, t_val_min_idx] = -10
                sparsity_rate_matrix[t_val_max_idx, t_val_min_idx] = 0
            else:
                metric_matrix[t_val_max_idx, t_val_min_idx] = metrics_dict[(t_vals_min[t_val_min_idx], t_vals_max[t_val_max_idx])]
                sparsity_rate_matrix[t_val_max_idx, t_val_min_idx] = sparsity_rate_dict[(t_vals_min[t_val_min_idx], t_vals_max[t_val_max_idx])]

    # Replacing the -10 values of the metric matrix by the max real value obtained during the experiment
    max_val_metric = metric_matrix.max()
    max_sparsity_rate = sparsity_rate_matrix.max()
    for i in range(metric_matrix.shape[0]):
        for j in range(metric_matrix.shape[1]):
            if (metric_matrix[i,j] == -10):
                # metric_matrix[i,j] = min_val_metric
                # sparsity_rate_matrix[i,j] = min_sparsity_rate
                metric_matrix[i,j] = max_val_metric
                sparsity_rate_matrix[i,j] = max_sparsity_rate

    # Matplotlib params
    mpl.rcParams['figure.figsize'] = (20, 10) # To increase the size of the plots
    # Defining the plot
    params = {'axes.labelsize': 35,
              'axes.titlesize': 35}
    plt.rcParams.update(params)
    axisTickSize = 30
    legendFontSize = 25

    # Plotting a MCC matrix
    fig, ax = plt.subplots()
    psm = ax.pcolormesh(metric_matrix, cmap='gist_gray')
    cbar = fig.colorbar(psm, ax=ax)
    plt.xlabel(r'$t_{min}$')
    plt.ylabel(r'$t_{max}$')
    plt.xticks([i for i in range(len(t_vals_min))], t_vals_min, fontsize=axisTickSize)
    plt.yticks([i for i in range(len(t_vals_max))], t_vals_max, fontsize=axisTickSize)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=legendFontSize)
    cbar.ax.tick_params(labelsize=legendFontSize)
    # plt.show()
    plt.savefig(exp_results_folder+'/metric_matrix.png', dpi=DPI_VAL)

    # Plotting a sparsity rate matrix
    fig, ax = plt.subplots()
    psm = ax.pcolormesh(sparsity_rate_matrix, cmap='gist_gray')
    cbar = fig.colorbar(psm, ax=ax)
    plt.xlabel(r'$t_{min}$')
    plt.ylabel(r'$t_{max}$')
    plt.xticks([i for i in range(len(t_vals_min))], t_vals_min, fontsize=axisTickSize)
    plt.yticks([i for i in range(len(t_vals_max))], t_vals_max, fontsize=axisTickSize)
    plt.gca().invert_yaxis()
    plt.legend(fontsize=legendFontSize)
    cbar.ax.tick_params(labelsize=legendFontSize)
    # plt.show()
    plt.savefig(exp_results_folder+'/sparsity_rate_matrix.png', dpi=DPI_VAL)

def main():
    #==========================================================================#
    # Construct the argument parser
    ap = argparse.ArgumentParser()
    # Add the arguments to the parser
    ap.add_argument('--exp_results_folder', required=True, help="Path to the results folder of the experiment", type=str)
    args = vars(ap.parse_args())

    # Getting the value of the arguments
    exp_results_folder = args['exp_results_folder']

    #==========================================================================#
    # Getting the results and the sparsity rates
    metrics_dict, sparsity_rate_dict = get_results_and_sparsity_rates_influence_tmin_tmax(exp_results_folder)

    # Plotting the matrices
    plotMetricsAndSparsityMatrices(exp_results_folder, metrics_dict, sparsity_rate_dict)

if __name__=="__main__":
    main()
