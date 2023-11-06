import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import SSIM

def calculate_ssim(outputs, targets, ssim):
    """
    Calculates the Structural Similarity Index (SSIM) between predicted outputs and targets for each sequence step.

    Args:
        outputs (torch.Tensor): The tensor of predicted values.
        targets (torch.Tensor): The tensor of ground truth values.

    Returns:
        list: A list containing SSIM values for each sequence step.
    """
    num_steps = outputs.shape[1]
    # calculate ssim
    ssims = []
    for j in range(num_steps):
        ssim.update((outputs[:, j, :], targets[:, j, :]))
        ssim_value = ssim.compute()
        ssims.append(ssim_value)

    return ssims

def get_seperate_loss_std(outputs, targets):
    """
    Calculates the mean squared error and its standard deviation for each sequence step.

    Args:
        outputs (torch.Tensor): The tensor of predicted values.
        targets (torch.Tensor): The tensor of ground truth values.

    Returns:
        tuple: Two lists containing mean squared errors and standard deviations for each sequence step, respectively.
    """

    losses = []
    stds = []
    criterion = nn.MSELoss(reduction='none')
    num_steps = outputs.shape[1]
    for i in range(num_steps):
        # For each sequence_length, get the loss for each item in the batch
        individual_losses = criterion(outputs[:,i,:], targets[:,i,:]).mean(dim=-1)
        mean_loss = torch.mean(individual_losses).item()
        std = torch.std(individual_losses).item()

        losses.append(mean_loss)
        stds.append(std)

    return losses, stds

def plot_errors_with_std(x, y_values, std_values, labels, title='Error with Standard Deviation', xlabel='Steps', ylabel='Error Value'):
    """
    Plots multiple error lines with their respective standard deviations.

    Args:
        x (list or array): The x-axis values.
        y_values (list of lists or arrays): Multiple sets of y-values to be plotted.
        std_values (list of lists or arrays): Standard deviation values corresponding to y_values.
        labels (list of str): Labels for each line plotted.
        title (str, optional): Title of the plot. Defaults to 'Error with Standard Deviation'.
        xlabel (str, optional): X-axis label. Defaults to 'Steps'.
        ylabel (str, optional): Y-axis label. Defaults to 'Error Value'.
    """

    colors = ['red', 'blue','green']
    fill_colors = ['#FFC8DD', '#90DBF4','#B9FBC0']  # You can add more colors if needed
    plt.figure(figsize=(10, 5))

    for idx, (y, std, label) in enumerate(zip(y_values, std_values, labels)):
        plt.plot(x, y, label=f'{label} Error', color=colors[idx])
        plt.fill_between(x, np.maximum(0,y - std), y + std, color=fill_colors[idx], alpha=0.8, label=f'{label} 1 std deviation')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def calculate_channel_error(original_data, predicted_data, channel, batch_num, index):
    """
    Computes the channel-wise error between the original and predicted data.

    Args:
        original_data (torch.Tensor): Tensor containing the ground truth values.
        predicted_data (torch.Tensor): Tensor containing the predicted values.
        channel (int): Channel index to extract and compute error for.
        batch_num (int): Batch index to extract data from.
        index (int): Sequence step index.

    Returns:
        np.ndarray: 2D array representing the channel-wise error.
    """

    # Extract the specified channel from the original and predicted data
    original_channel = original_data[batch_num, index, channel, :, :].cpu().detach().numpy()
    predicted_channel = predicted_data[batch_num, index, channel, :, :].cpu().detach().numpy()

    # Calculate the channel-wise error
    channel_error = np.abs(original_channel - predicted_channel)

    return channel_error

def plot_images_and_loss(original_data, predicted_data, channel, batch_num, indices, nDisplay=5, initial_gap=0, name=None):
    """
    Visualizes original data, predicted data, and their L1-norm error for a specific channel.

    Args:
        original_data (torch.Tensor): Tensor containing the ground truth values.
        predicted_data (torch.Tensor): Tensor containing the predicted values.
        channel (int): Channel index to extract and visualize.
        batch_num (int): Batch index to extract data from.
        indices (list of int): Sequence step indices for which to visualize data.
        nDisplay (int, optional): Number of sequence steps to display. Defaults to 5.
        initial_gap (int, optional): Initial gap in the sequence steps. Defaults to 0.
        name (str, optional): Name to be displayed in the title. Defaults to None.

    Returns:
        list: A list containing histogram data for the L1-norm errors.
    """

    figure = plt.figure(figsize=(30,20))
    original_data = original_data[:,initial_gap:,:]
    histogram_data = []  # This list will store the histogram data

    for i in range(nDisplay):
        batch_idx = batch_num
        idx = indices[i]

        # Original data
        ax = plt.subplot(4, nDisplay, i+1)  # Note the change in the subplot layout
        ax.set_title(f't={idx+1}')
        # if i == 0:  # If it's the first column
        ax.set_ylabel('Ground True', rotation=90, size=15)

        data_display = original_data.cpu().detach().numpy()[batch_idx][idx][channel]
        vmin = data_display.min()
        vmax = data_display.max()

        plt.imshow(data_display, vmin=vmin, vmax=vmax)
        plt.colorbar()


        # Reconstruction
        ax = plt.subplot(4, nDisplay, i+1+nDisplay)
        # if i == 0:  # If it's the first column
        ax.set_ylabel('Prediction', rotation=90, size=15)
        plt.imshow(predicted_data.cpu().detach().numpy()[batch_idx][idx][channel], vmin=vmin, vmax=vmax)
        plt.colorbar()


        # Channel error with unified colorbar
        channel_error = calculate_channel_error(original_data, predicted_data, channel, batch_num, idx)
        ax = plt.subplot(4, nDisplay, i+1+2*nDisplay)  # Note the change in the subplot layout
        # if i == 0:  # If it's the first column
        ax.set_ylabel('L1-norm', rotation=90, size=15)
        plt.imshow(channel_error, cmap='coolwarm', vmin = 0, vmax=vmax/10+0.1)
        plt.colorbar()

        histogram_data.append(channel_error.flatten())  # Aprpend the flattened data to the list

    if channel == 0:
        channel_name = 'u'
    elif channel == 1:
        channel_name = 'v'
    elif channel == 2:
        channel_name = 'h'

    # figure.suptitle(f'Results of {name} in Channel {channel_name}', fontsize=20)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)  # Adjust the space between subplots
    plt.show()

    return histogram_data

def plot_histograms(*data_sets, labels=None, bins=100, alpha=0.3):
    """
    Displays side-by-side histograms and cumulative histograms for provided data sets.

    Args:
        data_sets (tuple of np.ndarray): Data sets to be plotted. Supports 2 or 3 data sets.
        labels (list of str, optional): Labels corresponding to each data set. Defaults to generic names.
        bins (int, optional): Number of bins for histograms. Defaults to 100.
        alpha (float, optional): Transparency for histogram bars. Defaults to 0.3.
    """

    if len(data_sets) not in [2, 3]:
        raise ValueError("Only 2 or 3 data sets are supported.")

    if labels is None:
        labels = ['Data ' + str(i) for i in range(len(data_sets))]

    # Determine global min and max across all datasets
    global_min = min([data.min() for data in data_sets])
    global_max = max([data.max() for data in data_sets])

    # Set bin edges based on global min and max
    bin_edges = np.linspace(global_min, global_max, bins+1)

    # Plotting side by side histograms
    plt.figure(figsize=(12, 5))
    for data, label in zip(data_sets, labels):
        plt.hist(data, bins=bin_edges, alpha=alpha, label=label, log=True)
    plt.xlabel('Error Magnitude')  # X-axis representing the distribution of error sizes
    plt.ylabel('Frequency (Log Scale)')        # Y-axis representing the number of occurrences in each bin
    plt.legend()
    plt.show()

    # Plotting cumulative histograms
    plt.figure(figsize=(12, 5))
    for data, label in zip(data_sets, labels):
        plt.hist(data, bins=bin_edges, alpha=1, label=label, cumulative=True, histtype='step', log=True)
    plt.xlabel('Error Magnitude')   # X-axis representing the distribution of error sizes
    plt.ylabel('Cumulative Count (Log Scale)')  # Y-axis representing cumulative count of data points in log scale
    plt.legend()
    plt.show()
