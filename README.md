# MSPCNN Model Repository

Welcome to the MSPCNN model repository. This README provides a brief overview of the structure and content of the repo, as well as some guidance on how to utilise the provided resources. In this repo, we set the LSTM as the default predictive model.

## Overview

- **MSPC-LSTM Class**: Our primary contribution is the encapsulation of an MSPC-LSTM class. This class can accept any structure of two AEs and an LSTM. It has been equipped with numerous functions designed to train both AEs and an LSTM without physical constraints.

- **Dataloader**: Due to the varying dataset formats required by different AEs and LSTMs, we've encapsulated functions returning diverse datasets within `models/dataloader.py`. Users simply need to provide raw data or a data path and select the model type for training. The appropriate dataloader will be returned automatically.

- **Physical Constraints**: If you wish to employ physical constraints within the model, initialize a child class of MSPC-LSTM. Within this subclass, you can add the necessary functions for calculating the physical constraints, computing the physical loss, and establishing a function for training an LSTM with these constraints. A hands-on example of this process can be found in `HowToTrainMSPC.ipynb` located in the root directory.

- **Subclasses for Specific Cases**: Within `models/model.py`, we've constructed two subclasses named `Burgers_MSPC_LSTM` and `SW_MSPC_LSTM`. These are tailored for the Burgers' equation and shallow water physics problems respectively, both employing physical constraints related to energy conservation and flow operators.

- **Evaluation and Plotting**: Functions for evaluating and visualizing the predictions against actual results have been encapsulated in `models/results_plot.py`. To understand how to employ a trained model for predictions and process these results, refer to the example notebooks `Burgers'_equation.ipynb` and `Shallow_Water.ipynb` located in the root directory.

## Getting Started

1. Begin by examining the core class and its functions within `models/model.py`.
2. Familiarize yourself with the dataloading functions in `models/dataloader.py`.
3. For hands-on examples and a deeper understanding, explore the provided Jupyter notebooks in the root directory.

## Model Weights and Pre-trained Models

The trained models and weights associated with this repository can be downloaded from [Model_Paths](https://1drv.ms/f/s!AsXCfra7D_MSzTbZvR8mIWTlK9Kt?e=bLgXgF). Make sure to place them in the appropriate directories as indicated by the respective scripts or notebooks.

## Contact

Hao Zhou - hao.zhou22@imperial.ac.uk
Sibo Cheng - sibo.cheng@imperial.ac.uk
