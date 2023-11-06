import numpy as np
from pylab import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import SSIM
from . import dataloader

def load_loss_function(criterion_type):
    """
    Load a PyTorch loss function based on a given string.

    Args:
        criterion_type (str): A string indicating the desired loss function 
                              ('mse', 'l1', ...).

    Returns:
        nn.Module: The corresponding loss function.

    Raises:
        ValueError: If the input string does not correspond to a known loss function.
    """
    if criterion_type == 'mse':
        return nn.MSELoss()
    elif criterion_type == 'l1':
        return nn.L1Loss()
    # elif criterion_type == 'your_criterion':
    #     return YourCriterionFunction()
    else:
        raise ValueError(f"Unknown criterion type: {criterion_type}")

class MSPC_LSTM():
    """
    This class encapsulates the process of training the High-Frequency Convolutional Autoencoder (HF_CAE),
    Low-Frequency Convolutional Autoencoder (LF_CAE) and the LSTM networks sequentially. The HF_CAE and LF_CAE is responsible
    for compressing the physcial data into shared latent space and decompressing latent representation to different fidelity, 
    while the LSTM handle sequence-to-sequence forecasting in the compressed space.
    """
    def __init__(self, HF_CAE, LF_CAE, LSTM, device=None):
        """
        Initialize the MSPC_LSTM model.
        
        Args:
            HF_CAE (nn.Module): High frequency Convolutional AutoEncoder.
            LF_CAE (nn.Module): Low frequency Convolutional AutoEncoder.
            LF_encoder (nn.Module): Encoder part of LF_CAE.
            LF_decoder (nn.Module): Decoder part of LF_CAE.
            LSTM(nn.Module): LSTM network for sequence prediction in the compressed space.
            device (str or torch.device, optional): Device to move the models to (e.g., 'cuda' for GPU).
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.HF_CAE = HF_CAE.to(self.device)
        self.LF_CAE = LF_CAE.to(self.device)
        self.LSTM = LSTM.to(self.device)
        self.HF_encoder = self.HF_CAE.encoder
        self.HF_decoder = self.HF_CAE.decoder
        self.LF_encoder = self.LF_CAE.encoder
        self.LF_decoder = self.LF_CAE.decoder
    
    def train_HFCAE(self, train_loader, test_data, num_epochs=30, lr=0.0001, criterion_type='mse'):
        """
        Train the High-Frequency CAE using the provided training data.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            test_data (torch.Tensor): Test data to validate the training.
            num_epochs (int, optional): Number of epochs for training.
            lr (float, optional): Learning rate for the optimizer.
            criterion_type (str, optional): Type of loss function (e.g., 'mse' for Mean Squared Error).
        """
        criterion = load_loss_function(criterion_type)
        optimizer = torch.optim.Adam(self.HF_CAE.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        # transfer the test data to torch and move the test tensor to the same device as the model
        test_tensor = torch.from_numpy(test_data).float().to(self.device)

        total_loss = []
        total_test_loss = []

        for epoch in range(num_epochs):
            epoch_loss = 0
            num_image = 0
            for images in train_loader:
                # Move images to the device and change the data type
                images = images.float().to(self.device)
                # clear the gradients
                optimizer.zero_grad()

                # forward pass
                outputs = self.HF_CAE(images)

                # calculate loss
                loss = criterion(outputs, images)

                # backward pass
                loss.backward()

                # perform a single optimization step
                optimizer.step()

                # accumulate loss for each epoch
                epoch_loss += loss.item()
                num_image += 1

                # Call the scheduler step to update the learning rate
            scheduler.step()

            test_results = self.HF_CAE(test_tensor)
            total_test_loss.append(F.mse_loss(test_tensor, test_results).detach())
            
            epoch_loss = epoch_loss / num_image
            total_loss.append(epoch_loss)
            
            # Print loss for this epoch
            print('Epoch [{}/{}], Loss: {:.6f}, Test Loss: {:.6f}'.format(epoch+1, num_epochs, loss.item(), total_test_loss[-1]))

    def train_LFCAE(self, train_loader, test_LF_tensor, test_HF_tensor, num_epochs=30, lr=0.0001, criterion_type='mse'):
        """
        Train both the LF encoder and decoder using the provided training data. The encoder learns to match the 
        outputs of the HF encoder, and the decoder learns to reconstruct the original LF data from these representations.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training pairs of LF and HF data.
            test_LF_tensor (torch.Tensor): Test data for LF.
            test_HF_tensor (torch.Tensor): Test data for HF.
            num_epochs (int, optional): Number of epochs for training.
            lr (float, optional): Learning rate for the optimizer.
            criterion_type (str, optional): Type of loss function (e.g., 'mse' for Mean Squared Error).
        """

        # Load loss function
        criterion = load_loss_function(criterion_type)

        # Optimizers for both LF_encoder and LF_decoder
        optimizer_encoder = torch.optim.Adam(self.LF_encoder.parameters(), lr=lr)
        optimizer_decoder = torch.optim.Adam(self.LF_decoder.parameters(), lr=lr)
        scheduler_encoder = torch.optim.lr_scheduler.StepLR(optimizer_encoder, step_size=1, gamma=0.8)
        scheduler_decoder = torch.optim.lr_scheduler.StepLR(optimizer_decoder, step_size=1, gamma=0.8)

        # Move test tensors to device
        test_LF_tensor = test_LF_tensor.to(self.device)
        test_HF_tensor = test_HF_tensor.to(self.device)

        # Store loss values
        total_loss_encoder = []
        total_loss_decoder = []
        total_test_loss_encoder = []
        total_test_loss_decoder = []

        for epoch in range(num_epochs):

            # Initialize epoch losses
            epoch_loss_encoder = 0
            epoch_loss_decoder = 0
            num_iteration = 0

            for LF_data, HF_data in train_loader:
                # Move data to device
                LF_data = LF_data.to(self.device)
                HF_data = HF_data.to(self.device)

                # Train the encoder
                optimizer_encoder.zero_grad()
                targets = self.HF_encoder(HF_data)
                outputs = self.LF_encoder(LF_data)
                encoder_loss = criterion(outputs, targets)
                encoder_loss.backward()
                optimizer_encoder.step()
                epoch_loss_encoder += encoder_loss.item()

                # Train the decoder using the latent representation from the encoder
                optimizer_decoder.zero_grad()
                latent_representation = targets
                reconstructed_data = self.LF_decoder(latent_representation)
                decoder_loss = criterion(reconstructed_data, LF_data)
                decoder_loss.backward()
                optimizer_decoder.step()
                epoch_loss_decoder += decoder_loss.item()

                num_iteration += 1

            # Scheduler steps
            scheduler_encoder.step()
            scheduler_decoder.step()

            # Append average loss for the epoch
            total_loss_encoder.append(epoch_loss_encoder/num_iteration)
            total_loss_decoder.append(epoch_loss_decoder/num_iteration)

            # Testing step for encoder
            test_encoded_outputs = self.LF_encoder(test_LF_tensor)
            test_encoded_targets = self.HF_encoder(test_HF_tensor)
            total_test_loss_encoder.append(criterion(test_encoded_outputs, test_encoded_targets).item())

            # Testing step for decoder
            test_decoded_outputs = self.LF_decoder(test_encoded_outputs)
            total_test_loss_decoder.append(criterion(test_decoded_outputs, test_LF_tensor).item())

            # Print epoch stats
            print(f'Epoch [{epoch+1}/{num_epochs}] - Encoder Loss: {total_loss_encoder[-1]:.6f} - Decoder Loss: {total_loss_decoder[-1]:.6f} - Test Encoder Loss: {total_test_loss_encoder[-1]:.6f} - Test Decoder Loss: {total_test_loss_decoder[-1]:.6f}')

    def train_LSTM(self, train_loader, test_data, num_epochs=30, lr=0.0001, criterion_type='mse'):
        """
        Train the LSTM network using the provided training data. The data is first compressed using HF encoder 
        and then fed into the LSTM to learn sequence-to-sequence predictions.
        
        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader with training data.
            test_data (torch.Tensor): Test data to validate the training.
            batch_size (int, optional): Size of training batches.
            num_epochs (int, optional): Number of epochs for training.
            lr (float, optional): Learning rate for the optimizer.
            criterion_type (str, optional): Type of loss function (e.g., 'mse' for Mean Squared Error).
        """
        
        # Load loss function
        criterion = load_loss_function(criterion_type)

        # Optimizers for the LSTM
        optimizer = torch.optim.Adam(self.LSTM.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        # Set test dataset
        b, s, c, h, w = test_data.shape
        test_compr = self.HF_encoder(test_data.view(-1, c, h, w))
        test_dataset = dataloader.load_data(test_compr.reshape(b,s,-1), model='seq2seq', lookback=3, lookahead=3)

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            train_epoch_loss, test_epoch_loss = 0., 0.

            # Training
            self.LSTM.train()
            for i, train_batch in enumerate(train_loader):
                b, s, c, h, w = train_batch.shape
                train_compr = self.HF_encoder(train_batch.view(-1, c, h, w))
                train_compr_dataset = dataloader.load_data(train_compr.reshape(b, s, -1).cpu().detach().numpy(), model='Seq2Seq', lookback=3, lookahead=3)
                for j in range(train_compr_dataset.__len__()):
                    inputs, targets = train_compr_dataset[j]
                    inputs = torch.tensor(inputs).to(self.device)
                    targets = torch.tensor(targets).to(self.device)

                    outputs = self.LSTM(inputs)

                    mse_loss = criterion(outputs, targets)
                    total_loss = mse_loss

                    train_epoch_loss += total_loss.item()/len(train_loader)/len(train_compr_dataset)

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            # Testing
            self.LSTM.eval()
            with torch.no_grad():
                for j in range(test_dataset.__len__()):
                    test_inputs, test_targets = test_dataset[j]
                    test_inputs = test_inputs.to(self.device)
                    test_targets = test_targets.to(self.device)

                    test_outputs = self.LSTM(test_inputs)

                    test_mse_loss = criterion(test_outputs, test_targets)
                    test_total_loss = test_mse_loss

                    test_epoch_loss += test_total_loss.item()/len(test_data)

            # Print the averaged loss per epoch
            print ('Epoch [{}/{}], MSE_Loss: {:.6f}, Total_Loss: {:.6f}, Test_MSE_Loss: {:.6f}, Test_Total_Loss: {:.6f}'
                .format(epoch+1, num_epochs, mse_loss.item(), train_epoch_loss, test_mse_loss.item(), test_epoch_loss))

            train_losses.append(train_epoch_loss)
            test_losses.append(test_epoch_loss)


        """
        Calculates the loss for each timestep between the outputs and targets.
        
        Args:
            outputs (torch.Tensor): The output tensor from the LSTM.
            targets (torch.Tensor): The ground truth or target tensor.
            criterion (callable): The loss function to compute the error.

        Returns:
            list: A list of loss values for each timestep.
        """
        num_steps = outputs.shape[1]
        losses = []
        for i in range(num_steps):
            loss = criterion(outputs[:, i, :], targets[:, i, :])
            losses.append(loss.item())
        return losses

    def predict(self, input_data, lookback, n):
        """
        Predict the future timesteps based on given input_data.

        Args:
            input_data (torch.Tensor): Input data tensor of shape (batch, seq, channel, width, height).
            lookback (int): Number of timesteps to be used for prediction.
            n (int): Number of times the prediction should be looped.

        Returns:
            torch.Tensor: Predicted data tensor of shape (batch, seq, channel, width, height).
        """

        # Check the input sequence length
        if input_data.size(1) < lookback:
            raise ValueError(f"Input sequence length {input_data.size(1)} is less than the required lookback {lookback}.")

        # Move data to the device
        input_data = input_data.to(self.device)

        # Compress the data with HF_encoder
        compressed_data = self.HF_encoder(input_data.reshape(-1, *input_data.shape[-3:]))
        compressed_data = compressed_data.reshape(input_data.shape[0], input_data.shape[1], -1)  # Reshape to (batch, seq, latent)

        # Initial sequence for predictions starting with compressed input data
        predictions = compressed_data.clone()

        for _ in range(n):
            # Get the last 'lookback' timesteps
            inputs = predictions[:, -lookback:, :]

            # Predict 'lookahead' timesteps
            outputs = self.LSTM(inputs)

            # Append the outputs to predictions
            predictions = torch.cat([predictions, outputs], dim=1)

        # Decode and reshape
        decoded_predictions = self.HF_decoder(predictions.reshape(-1, predictions.shape[-1]))
        reshaped_predictions = decoded_predictions.reshape(input_data.shape[0], predictions.shape[1], *input_data.shape[-3:])

        return reshaped_predictions

class Burgers_MSPC_LSTM(MSPC_LSTM):

    def __init__(self, HF_CAE, LF_CAE, LSTM, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.HF_CAE = HF_CAE.to(self.device)
        self.LF_CAE = LF_CAE.to(self.device)
        self.LSTM = LSTM.to(self.device)
        self.HF_encoder = self.HF_CAE.encoder
        self.HF_decoder = self.HF_CAE.decoder
        self.LF_encoder = self.LF_CAE[0]
        self.LF_decoder = self.LF_CAE[1]

    def train_LSTM_with_PC(self, train_loader, test_data, energy_coef=None, fo_coef=None, decoder_type=None, batch_size=20, num_epochs=30, lr=0.0001, criterion_type='mse'):
        # Coerce coefficients to lists if they are not
        if isinstance(energy_coef, (int, float)):
            energy_coef = [energy_coef] * num_epochs
        if isinstance(pde_coef, (int, float)):
            pde_coef = [pde_coef] * num_epochs
        # Load loss function
        criterion = load_loss_function(criterion_type)

        # Optimizers for the LSTM
        optimizer = torch.optim.Adam(self.LSTM.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        # Set test dataset
        b, s, c, w, h = test_data.shape
        test_compr = self.HF_encoder(test_data.view(-1, c, w, h))
        test_dataset = dataloader.load_data(test_compr.reshape(b,s,-1), model='Seq2Seq', lookback=3, lookahead=3)

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            train_epoch_loss, test_epoch_loss = 0., 0.

            # Training
            self.LSTM.train()
            for i, train_batch in enumerate(train_loader):
                b, s, c, h, w = train_batch.shape
                train_compr = self.HF_encoder(train_batch.view(-1, c, w, h))
                train_compr_dataset = dataloader.load_data(train_compr.reshape(b, s, -1).cpu().detach().numpy(), model='seq2seq', lookback=3, lookahead=3)
                for j in range(train_compr_dataset.__len__()):
                    inputs, targets = train_compr_dataset[j]
                    inputs = torch.tensor(inputs).to(self.device)
                    targets = torch.tensor(targets).to(self.device)

                    outputs = self.LSTM(inputs)

                    mse_loss = criterion(outputs, targets)
                    total_loss = mse_loss
                    if energy_coef is not None:
                        energy_loss = torch.abs(torch.mean((self.calculate_total_energy(inputs, decoder_type) -  #, dx=0.02
                                                    torch.mean(self.calculate_total_energy(outputs, decoder_type), dim=1, keepdim=True)))) # , dx=0.02
                        total_loss += energy_coef[epoch]*energy_loss
                    if fo_coef is not None:
                        fo_loss = self.compute_evolve_loss(inputs, outputs, decoder_type)
                        total_loss += fo_coef[epoch]*fo_loss

                    train_epoch_loss += total_loss.item()/len(train_loader)/len(train_compr_dataset)

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            # Testing
            self.LSTM.eval()
            with torch.no_grad():
                for j in range(test_dataset.__len__()):
                    test_inputs, test_targets = test_dataset[j]
                    test_inputs = test_inputs.to(self.device)
                    test_targets = test_targets.to(self.device)

                    test_outputs = self.LSTM(test_inputs)

                    test_mse_loss = criterion(test_outputs, test_targets)
                    test_total_loss = test_mse_loss

                    if energy_coef is not None:
                        test_energy_loss = torch.abs(torch.mean((self.calculate_total_energy(test_inputs, decoder_type) -  #, dx=0.02
                                                    torch.mean(self.calculate_total_energy(test_outputs, decoder_type), dim=1, keepdim=True)))) # , dx=0.02
                        test_total_loss += energy_coef[epoch]*test_energy_loss
                    if fo_coef is not None:
                        test_fo_loss = self.compute_evolve_loss(test_inputs, test_outputs, decoder_type)
                        test_total_loss += fo_coef[epoch]*test_fo_loss

                    test_epoch_loss += test_total_loss.item()/len(test_data)

            # Print the averaged loss per epoch
            print ('Epoch [{}/{}], MSE_Loss: {:.6f}, Total_Loss: {:.6f}, Test_MSE_Loss: {:.6f}, Test_Total_Loss: {:.6f}'
                .format(epoch+1, num_epochs, mse_loss.item(), train_epoch_loss, test_mse_loss.item(), test_epoch_loss))

            train_losses.append(train_epoch_loss)
            test_losses.append(test_epoch_loss)

    def calculate_total_energy(self, data_compr, decoder_type):
        
        # Set decoder for physcial constraints
        if decoder_type == 'high':
            decoder = self.HF_decoder
        else:
            decoder = self.LF_decoder

        N, Nt, latent = data_compr.shape
        energies = torch.zeros(N, Nt)
        # reconstruct the data
        data = decoder(data_compr.reshape(N*Nt, latent))
        data = data.reshape(N, Nt, *data.shape[-3:])
        for i in range(N):
            for j in range(Nt):
                # Calculate total energy
                energies[i, j] = 0.5*torch.sum(data[i][j][0]**2+data[i][j][1]**2)

        return energies

    def compute_evolve_loss(self, input_data_compr, output_data_compr, decoder_type):
        N = len(output_data_compr)
        Nt = len(output_data_compr[0])
        
        # Set decoder for physical constraints
        if decoder_type == 'high':
            decoder = self.HF_decoder
            grid = 129
        else:
            decoder = self.LF_decoder
            grid = 33

        input_images = decoder(input_data_compr.reshape(N*Nt, -1)).reshape(N, Nt, 2, grid, grid)
        output_images = decoder(output_data_compr.reshape(N*Nt, -1)).reshape(N, Nt, 2, grid, grid)

        u = input_images[:, -1, 0, :].cpu().detach().numpy()
        v = input_images[:, -1, 1, :].cpu().detach().numpy()
        input_images_evolved = torch.empty((N, Nt, 2, grid, grid))
        for i in range(Nt):
            u, v = self.evolve(u, v, grid)
            input_image_evolved = torch.stack((torch.tensor(u), torch.tensor(v)), dim=1)
            # Assign input_image_evolved to the corresponding slice in input_images_evolved tensor
            input_images_evolved[:, i, :, :, :] = input_image_evolved

        input_images_evolved = input_images_evolved.to(self.device)

        # calculate the asymmetry
        FO_loss = nn.MSELoss()(input_images_evolved, output_images)

        return FO_loss
    
    def get_params(self, nx, ny, cfl=0.009):
        dx = 2 / (nx - 1)
        dy = 2 / (ny - 1)

        # Viscosity
        nu = 0.01

        # Calculate time step size
        dt = cfl * dx * dy / nu

        return dx, dy, nu, dt

    def evolve(self, u, v, grid):
        """
        Evolve state (h, u, v) forward in time using simple Euler method
        x_{N+1} = x_{N} +   dx/dt * d_t
        """

        dx, dy, nu, dt = self.get_params(grid, grid)

        un = u.copy()
        vn = v.copy()

        # Backward Difference Scheme for Convection Term
        # Central Difference Scheme for Diffusion Term
        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt/dx * un[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
                        dt/dy * vn[1:-1, 1:-1] * (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                        nu * dt/dx**2 * (un[1:-1, 2:] - 2*un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt/dy**2 * (un[2:, 1:-1] - 2*un[1:-1, 1:-1] + un[0:-2, 1:-1]))

        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt/dx * un[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        dt/dy * vn[1:-1, 1:-1] * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                        nu * dt/dx**2 * (vn[1:-1, 2:] - 2*vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        nu * dt/dy**2 * (vn[2:, 1:-1] - 2*vn[1:-1, 1:-1] + vn[0:-2, 1:-1]))

        # Apply boundary conditions
        u[0, :] = 1
        u[-1, :] = 1
        u[:, 0] = 1
        u[:, -1] = 1

        v[0, :] = 1
        v[-1, :] = 1
        v[:, 0] = 1
        v[:, -1] = 1

        return u, v

class SW_MSPC_LSTM(MSPC_LSTM):
    
    def __init__(self, HF_CAE, LF_CAE, LSTM, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.HF_CAE = HF_CAE.to(self.device)
        self.LF_CAE = LF_CAE.to(self.device)
        self.LSTM = LSTM.to(self.device)
        self.HF_encoder = self.HF_CAE.encoder
        self.HF_decoder = self.HF_CAE.decoder
        self.LF_encoder = self.LF_CAE[0]
        self.LF_decoder = self.LF_CAE[1]

    def train_LSTM_with_PC(self, train_loader, test_data, energy_coef=None, fo_coef=None, decoder_type=None, batch_size=20, num_epochs=30, lr=0.0001, criterion_type='mse'):
        # Coerce coefficients to lists if they are not
        if isinstance(energy_coef, (int, float)):
            energy_coef = [energy_coef] * num_epochs
        if isinstance(pde_coef, (int, float)):
            pde_coef = [pde_coef] * num_epochs
        # Load loss function
        criterion = load_loss_function(criterion_type)

        # Optimizers for the LSTM
        optimizer = torch.optim.Adam(self.LSTM.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

        # Set test dataset
        b, s, c, w, h = test_data.shape
        test_compr = self.HF_encoder(test_data.view(-1, c, w, h))
        test_dataset = dataloader.load_data(test_compr.reshape(b,s,-1), model='Seq2Seq', lookback=3, lookahead=3)

        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            train_epoch_loss, test_epoch_loss = 0., 0.

            # Training
            self.LSTM.train()
            for i, train_batch in enumerate(train_loader):
                b, s, c, h, w = train_batch.shape
                train_compr = self.HF_encoder(train_batch.view(-1, c, w, h))
                train_compr_dataset = dataloader.load_data(train_compr.reshape(b, s, -1).cpu().detach().numpy(), model='seq2seq', lookback=3, lookahead=3)
                for j in range(train_compr_dataset.__len__()):
                    inputs, targets = train_compr_dataset[j]
                    inputs = torch.tensor(inputs).to(self.device)
                    targets = torch.tensor(targets).to(self.device)

                    outputs = self.LSTM(inputs)

                    mse_loss = criterion(outputs, targets)
                    total_loss = mse_loss
                    if energy_coef is not None:
                        energy_loss = torch.abs(torch.mean((self.calculate_total_energy(inputs, decoder_type) -  #, dx=0.02
                                                    torch.mean(self.calculate_total_energy(outputs, decoder_type), dim=1, keepdim=True)))) # , dx=0.02
                        total_loss += energy_coef[epoch]*energy_loss
                    if fo_coef is not None:
                        fo_loss = self.compute_evolve_loss(inputs, outputs, decoder_type)
                        total_loss += fo_coef[epoch]*fo_loss

                    train_epoch_loss += total_loss.item()/len(train_loader)/len(train_compr_dataset)

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

            # Testing
            self.LSTM.eval()
            with torch.no_grad():
                for j in range(test_dataset.__len__()):
                    test_inputs, test_targets = test_dataset[j]
                    test_inputs = test_inputs.to(self.device)
                    test_targets = test_targets.to(self.device)

                    test_outputs = self.LSTM(test_inputs)

                    test_mse_loss = criterion(test_outputs, test_targets)
                    test_total_loss = test_mse_loss

                    if energy_coef is not None:
                        test_energy_loss = torch.abs(torch.mean((self.calculate_total_energy(test_inputs, decoder_type) -  #, dx=0.02
                                                    torch.mean(self.calculate_total_energy(test_outputs, decoder_type), dim=1, keepdim=True)))) # , dx=0.02
                        test_total_loss += energy_coef[epoch]*test_energy_loss
                    if fo_coef is not None:
                        test_fo_loss = self.compute_evolve_loss(test_inputs, test_outputs, decoder_type)
                        test_total_loss += fo_coef[epoch]*test_fo_loss

                    test_epoch_loss += test_total_loss.item()/len(test_data)

            # Print the averaged loss per epoch
            print ('Epoch [{}/{}], MSE_Loss: {:.6f}, Total_Loss: {:.6f}, Test_MSE_Loss: {:.6f}, Test_Total_Loss: {:.6f}'
                .format(epoch+1, num_epochs, mse_loss.item(), train_epoch_loss, test_mse_loss.item(), test_epoch_loss))

            train_losses.append(train_epoch_loss)
            test_losses.append(test_epoch_loss)

    def calculate_total_energy(self, data_compr, decoder_type):
        # Set decoder for physcial constraints
        if decoder_type == 'high':
            decoder = self.HF_decoder
            dx = 0.01
        else:
            decoder = self.LF_decoder
            dx = 0.02

        g = 9.8

        N, Nt, latent = data_compr.shape
        energies = torch.zeros(N, Nt)
        # reconstruct the data
        data = decoder(data_compr.reshape(N*Nt, latent))
        data = data.reshape(N, Nt, *data.shape[-3:])
        for i in range(N):
            for j in range(Nt):
                # Calculate kinetic energy
                kinetic_energy = 0.5 * (torch.sum(data[i][j][0]**2) + torch.sum(data[i][j][1]**2)) * dx**2

                # Calculate potential energy
                potential_energy = torch.sum(0.5 * g * data[i][j][2]**2) * dx**2

                # Calculate total energy
                energies[i, j] = kinetic_energy + potential_energy

        return energies
    
    def dxy(self, A, dx, axis=0):
        return (roll(A, -1, axis) - roll(A, 1, axis)) / (dx*2.)

    def d_dx(self, A, dx):
        return self.dxy(A, dx, 2)

    def d_dy(self, A, dx):
        return self.dxy(A, dx, 1)

    def d_dt(self, h, u, v, dx):
        for x in [h, u, v]:
            assert isinstance(x, ndarray) and not isinstance(x, matrix)
        g, b = 1., 0.2
        du_dt = -g*self.d_dx(h, dx) - b*u
        dv_dt = -g*self.d_dy(h, dx) - b*v
        H = 0 
        dh_dt = -self.d_dx(u * (H+h), dx) - self.d_dy(v * (H+h), dx)
        return dh_dt, du_dt, dv_dt
    
    def evolve(self, h, u, v, dt=0.0001, dx=0.01):
        dh_dt, du_dt, dv_dt = self.d_dt(h, u, v, dx)
        h += dh_dt * dt
        u += du_dt * dt
        v += dv_dt * dt
        return h, u, v

    def compute_evolve_loss(self, input_data_compr, outputdata_compr, decoder_type, dt=0.0001):
        if decoder_type == 'high':
            decoder = self.HF_decoder
            dx = 0.01
        else:
            decoder = self.LF_decoder
            dx = 0.02

        N, Nt_in, latent = input_data_compr.shape
        N, Nt_out, latent = input_data_compr.shape
        input_images = decoder(input_data_compr.reshape(N*Nt_in, latent))
        input_images = input_images.reshape(N, Nt_in, *input_images.shape[-3:])
        output_images = decoder(outputdata_compr.reshape(N*Nt_out, latent))
        output_images = output_images.reshape(N, Nt_out, *output_images.shape[-3:])
        u = input_images[:,-1,0,:].cpu().detach().numpy()
        v = input_images[:,-1,1,:].cpu().detach().numpy()
        h = input_images[:,-1,2,:].cpu().detach().numpy()
        output_images_evolved = torch.empty((N, Nt_out, *output_images.shape[-3:]))
        for i in range(Nt_out):
            h, u, v = self.evolve(h, u, v, dt, dx)
            output_image_evolved = torch.stack((torch.tensor(u), torch.tensor(v), torch.tensor(h)), dim=1)
            output_images_evolved[:, i, :, :, :] = output_image_evolved
        output_images_evolved = output_images_evolved.to(self.device)
        # calculate the asymmetry
        FO_loss = nn.MSELoss()(output_images_evolved, output_images)
        return FO_loss