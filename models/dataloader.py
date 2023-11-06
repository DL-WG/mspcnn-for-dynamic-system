import numpy as np
import pathlib
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class HighFidelityCAEDataset(Dataset):
    """
    Dataset for High Fidelity Convolutional AutoEncoder (CAE).
    
    Args:
        high_original_data (numpy.ndarray): High fidelity data.
    """
    def __init__(self, high_original_data):
        self.data = high_original_data

    def __len__(self):
        """Returns the total number of data samples."""
        return self.data.shape[0]

    def __getitem__(self, idx):
        """Fetches a sample at the given index."""
        return torch.from_numpy(self.data[idx]).float()
    
class LowFidelityCAEDataset(Dataset):
    """
    Dataset for Low Fidelity Convolutional AutoEncoder (CAE).
    
    Args:
        low_original_data (numpy.ndarray): Low fidelity data.
        high_original_data (numpy.ndarray): Corresponding high fidelity data.
    """
    def __init__(self, low_original_data, high_original_data):
        self.low_original_data = low_original_data
        self.high_original_data = high_original_data

    def __len__(self):
        """Returns the total number of data samples."""
        return self.low_original_data.shape[0]

    def __getitem__(self, idx):
        """Fetches a sample at the given index."""
        return torch.from_numpy(self.low_original_data[idx]).float(), torch.from_numpy(self.high_original_data[idx]).float()
    
class Seq2SeqDataset(Dataset):
    """
    Dataset for Sequence to Sequence models.
    
    Args:
        data (numpy.ndarray): Input data.
        lookback (int): Number of previous sequences to look back on.
        lookahead (int): Number of future sequences to predict.
    """
    def __init__(self, data, lookback, lookahead) -> None:
        super().__init__()
        self.data = data
        self.lookback = lookback
        self.lookahead = lookahead
    
    def __len__(self):
        """Returns the total number of sequence samples."""
        return self.data.shape[1] - self.lookback - self.lookahead + 1

    def __getitem__(self, index):
        """Fetches the input and target sequences based on lookback and lookahead."""
        inputs = torch.from_numpy(self.data[:, index:index+self.lookback, :]).float()
        targets = torch.from_numpy(self.data[:, index+self.lookback:index+self.lookback+self.lookahead, :]).float()
        return inputs, targets


def load_data(dataset_path='train', model=None, timesteps=3, lookahead=3, batch_size=20, shuffle=True, low_fidelity_path=None, test = False):
    """
    Creates a DataLoader based on the given model and dataset path.

    Args:
        dataset_path (str or numpy.ndarray): Path to the high fidelity dataset or the numpy array of the data. Defaults to 'train'.
        model (str): Type of model for which the data loader is created, can be "HighFidelityCAE", "LowFidelityCAE", "LSTM", or "Seq2Seq".
        timesteps (int): Number of previous sequences to look back on for Seq2Seq model. Defaults to 3.
        lookahead (int): Number of future sequences to predict for Seq2Seq model. Defaults to 3.
        batch_size (int): Number of samples per batch. Defaults to 32.
        shuffle (bool): Whether to shuffle the dataset. Defaults to True.
        low_fidelity_path (str, optional): Path to the low fidelity dataset when using "LowFidelityCAE" model.

    Raises:
        FileNotFoundError: If the given dataset path does not exist.
        ValueError: If provided invalid path/data if low fidelity and high fidelity data shapes mismatch.
        NotImplementedError: If provided model is not recognized or implemented.

    Returns:
        DataLoader: DataLoader object for the specified model.
    """
    # Load the high fidelity data
    if isinstance(dataset_path, str):
        dataset_path = pathlib.Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError("Failed to load. Give correct path")
        
        high_data = np.load(dataset_path)
        
    elif isinstance(dataset_path, np.ndarray):
        high_data = dataset_path
    else:
        raise ValueError('Invalid path/data provided. Provide data path or Numpy array')
    
    print("The received data shape is: ", high_data.shape)
    # Create dataset based on model
    if model == "HighFidelityCAE":
        # Merge the first two dimensions for CAE models
        data = high_data.reshape(-1, high_data.shape[2], high_data.shape[3], high_data.shape[4])
        if test == True: 
            print(f"Data loaded successfully")
            return torch.from_numpy(data).float()
        dataset = HighFidelityCAEDataset(data)
    
    elif model == "LowFidelityCAE":
        # Load low fidelity data
        if low_fidelity_path is None:
            raise ValueError("For LowFidelityCAE, provide the path for low fidelity data using 'low_fidelity_path' parameter.")
        if isinstance(low_fidelity_path, str):
            low_fidelity_path = pathlib.Path(low_fidelity_path)
            if not low_fidelity_path.exists():
                raise FileNotFoundError("Failed to load. Give correct path")
            
            low_data = np.load(low_fidelity_path)
        
        elif isinstance(low_fidelity_path, np.ndarray):
            low_data = low_fidelity_path
        else:
            raise ValueError('Invalid path/data provided. Provide data path or Numpy array')

        # Check the shapes of low fidelity and high fidelity data (for the first 3 dimensions)
        if low_data.shape[:3] != high_data.shape[:3]:
            raise ValueError("The first three dimensions of low fidelity data do not match with high fidelity data")

        # Merge the first two dimensions for CAE models
        low_data = low_data.reshape(-1, low_data.shape[2], low_data.shape[3], low_data.shape[4])
        high_data = high_data.reshape(-1, high_data.shape[2], high_data.shape[3], high_data.shape[4])
        if test==True: 
            print(f"Data loaded successfully")
            return torch.from_numpy(low_data).float(), torch.from_numpy(high_data).float()
        dataset = LowFidelityCAEDataset(low_data, high_data)

    elif model =="LSTM":
        high_data_tensor = torch.from_numpy(high_data).float()
        print(f"Data loaded successfully")
        return DataLoader(high_data_tensor, batch_size=batch_size, shuffle=shuffle)
    elif model == "Seq2Seq":
        print(f"Data loaded successfully")
        return Seq2SeqDataset(high_data, timesteps, lookahead)
    else:
        raise NotImplementedError(f'Dataloader for this {model} not implemented')
    
    print(f"Data loaded successfully")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
