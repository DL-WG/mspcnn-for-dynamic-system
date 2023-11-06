import numpy as np
from dataloader import load_data  # Assuming your original file is named data_loader.py

def test_high_fidelity_cae():
    # Generate mock data
    data = np.random.rand(100, 10, 3, 32, 32)  # [batch_size, timesteps, channels, width, height]

    # Load data
    dataloader = load_data(data, model="HighFidelityCAE")

    # Iterate over the dataloader
    for batch in dataloader:
        assert batch.shape == (20, 3, 32, 32)

def test_low_fidelity_cae():
    # Generate mock data
    high_data = np.random.rand(100, 10, 3, 32, 32)
    low_data = np.random.rand(100, 10, 3, 28, 28)

    # Load data
    dataloader = load_data(high_data, model="LowFidelityCAE", low_fidelity_path=low_data)

    # Iterate over the dataloader
    for low_batch, high_batch in dataloader:
        assert low_batch.shape == (20, 3, 28, 28)
        assert high_batch.shape == (20, 3, 32, 32)

def test_seq2seq():
    # Generate mock data
    data = np.random.rand(100, 20, 512)  # [batch_size, timesteps, channels, width, height]

    # Load data
    dataset = load_data(data, model="Seq2Seq")

    # Iterate over the dataloader
    for j in range(dataset.__len__()):
        inputs, targets = dataset[j]
        assert inputs.shape == (100, 3, 512)
        assert targets.shape == (100, 3, 512)
def test_high_fidelity_cae_data_return():
    # Generate mock data
    data = np.random.rand(100, 10, 3, 32, 32)  # [batch_size, timesteps, channels, width, height]

    # Load data
    returned_data = load_data(data, model="HighFidelityCAE", test=True)

    # Check shape of returned data
    assert returned_data.shape == (1000, 3, 32, 32)

def test_low_fidelity_cae_data_return():
    # Generate mock data
    high_data = np.random.rand(100, 10, 3, 32, 32)
    low_data = np.random.rand(100, 10, 3, 28, 28)

    # Save mock data to numpy files
    np.save('high_data.npy', high_data)
    np.save('low_data.npy', low_data)

    # Load data
    low_returned_data, high_returned_data = load_data('high_data.npy', model="LowFidelityCAE", low_fidelity_path='low_data.npy', test=True)

    # Check shape of returned data
    assert low_returned_data.shape == (1000, 3, 28, 28)
    assert high_returned_data.shape == (1000, 3, 32, 32)

if __name__ == "__main__":
    print("-------------Test High Fidelity CAE Data Return-------------")
    test_high_fidelity_cae_data_return()
    print("-------------Test Low Fidelity CAE Data Return--------------")
    test_low_fidelity_cae_data_return()
    print("-------------Test High Fidelity CAE-------------")
    test_high_fidelity_cae()
    print("-------------Test Low Fidelity CAE--------------")
    test_low_fidelity_cae()
    print("-----------------Test Seq2Seq-------------------")
    test_seq2seq()

    print("All tests passed!")

