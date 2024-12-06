import torch
import pytest
import sys
import os
from torchvision import datasets, transforms

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTModel

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_model_parameters():
    model = MNISTModel()
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params}")
    assert num_params < 100000, "Model has too many parameters"

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), "Output shape is incorrect"

def test_model_accuracy():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel().to(device)
    
    # Load the latest trained model
    models_dir = 'models'
    try:
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not model_files:
            pytest.skip("No model file found")
        latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
        model.load_state_dict(torch.load(os.path.join(models_dir, latest_model), map_location=device))
    except FileNotFoundError:
        pytest.skip("Models directory not found")
    
    # Prepare test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000)
    except Exception as e:
        pytest.skip(f"Failed to load test dataset: {str(e)}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Model accuracy: {accuracy:.2f}%")
    assert accuracy > 80, f"Model accuracy {accuracy:.2f}% is below 80%" 