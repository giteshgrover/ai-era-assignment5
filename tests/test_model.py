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
    assert num_params < 25000, f"Model has {num_params} parameters, which exceeds limit of 25000"
    print(f"✓ Model parameter count ({num_params}) is within limit")

def test_model_architecture():
    model = MNISTModel()
    
    # Test input shape
    test_input = torch.randn(1, 1, 28, 28)
    
    # Test forward pass and shape at each layer
    x = model.conv1(test_input)
    assert x.shape == (1, 4, 28, 28), "Conv1 output shape incorrect"
    x = model.pool(model.relu(x))
    assert x.shape == (1, 4, 14, 14), "After first pooling shape incorrect"
    
    x = model.conv2(x)
    assert x.shape == (1, 8, 14, 14), "Conv2 output shape incorrect"
    x = model.pool(model.relu(x))
    assert x.shape == (1, 8, 7, 7), "After second pooling shape incorrect"
    
    x = model.conv3(x)
    assert x.shape == (1, 16, 7, 7), "Conv3 output shape incorrect"
    x = model.pool(model.relu(x))
    assert x.shape == (1, 16, 3, 3), "After third pooling shape incorrect"
    
    print("✓ All layer shapes are correct")

def test_input_output_shape():
    model = MNISTModel()
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    assert output.shape == (1, 10), f"Output shape is {output.shape}, expected (1, 10)"
    print("✓ Model input/output shapes are correct")

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
        print(f"✓ Loaded model: {latest_model}")
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
        print("✓ Test dataset loaded successfully")
    except Exception as e:
        pytest.skip(f"Failed to load test dataset: {str(e)}")
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if batch_idx % 10 == 0:
                print(f"Processed {total} test images...")
    
    accuracy = 100 * correct / total
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    assert accuracy > 95, f"Model accuracy {accuracy:.2f}% is below 95%"
    print(f"✓ Model achieved accuracy of {accuracy:.2f}%")

def test_model_robustness():
    model = MNISTModel()
    
    # Test with different batch sizes
    batch_sizes = [1, 32, 64, 128]
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 1, 28, 28)
        output = model(test_input)
        assert output.shape == (batch_size, 10), f"Failed for batch size {batch_size}"
    
    # Test with different input intensities
    test_input = torch.randn(1, 1, 28, 28) * 2.5  # Test with higher intensity
    output = model(test_input)
    assert not torch.isnan(output).any(), "Model produced NaN values"
    
    print("✓ Model is robust to different batch sizes and input intensities")