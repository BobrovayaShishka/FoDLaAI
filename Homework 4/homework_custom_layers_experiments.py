import torch
import torch.nn as nn
import torch.nn.functional as F
from models import custom_layers
from datasets import get_cifar_loaders
from utils.training_utils import train_model, count_parameters
from utils.visualization_utils import plot_training_history

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_custom_layers():
    """Тестирование кастомных слоев"""
    print("\nTesting custom layers...")
    
    # Тест CustomConv2d
    conv = custom_layers.CustomConv2d(3, 16, 3, padding=1).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    print("CustomConv output shape:", conv(x).shape)
    
    # Тест AttentionBlock
    attn = custom_layers.AttentionBlock(64).to(device)
    x = torch.randn(2, 64, 16, 16).to(device)
    print("Attention output shape:", attn(x).shape)
    
    # Тест Swish
    swish = custom_layers.Swish()
    x = torch.randn(2, 64)
    print("Swish output:", swish(x).shape)
    
    # Тест CustomMaxPool
    pool = custom_layers.CustomMaxPool(2).to(device)
    x = torch.randn(2, 3, 32, 32).to(device)
    print("CustomPool output shape:", pool(x).shape)

def residual_block_experiment():
    """Сравнение Residual блоков"""
    print("\nComparing residual blocks...")
    save_dir = "results/custom_layers/residual_blocks"
    
    train_loader, test_loader = get_cifar_loaders(128)
    
    blocks = {
        "Basic": custom_layers.ResidualBlock,
        "Bottleneck": custom_layers.BottleneckResidualBlock,
        "Wide": custom_layers.WideResidualBlock
    }
    
    results = {}
    for name, block in blocks.items():
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)
                self.block1 = block(64, 128, stride=2)
                self.block2 = block(128, 256, stride=2)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(256, 10)
            
            def forward(self, x):
                x = F.relu(self.bn1(self.conv1(x)))
                x = self.block1(x)
                x = self.block2(x)
                x = self.pool(x)
                return self.fc(x.view(x.size(0), -1))
        
        model = TestModel().to(device)
        print(f"\nTraining {name} Residual Block...")
        print(f"Parameters: {count_parameters(model):,}")
        
        history = train_model(
            model, train_loader, test_loader, 
            epochs=10, device=device
        )
        
        results[name] = {
            "history": history,
            "params": count_parameters(model),
            "test_acc": history['test_accs'][-1]
        }
        
        plot_training_history(
            history, 
            f"{name} Residual Block Training",
            save_dir
        )
    
    # Сравнение результатов
    print("\nResidual Block Comparison:")
    for name, res in results.items():
        print(f"{name}: Test Acc={res['test_acc']:.4f}, Params={res['params']:,}")

if __name__ == "__main__":
    test_custom_layers()
    residual_block_experiment()
