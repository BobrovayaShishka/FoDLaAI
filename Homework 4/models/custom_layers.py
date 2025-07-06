import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomConv2d(nn.Module):
    """Кастомный сверточный слой с регуляризацией"""
    def __init__(self, in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 l1_lambda=0.01, spectral_norm=False):
        super().__init__()
        self.l1_lambda = l1_lambda
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride, padding, dilation, groups, bias
        )
        if spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)
            
    def forward(self, x):
        return self.conv(x)
    
    def regularization_loss(self):
        if self.l1_lambda > 0:
            return self.l1_lambda * torch.norm(self.conv.weight, p=1)
        return 0

class AttentionBlock(nn.Module):
    """Attention механизм для CNN"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, 1)
        self.key = nn.Conv2d(channels, channels // 8, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        # Генерация query, key, value
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        
        # Расчет матрицы внимания
        attention = torch.bmm(q, k)
        attention = F.softmax(attention, dim=-1)
        
        # Применение внимания
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x

class Swish(nn.Module):
    """Кастомная функция активации Swish"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class CustomMaxPool(nn.Module):
    """Кастомный pooling слой с сохранением информации"""
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        
    def forward(self, x):
        pooled = self.pool(x)
        # Сохранение индексов для последующего использования
        self.indices = self.pool.indices
        return pooled

class BottleneckResidualBlock(nn.Module):
    """Bottleneck Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)

class WideResidualBlock(nn.Module):
    """Wide Residual Block"""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
