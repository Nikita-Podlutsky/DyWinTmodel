import torch
from torch import nn
import torch.nn.functional as F
from functorch import vmap
from einops import rearrange
import torchaudio

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, height: int, width: int):
        super().__init__()
        pe_height = torch.zeros(height, d_model)
        pe_width = torch.zeros(width, d_model)

        position_height = torch.arange(0, height, dtype=torch.float).unsqueeze(1)
        position_width = torch.arange(0, width, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))

        pe_height[:, 0::2] = torch.sin(position_height * div_term)
        pe_height[:, 1::2] = torch.cos(position_height * div_term)
        pe_width[:, 0::2] = torch.sin(position_width * div_term)
        pe_width[:, 1::2] = torch.cos(position_width * div_term)

        self.register_buffer('pe_height', pe_height)
        self.register_buffer('pe_width', pe_width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        position_enc = torch.zeros_like(x)
        for pos in range(seq_len):
            position_enc[:, pos, :] = torch.cat([
                torch.sin(pos / torch.pow(10000, torch.arange(0, dim, 2, device=x.device) / dim)),
                torch.cos(pos / torch.pow(10000, torch.arange(1, dim, 2, device=x.device) / dim))
            ], dim=0)[:dim].unsqueeze(0)
        return x + position_enc

    
class ConvolutionModule(nn.Module):
    """Конволюционный модуль с гейтами для Conformer."""
    def __init__(self, d_model, kernel_size=31, expansion_factor=2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * expansion_factor, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T, C]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)  # [B, 2*C, T]
        x = self.glu(x)  # [B, C, T]
        x = self.depthwise_conv(x)  # [B, C, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # [B, C, T]
        x = self.dropout(x)
        x = x.transpose(1, 2)  # [B, T, C]
        return residual + x

    
class FeedForwardModule(nn.Module):
    """Модуль FFN для Conformer."""
    def __init__(self, d_model, expansion_factor=4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * expansion_factor)
        self.activation = nn.SiLU()
        self.dropout1 = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_model * expansion_factor, d_model)
        self.dropout2 = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x  # масштабирование, как в оригинальной статье

    
class ConformerBlock(nn.Module):
    """Блок Conformer с архитектурой FFN -> Self-Attention -> Conv -> FFN."""
    def __init__(self, d_model, nhead=8, kernel_size=31):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1, batch_first=True)
        self.conv_module = ConvolutionModule(d_model, kernel_size)
        self.ffn2 = FeedForwardModule(d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None):
        x = self.ffn1(x)
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        x = self.conv_module(x)
        x = self.ffn2(x)
        x = self.final_layer_norm(x)
        return x

    
class ConformerEncoder(nn.Module):
    """Encoder на основе Conformer."""
    def __init__(self, d_model, nhead=8, num_layers=6, kernel_size=31):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(d_model, nhead, kernel_size) for _ in range(num_layers)
        ])
    
    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

    
class ConformerSpectrogramTransformer(nn.Module):
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, hid_dim, window_size, overlap_ratio, vocab_size):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_mels = n_mels
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        

        # Проекция вдоль оси частот
        self.embedding = nn.Linear(n_mels, hid_dim)
        self.pos_enc = PositionalEncoding2D(hid_dim, n_mels, window_size)
        
        self.conformer_encoder = ConformerEncoder(
            d_model=hid_dim, nhead=8, num_layers=6, kernel_size=31
        )

        self.fc = nn.Linear(hid_dim, vocab_size)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels)
        
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        
        if x.dim() == 3:
            x = self.mel_transform(x)
        B, C, H, T = x.shape
        
        stride = int(self.window_size * (1 - self.overlap_ratio))  # например, 16 при window_size=32 и overlap_ratio=0.5

        x = self.batch_norm(x)
        
        # ПАДДИНГ: добавляем по оси времени по window_size//2 с каждой стороны
        x_padded = F.pad(x, (self.window_size//2, self.window_size//2))
        T_pad = T + self.window_size  # новая длина по времени

        # Извлечение окон по оси времени с помощью unfold
        # Получаем окна shape: [B, C, H, num_windows, window_size]
        windows = x_padded.unfold(dimension=3, size=self.window_size, step=stride)
        # Переставляем оси: [B, num_windows, C, H, window_size]
        windows = windows.permute(0, 3, 1, 2, 4)
        # Если C==1, убираем измерение каналов: [B, num_windows, H, window_size]
        windows = windows.reshape(B, windows.shape[1], H, self.window_size)

        # Подготовка окон для Conformer: объединяем батч и окна
        B, N, H, W = windows.shape  # W == window_size
        windows = windows.reshape(B * N, H, W)
        # Переставляем так, чтобы размер окна (время) был первым: [B*N, window_size, H]
        windows = windows.permute(0, 2, 1)
        
        # Проекция по частотам и позиционное кодирование
        x_proj = self.embedding(windows)  # [B*N, window_size, hid_dim]
        x_proj = x_proj + self.pos_enc(x_proj)
        
        # Обработка через Conformer Encoder
        x_convformer = self.conformer_encoder(x_proj)  # [B*N, window_size, hid_dim]
        
        # Возвращаем обратно оконную форму: [B, N, window_size, hid_dim]
        x_convformer = x_convformer.reshape(B, N, W, self.hid_dim)
        # Переставляем оси для fold: [B, hid_dim, num_windows, window_size]
        x_convformer = x_convformer.permute(0, 3, 1, 2)
        
        # Подготовка для fold:
        # Объединяем измерения hid_dim и window_size в канал
        B, C_conv, N, W = x_convformer.shape  # C_conv == hid_dim
        x_fold = x_convformer.reshape(B, C_conv * W, N)
        
        # Реконструкция с помощью F.fold:
        # Здесь output_size соответствует размеру ПАДДИНГОВОГО сигнала: (1, T_pad)
        # Важно: padding=0, так как паддинг уже применён при извлечении окон.
        output = F.fold(
            x_fold,
            output_size=(1, T_pad),
            kernel_size=(1, self.window_size),
            stride=(1, stride),
            padding=(0, 0)
        )
        # Получаем выход [B, hid_dim, 1, T_pad], приводим к [B, T_pad, hid_dim]
        output = output.view(B, self.hid_dim, T_pad).permute(0, 2, 1)
        # Убираем добавленный паддинг: оставляем T значащих временных шагов
        output = output[:, self.window_size//2 : self.window_size//2 + T, :]
        
        logits = self.fc(output)  # [B, T, vocab_size]
        return logits

        
if __name__ == "__main__":
    # Параметры
    hid_dim = 128
    n_mels = 128
    
    window_size = 80
    overlap_ratio = 0.5
    vocab_size = 100

    # Создаём модель
    
    model = ConformerSpectrogramTransformer(48000, 512, 160, n_mels, hid_dim, window_size, overlap_ratio, vocab_size).cuda()
    model.eval()
    # Тестовые данные: [B, C, H, T]
    test_data = torch.randn(4,1,1000000).cuda()
    import time
    st = time.time()
    output = model(test_data)
    print("Время выполнения: ", time.time() - st)
    print("Размер входных данных:", test_data.shape)
    print("Размер выходных данных:", output.shape)