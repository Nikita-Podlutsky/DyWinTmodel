import torch
from torch import nn
import torch.nn.functional as F
from functorch import vmap
from einops import rearrange

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
        self.pointwise_conv1 = nn.Conv1d(
            d_model, d_model * expansion_factor, kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
            groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            d_model, d_model, kernel_size=1
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T, C]
        residual = x
        x = self.layer_norm(x)
        
        # Меняем порядок размерностей для сверточных слоев
        x = x.transpose(1, 2)  # [B, C, T]
        
        x = self.pointwise_conv1(x)  # [B, 2*C, T]
        x = self.glu(x)  # [B, C, T]
        x = self.depthwise_conv(x)  # [B, C, T]
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)  # [B, C, T]
        x = self.dropout(x)
        
        # Возвращаем исходный порядок размерностей
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
        return residual + 0.5 * x  # Масштабирование 0.5 как в оригинальной статье


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
        # FFN 1
        x = self.ffn1(x)
        
        # Self-Attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = residual + x
        
        # Convolution
        x = self.conv_module(x)
        
        # FFN 2
        x = self.ffn2(x)
        
        # Final Layer Norm
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
    def __init__(self, hid_dim, height, width, window_size, overlap_ratio, vocab_size, vocab=None):
        super().__init__()
        self.hid_dim = hid_dim
        self.height = height
        self.width = width
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.vocab_size = vocab_size
        self.vocab = vocab

        self.embedding = nn.Linear(height, hid_dim)
        self.pos_enc = PositionalEncoding2D(hid_dim, height, window_size)

        # Заменяем стандартный трансформер на Conformer
        self.conformer_encoder = ConformerEncoder(
            d_model=hid_dim, 
            nhead=8, 
            num_layers=6, 
            kernel_size=31
        )

        self.fc = nn.Linear(hid_dim, vocab_size)
        self.softmax = nn.LogSoftmax()

    def create_windows(self, x):
        N, C, H, W = x.shape
        stride = int(self.window_size * (1 - self.overlap_ratio))
        num_windows = (W - self.window_size) // stride + 1

        # Создаем окна эффективно с использованием unfold
        # Сначала перегруппируем размерности для корректного использования unfold
        x_reshaped = x.view(N*C, H, W)
        
        # Применяем unfold для создания окон вдоль временной оси
        windows = x_reshaped.unfold(2, self.window_size, stride)
        
        # Восстанавливаем размерности пакета
        windows = windows.reshape(N, C, H, num_windows, self.window_size)
        
        # Перегруппируем размерности для совместимости с дальнейшей обработкой
        windows = windows.permute(0, 3, 1, 2, 4)  # [N, num_windows, C, H, window_size]
        
        # Проверка наличия непокрытой части
        last_w_start = (num_windows - 1) * stride + self.window_size
        if last_w_start < W:
            last_window = x[:, :, :, last_w_start:]
            pad_size = self.window_size - last_window.shape[-1]
            last_window = F.pad(last_window, (0, pad_size), "constant", 0)
            last_window = last_window.unsqueeze(1)  # [N, 1, C, H, window_size]
            windows = torch.cat([windows, last_window], dim=1)
        
        return windows
    
    def forward(self, x):
        final_len = x.shape[-1]
        windows = self.create_windows(x)
        N, num_windows, C, H, win_size = windows.shape

        processed_windows = []
        for i in range(num_windows):
            window = windows[:, i].squeeze(1)
            window = window.permute(0, 2, 1)
            window = self.embedding(window)
            window = self.pos_enc(window)
            window = self.conformer_encoder(window)
            processed_windows.append(window)

        stride = int(self.window_size * (1 - self.overlap_ratio))
        for i in range(1, len(processed_windows)):
            processed_windows[i] = processed_windows[i][:, stride:, :]

        batched_windows = [torch.cat([pw[b] for pw in processed_windows], dim=0) for b in range(N)]
        processed_sequence = torch.stack(batched_windows)

        output = self.softmax(self.fc(processed_sequence[:, :final_len]))
        return output


if __name__ == "__main__":
    # Параметры
    hid_dim = 64
    height = 128
    width = 100
    window_size = 32
    overlap_ratio = 0.5
    vocab_size = 100

    # Создаем модель
    model = ConformerSpectrogramTransformer(hid_dim, height, width, window_size, overlap_ratio, vocab_size)

    # Тестовые данные
    test_data = torch.randn(4, 1, height, 10000)
    import time
    st = time.time()
    # Прогон через модель
    output = model(test_data)
    print(time.time()-st)

    print("Размерность входных данных:", test_data.shape)
    print("Размерность выходных данных:", output.shape)