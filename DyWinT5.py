import math
import torch
from torch import nn
import torch.nn.functional as F
from functorch import vmap
from einops import rearrange
import torchaudio

class PositionalEncoding2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.pos_scale = nn.Parameter(torch.ones(1))
    def forward(self, x):
        B, H, W, D = x.shape
        
        # Создаем позиционные индексы для высоты и ширины
        height_pos = torch.arange(H, device=x.device).float()
        width_pos = torch.arange(W, device=x.device).float()
        
        # Преобразуем индексы в embedding
        height_enc = self._positional_encoding(height_pos, self.dim)  # [H, D]
        width_enc = self._positional_encoding(width_pos, self.dim)    # [W, D]
        
        # Добавляем позиционное кодирование
        pos_enc = height_enc[:, None, :] + width_enc[None, :, :]  # [H, W, D]
        pos_enc = pos_enc.expand(B, -1, -1, -1)  # [B, H, W, D]
        
        return x + pos_enc* self.pos_scale

    def _positional_encoding(self, pos, dim):
        """
        Генерирует синусоидальное позиционное кодирование.
        pos: [L], где L — длина последовательности.
        dim: размерность embedding.
        """
        position = pos.unsqueeze(1)  # [L, 1]
        div_term = torch.exp(torch.arange(0, dim, 2, device=pos.device).float() * 
                           -(math.log(10000.0) / dim))  # [dim // 2]
        
        # Синусоидальное кодирование
        pe = torch.zeros(len(pos), dim, device=pos.device)  # [L, dim]
        pe[:, 0::2] = torch.sin(position * div_term)  # Четные индексы
        pe[:, 1::2] = torch.cos(position * div_term)  # Нечетные индексы
        return pe

    
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
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T, C]
        residual = x
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.pointwise_conv1(x)  # [B, 2*C, T]
        x = self.layer_norm(x)
        x = self.glu(x)  # [B, C, T]
        x = self.depthwise_conv(x)  # [B, C, T]
        x = x.transpose(1, 2)
        x = self.layer_norm2(x)
        x = x.transpose(1, 2)
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
        
        x = torch.tanh(x)
        return residual + 0.5 * x

    
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
    def __init__(self, sample_rate, n_fft, hop_length, n_mels, n_mfcc, hid_dim, window_size, overlap_ratio, vocab_size):
        super().__init__()
        self.hid_dim = hid_dim
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.stride = int(window_size * (1 - overlap_ratio))

        # Mel processing components
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels)
        self.mel_bn = nn.BatchNorm2d(1)
        self.mel_embedding = nn.Linear(n_mels, hid_dim)

        # MFCC processing components
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': n_fft, 'hop_length': hop_length, 'n_mels': n_mels})
        self.mfcc_bn = nn.BatchNorm2d(1)
        self.mfcc_embedding = nn.Linear(n_mfcc, hid_dim)

        # Shared components
        self.pos_enc = PositionalEncoding2D(hid_dim)
        self.conformer = ConformerEncoder(
            d_model=hid_dim, nhead=8, num_layers=6, kernel_size=31
        )
        self.cross_attention = nn.MultiheadAttention(hid_dim, num_heads=8, dropout=0.1, batch_first=True)
        self.fc = nn.Linear(hid_dim, vocab_size)
        self.norm = nn.LayerNorm(hid_dim)

    def _process_features(self, x, transform, bn, embedding):
        # Feature extraction
        x_features = transform(x).unsqueeze(1)
        x_features = bn(x_features).squeeze(1)

        # Project features
        B, D, T = x_features.shape
        x_features = x_features.permute(0, 2, 1)
        x_features = embedding(x_features)
        x_features = x_features.permute(0, 2, 1)  # [B, hid_dim, T]

        # Positional encoding
        x_features = x_features.permute(0, 2, 1).unsqueeze(2)  # [B, T, 1, D]
        x_features = self.pos_enc(x_features).squeeze(2).permute(0, 2, 1)  # [B, D, T]

        # Windowing
        windows = x_features.unfold(2, self.window_size, self.stride) # Removed padding
        B, D, N, W = windows.shape
        windows = windows.permute(0, 2, 3, 1).reshape(B*N, W, D) # [B*N, W, D]

        return windows # Возвращаем окна вместо собранной последовательности

    def forward(self, x):
        # Process Mel and MFCC to get windows
        mel_windows = self._process_features(x, self.mel_transform, self.mel_bn, self.mel_embedding) # [B*N, W, D]
        mfcc_windows = self._process_features(x, self.mfcc_transform, self.mfcc_bn, self.mfcc_embedding) # [B*N, W, D]
        mfcc_windows = torch.nan_to_num(mfcc_windows)
        mel_windows = torch.nan_to_num(mel_windows)
        # Убедитесь, что количество окон и их размер совпадают для Mel и MFCC
        B_mel_n, W_mel, D_mel = mel_windows.shape
        B_mfcc_n, W_mfcc, D_mfcc = mfcc_windows.shape
        B = x.shape[0]
        N = B_mel_n // B
        W = W_mel
        D = D_mel
        mel_orig_length = self.mel_transform(x).shape[-1]

        # Применяем кросс-внимание к окнам
        attn_out_windows, _ = self.cross_attention(
            mfcc_windows.permute(1, 0, 2), # Q: [W, B*N, D]
            mel_windows.permute(1, 0, 2),  # K: [W, B*N, D]
            mfcc_windows.permute(1, 0, 2)  # V: [W, B*N, D]
        ) # attn_out_windows: [W, B*N, D]

        attn_out_windows = torch.tanh(attn_out_windows)
        attn_out_windows = attn_out_windows.permute(1, 0, 2) # [B*N, W, D]

        # Ручная сборка с использованием overlap-add
        output = torch.zeros((B, mel_orig_length, D), device=attn_out_windows.device)
        attn_out_windows_reshaped = attn_out_windows.reshape(B, N, W, D)

        for b in range(B):
            for n in range(N):
                start_time = n * self.stride
                end_time = min(start_time + self.window_size, mel_orig_length)
                window = attn_out_windows_reshaped[b, n, :end_time - start_time, :]
                output[b, start_time:end_time, :] += window

        return self.fc(torch.clamp(self.norm(output), min=-1.0, max=1.0))

        
if __name__ == "__main__":
    # Параметры
    hid_dim = 256
    n_mels = 128
    n_mfcc = 128
    n_fft = 512
    window_size = 100
    overlap_ratio = 0.5
    vocab_size = 100

    # Создаём модель
    
    model = ConformerSpectrogramTransformer(16000, n_fft, 160, n_mels, n_mfcc, hid_dim, window_size, overlap_ratio, vocab_size).cuda()
    model.eval()
    
    test_data = torch.randn(4,16000*200).cuda()
    import time
    with torch.amp.autocast("cuda"):
        st = time.time()
        output = model(test_data)
    print("Время выполнения: ", time.time() - st)
    print("Размер входных данных:", test_data.shape)
    print("Размер выходных данных:", output.shape)