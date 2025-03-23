import torch
from torch import nn
import torch.nn.functional as F

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


class MelSpectrogramTransformer(nn.Module):
    def __init__(self, hid_dim, height, width, window_size, overlap_ratio, vocab_size, vocab):
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

        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.fc = nn.Linear(hid_dim, vocab_size)
        self.softmax = nn.Softmax()

    def create_windows(self, x):
        N, C, H, W = x.shape
        stride = int(self.window_size * (1 - self.overlap_ratio))  # Шаг с учетом перекрытия
        num_windows = (W - self.window_size) // stride + 1

        windows = []
        for i in range(num_windows):
            w_start = i * stride
            w_end = w_start + self.window_size
            windows.append(x[:, :, :, w_start:w_end])  # (N, C, H, window_size)

        # Проверяем, осталась ли непокрытая часть
        last_w_start = num_windows * stride
        if last_w_start < W:
            last_window = x[:, :, :, last_w_start:]  # (N, C, H, оставшееся)
            pad_size = self.window_size - last_window.shape[-1]
            last_window = F.pad(last_window, (0, pad_size), "constant", 0)  # (N, C, H, window_size)
            windows.append(last_window)

        return torch.stack(windows, dim=1)  # (N, num_windows + 1, C, H, window_size)

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
            window = self.transformer_encoder(window)
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
    model = MelSpectrogramTransformer(hid_dim, height, width, window_size, overlap_ratio, vocab_size, vocab = ["F"])

    # Тестовые данные
    test_data = torch.randn(4, 1, height, 1000)

    # Прогон через модель
    output = model(test_data)

    print("Размерность входных данных:", test_data.shape)
    print("Размерность выходных данных:", output.shape)
