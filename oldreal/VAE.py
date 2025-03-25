import torch
from torch import optim, nn
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, latent_dim * 2, kernel_size=3, stride=1, padding=1)
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, input_dim, kernel_size=3, stride=1, padding=1)
        )
    
    
    
    
    # Для создания полуслучайного обучения стандартное отклонение совмещаем с случайными числами
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # Энкодирование
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mu, logvar)
        
        # Декодирование
        recon_x = self.decoder(z)
        return recon_x, mu, logvar


class PhonemeAlignmentModel(nn.Module):
    def __init__(self, acoustic_dim, linguistic_dim, hidden_dim, latent_dim, num_states_per_phoneme):
        super(PhonemeAlignmentModel, self).__init__()
        
        self.num_states_per_phoneme = num_states_per_phoneme
        
        # Акустический и лингвистический энкодеры
        self.acoustic_vae = VAE(acoustic_dim, hidden_dim, latent_dim)
        self.linguistic_vae = VAE(linguistic_dim, hidden_dim, latent_dim)
        
        # Градиентное аннелирование
        self.annealing_sigma = 30.0
        self.annealing_rate = 0.9
        
    def forward(self, acoustic_input, linguistic_input):
        # Энкодирование акустических и лингвистических признаков
        acoustic_recon, acoustic_mu, acoustic_logvar = self.acoustic_vae(acoustic_input)
        linguistic_recon, linguistic_mu, linguistic_logvar = self.linguistic_vae(linguistic_input)
        
        # Вычисление потерь VAE
        acoustic_kld = -0.5 * torch.sum(1 + acoustic_logvar - acoustic_mu.pow(2) - acoustic_logvar.exp())
        linguistic_kld = -0.5 * torch.sum(1 + linguistic_logvar - linguistic_mu.pow(2) - linguistic_logvar.exp())
        
        # Вычисление потерь выравнивания
        alignment_loss, probs = self.compute_alignment_loss(acoustic_mu, linguistic_mu)
        
        # Общие потери
        total_loss = alignment_loss + 0.1 * (acoustic_kld + linguistic_kld)
        return total_loss, probs
    
    def compute_alignment_loss(self, acoustic_emb, linguistic_emb):
        # Вычисление вероятностей совпадения
        logits = -torch.cdist(acoustic_emb.permute(0, 2, 1), linguistic_emb.permute(0, 2, 1), p=2)  # Отрицательное евклидово расстояние
        probs = F.softmax(logits, dim=-1)
        
        # Градиентное аннелирование
        gamma = self.anneal_gradients(probs)
        
        # Потери выравнивания
        alignment_loss = -torch.sum(gamma * torch.log(probs))
        return alignment_loss, probs
    
    def anneal_gradients(self, probs):
        # Применение градиентного аннелирования
        sequence_length = probs.size(1) 
        gaussian_filter = torch.exp(-torch.arange(sequence_length).float().pow(2) / (2 * self.annealing_sigma ** 2))
        gaussian_filter = gaussian_filter / gaussian_filter.sum()
        
        # Подготовка фильтра для свертки
        gaussian_filter = gaussian_filter.unsqueeze(0).unsqueeze(0)  # (1, 1, sequence_length)
        
        # Применение свертки по оси времени
        gamma = []
        for i in range(probs.size(2)):
            probs_slice = probs[:, :, i].unsqueeze(1)
            gamma_slice = F.conv1d(probs_slice, gaussian_filter, padding='same') 
            gamma.append(gamma_slice.squeeze(1))
        
        gamma = torch.stack(gamma, dim=2)
        
        # Обновление коэффициента аннелирования
        self.annealing_sigma *= self.annealing_rate
        return gamma



# Параметры модели
acoustic_dim = 80
linguistic_dim = 52  
hidden_dim = 256
latent_dim = 64
num_states_per_phoneme = 3

# Создание модели
model = PhonemeAlignmentModel(acoustic_dim, linguistic_dim, hidden_dim, latent_dim, num_states_per_phoneme)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Пример данных (замените на реальные данные)
acoustic_input = torch.randn(4, acoustic_dim, 1000)
linguistic_input = torch.randn(4, linguistic_dim, 50)


for epoch in range(10):
    optimizer.zero_grad()
    loss, _ = model(acoustic_input, linguistic_input)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Загрузка новых данных
new_acoustic_input = torch.randn(1, acoustic_dim, 200)  # Новые акустические данные
new_linguistic_input = torch.randn(1, linguistic_dim, 100)  # Новые лингвистические данные

# Предсказание границ фонем
with torch.no_grad():
    pred = model(new_acoustic_input, new_linguistic_input)

print("Границы:", pred)