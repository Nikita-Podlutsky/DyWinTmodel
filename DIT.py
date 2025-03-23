import torch
import torch.nn as nn
import torch.nn.functional as F
import random

#######################################################################
# 1. Разрежённое выравнивание (Sparse Alignment)
#######################################################################

def generate_sparse_alignment(phonemes, durations, mask_token_id, keep_random=True):
    """
    Генерирует разрежённое выравнивание на основе входных фонем и их длительностей.
    
    Для каждой фонемы из последовательности:
      - Повторяем её d_i раз (где d_i — длительность фонемы).
      - Сохраняем значение фонемы только в одном случайно выбранном месте (якорь).
      - Остальные позиции заменяем на mask_token_id.
    
    Пример:
      phonemes = [1, 2, 3]
      durations = [2, 2, 3]
      Возможный результат: [M, 1, 2, M, M, M, 3] (где M = mask_token_id)
    
    Аргументы:
      phonemes: список фонем (например, [1, 2, 3])
      durations: список длительностей для каждой фонемы (например, [2, 2, 3])
      mask_token_id: идентификатор маски (например, 0)
      keep_random: если True, якорь выбирается случайным образом для каждой фонемы;
                   если False, всегда сохраняется первая позиция.
    
    Возвращает:
      Список длины sum(durations), представляющий разрежённое выравнивание.
    """
    sparse_alignment = []
    for p, d in zip(phonemes, durations):
        anchor_index = random.randint(0, d - 1) if keep_random else 0
        for i in range(d):
            if i == anchor_index:
                sparse_alignment.append(p)
            else:
                sparse_alignment.append(mask_token_id)
    return sparse_alignment

#######################################################################
# 2. Понижающая дискретизация выравнивания и объединение с латентным представлением
#######################################################################

class AlignmentDownsampler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_dim, target_length):
        """
        Модуль для понижающей дискретизации разрежённого выравнивания.
        
        Порядок действий:
          1. Преобразование идентификаторов фонем в эмбеддинги.
          2. Применение свёрточного слоя для получения локальной информации.
          3. Интерполяция результата до целевой длины (длина латентного представления).
        
        Аргументы:
          vocab_size: размер словаря фонем.
          embedding_dim: размерность эмбеддингов.
          out_dim: число выходных каналов после свёрточного слоя.
          target_length: целевая длина последовательности (например, длина латентного представления z).
        """
        super(AlignmentDownsampler, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        # Свёрточный слой с kernel_size=3; padding=1 сохраняет размерность по временной оси
        self.conv = nn.Conv1d(in_channels=embedding_dim, out_channels=out_dim, kernel_size=3, padding=1)
        self.target_length = target_length

    def forward(self, token_ids):
        """
        Преобразует последовательность идентификаторов в эмбеддинги, сворачивает и интерполирует.
        
        Аргументы:
          token_ids: тензор размера (batch_size, seq_length).
          
        Возвращает:
          Тензор размера (batch_size, out_dim, target_length).
        """
        x = self.embedding(token_ids)            # (batch, seq_length, embedding_dim)
        x = x.transpose(1, 2)                      # (batch, embedding_dim, seq_length)
        x = self.conv(x)                           # (batch, out_dim, seq_length)
        # Интерполируем до target_length, чтобы соответствовать длине латентного представления
        x = F.interpolate(x, size=self.target_length, mode='linear', align_corners=False)
        return x

class SDiTAlignment(nn.Module):
    def __init__(self, vocab_size, embedding_dim, alignment_out_dim, target_length):
        """
        Модуль для объединения латентного представления речи с информацией разрежённого выравнивания.
        
        Используется для предоставления грубой позиционной информации, полученной из разрежённого выравнивания.
        
        Аргументы:
          vocab_size: размер словаря фонем.
          embedding_dim: размерность эмбеддингов.
          alignment_out_dim: число выходных каналов от AlignmentDownsampler.
          target_length: целевая длина временной оси (должна совпадать с длиной латентного представления).
        """
        super(SDiTAlignment, self).__init__()
        self.downsampler = AlignmentDownsampler(vocab_size, embedding_dim, alignment_out_dim, target_length)

    def forward(self, z, token_ids):
        """
        Объединяет латентное представление z и пониженную версию выравнивания.
        
        Аргументы:
          z: латентное представление речи, тензор размера (batch, latent_channels, T).
          token_ids: тензор размера (batch, seq_length) с идентификаторами разрежённого выравнивания.
          
        Возвращает:
          Тензор размера (batch, latent_channels + alignment_out_dim, T).
        """
        a_down = self.downsampler(token_ids)  # (batch, alignment_out_dim, T)
        z_aligned = torch.cat([z, a_down], dim=1)
        return z_aligned

#######################################################################
# 3. PeRFlow: Ускорение решения ОДУ (Piecewise Error-Rectified Flow)
#######################################################################

def sigma(t):
    """
    Функция расписания шума sigma(t).
    
    Здесь задаётся простая линейная функция, но на практике она может быть более сложной.
    
    Аргументы:
      t: время (скаляр или тензор)
      
    Возвращает:
      Значение sigma в момент времени t.
    """
    return 0.1 + 0.9 * t

def phi_theta(z_start, t_start, t_end):
    """
    Симулирует решение ОДУ от t_start до t_end с использованием решателя phi_theta.
    
    В реальном применении phi_theta — это обученная нейросетевая модель, здесь используется простейшая эволюция.
    
    Аргументы:
      z_start: начальное состояние в момент t_start.
      t_start: начальное время.
      t_end: конечное время.
      
    Возвращает:
      Эволюционированное состояние z в момент t_end.
    """
    dt = t_end - t_start
    # Линейная эволюция с добавлением небольшого шума:
    return z_start + dt * 0.5 + torch.randn_like(z_start) * 0.01

class VHat(nn.Module):
    def __init__(self, channel_dim):
        """
        Модель для оценки дрейфовой силы v_{\hat{\theta}}.
        
        Используется для обучения модели-ученика с целью предсказания изменения латентного представления.
        
        Аргументы:
          channel_dim: размерность входного канала (например, размер латентного представления).
        """
        super(VHat, self).__init__()
        self.fc = nn.Linear(channel_dim, channel_dim)
    
    def forward(self, z_t, t):
        """
        Прямой проход модели оценки.
        
        Аргументы:
          z_t: латентное представление в момент времени t.
          t: время (может быть включено в виде дополнительной информации; здесь не используется явно).
          
        Возвращает:
          Предсказанную дрейфовую силу.
        """
        return self.fc(z_t)

def perflow_loss(v_hat, z_t, z_tk, z_tk_minus1, t, t_k, t_k_minus1):
    """
    Вычисляет целевую функцию для обучения PeRFlow.
    
    Целевая функция:
      ℓ = || v_{\hat{θ}}(z_t, t) - (z_{t_k} - z_{t_{k-1}}) / (t_k - t_{k-1}) ||²
    
    Аргументы:
      v_hat: модель оценки дрейфовой силы v_{\hat{θ}}.
      z_t: латентное представление, выбранное в момент времени t ∈ (t_k_minus1, t_k].
      z_tk: конечное состояние окна, полученное решателем phi_theta.
      z_tk_minus1: начальное состояние окна.
      t: выбранное время в интервале (t_k_minus1, t_k].
      t_k, t_k_minus1: границы выбранного временного окна.
      
    Возвращает:
      Значение потерь (MSE) для данного временного окна.
    """
    dt = t_k - t_k_minus1
    target = (z_tk - z_tk_minus1) / dt
    prediction = v_hat(z_t, t)
    loss = torch.mean((prediction - target) ** 2)
    return loss

#######################################################################
# 4. Многокомпонентное безклассификаторное управление (Classifier-Free Guidance, CFG)
#######################################################################

def g_theta(z_t, p, z_prompt):
    """
    Пример функции генерации g_θ.
    
    В реальной задаче эта функция представляет собой сложную нейросеть, генерирующую выход на основе:
      - Латентного представления z_t.
      - Условного входа для фонем (p).
      - Условного запроса диктора (z_prompt).
    
    Если какое-либо условие отсутствует, вместо него используется значение 0.
    
    Аргументы:
      z_t: латентное представление.
      p: условие для фонем (например, эмбеддинг).
      z_prompt: условие диктора.
      
    Возвращает:
      Генерированное представление.
    """
    cond_p = p if p is not None else 0
    cond_z = z_prompt if z_prompt is not None else 0
    return z_t + cond_p + cond_z

def multi_component_cfg(z_t, p, z_prompt, alpha_spk, alpha_txt):
    """
    Реализует многокомпонентное безклассификаторное управление.
    
    Формула:
      𝑔̂_θ(z_t, p, z_prompt) = α_spk [g_θ(z_t, p, z_prompt) - g_θ(z_t, p, ∅)]
                                + α_txt [g_θ(z_t, p, ∅) - g_θ(z_t, ∅, ∅)]
                                + g_θ(z_t, ∅, ∅)
    
    Аргументы:
      z_t: латентное представление.
      p: условие для фонем.
      z_prompt: условие диктора.
      alpha_spk: масштаб управления для диктора.
      alpha_txt: масштаб управления для текстового условия.
      
    Возвращает:
      Итоговый управляемый вывод.
    """
    unconditional = g_theta(z_t, None, None)
    g_with_p = g_theta(z_t, p, None)
    g_with_all = g_theta(z_t, p, z_prompt)
    
    guided = (alpha_spk * (g_with_all - g_with_p) +
              alpha_txt * (g_with_p - unconditional) +
              unconditional)
    return guided

#######################################################################
# Основной блок: интеграция всех компонентов
#######################################################################

if __name__ == "__main__":
    ######################################
    # Шаг 1. Разрежённое выравнивание
    ######################################
    # Задаём фонемы и их длительности.
    phonemes = [1, 2, 3]       # Пример: фонемы заданы их индексами.
    durations = [2, 2, 3]      # Длительности каждой фонемы (суммарная длина = 7).
    mask_token_id = 0        # Идентификатор токена маски.
    
    # Генерируем разрежённое выравнивание.
    sparse_alignment = generate_sparse_alignment(phonemes, durations, mask_token_id)
    print("Разрежённое выравнивание:", sparse_alignment)
    
    # Преобразуем последовательность в тензор и добавляем batch размерность: (1, seq_length).
    token_ids_tensor = torch.tensor(sparse_alignment).unsqueeze(0)
    
    ######################################
    # Шаг 2. Интеграция выравнивания с латентным представлением
    ######################################
    batch_size = 1
    latent_channels = 64
    T = sum(durations)  # Ожидаемая длина латентного представления равна сумме длительностей (например, 7).
    
    # Симулируем латентное представление речи z размером (batch, latent_channels, T).
    z = torch.randn(batch_size, latent_channels, T)
    
    # Задаём параметры для модуля выравнивания.
    vocab_size = 100         # Размер словаря фонем.
    embedding_dim = 32       # Размерность эмбеддингов.
    alignment_out_dim = 16   # Число выходных каналов после свёрточного слоя.
    
    # Инициализируем модуль SDiTAlignment.
    sdit_alignment = SDiTAlignment(vocab_size, embedding_dim, alignment_out_dim, target_length=T)
    
    # Объединяем латентное представление с информацией выравнивания.
    aligned_z = sdit_alignment(z, token_ids_tensor)
    print("Форма объединённого представления:", aligned_z.shape)
    
    # Ожидается форма: (batch, latent_channels + alignment_out_dim, T)
    
    ######################################
    # Шаг 3. PeRFlow: Решение ОДУ и вычисление потерь
    ######################################
    # Задаём размерность канала для модели оценки дрейфовой силы.
    channel_dim = latent_channels
    v_hat_model = VHat(channel_dim)
    
    # Определяем границы временного окна: t ∈ [t_k_minus1, t_k]
    t_k_minus1 = 0.3
    t_k = 0.5
    
    # Симулируем начальное латентное представление z1 для данного примера.
    z1 = torch.randn(1, channel_dim)
    epsilon = torch.randn_like(z1)
    
    # Вычисляем начальное состояние в момент t_k_minus1 по формуле:
    # z_{t_{k-1}} = sqrt(1 - sigma(t)^2) * z1 + sigma(t) * epsilon
    t = t_k_minus1
    z_tk_minus1 = ((1 - sigma(t)**2)**0.5 * z1) + sigma(t) * epsilon
    
    # Решаем ОДУ для получения конечного состояния в момент t_k:
    z_tk = phi_theta(z_tk_minus1, t_k_minus1, t_k)
    
    # Выбираем представительное состояние z_t внутри интервала (например, среднее значение):
    z_t = (z_tk_minus1 + z_tk) / 2
    # Для времени выбираем среднее значение из интервала:
    t_random = torch.tensor([[ (t_k_minus1 + t_k) / 2 ]])
    
    # Вычисляем значение потерь PeRFlow:
    loss_value = perflow_loss(v_hat_model, z_t, z_tk, z_tk_minus1, t_random, t_k, t_k_minus1)
    print("PeRFlow loss:", loss_value.item())
    
    ######################################
    # Шаг 4. Многокомпонентное безклассификаторное управление (CFG)
    ######################################
    
    # Для примера используем скалярные значения, имитирующие латентное представление и условия.
    z_t_cfg = 1.0       # Латентное представление (примерно, может быть вектором)
    p_condition = 0.5   # Условие для фонем (например, эмбеддинг)
    z_prompt = 0.8      # Условие диктора
    
    # Задаём масштабы управления для условий.
    alpha_spk = 2.0   # Масштаб управления для диктора
    alpha_txt = 1.5   # Масштаб управления для текстового условия
    
    # Вычисляем итоговый управляемый вывод с помощью многокомпонентного CFG:
    guided_output = multi_component_cfg(z_t_cfg, p_condition, z_prompt, alpha_spk, alpha_txt)
    print("Вывод многокомпонентного CFG:", guided_output)