from DyWinT5 import ConformerSpectrogramTransformer

import re
import os
import json
import pickle
import warnings
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_

import torchaudio
from datasets import load_dataset
import tqdm


warnings.filterwarnings("ignore")

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Параметры
# batch_size = 4
epochs = 20
learning_rate = 1e-3
n_fft = 512
hop_length = 160
cache_dir = "cache"  # Директория для кэша
preprocessed_data_dir = "preprocessed_data"  # Директория для предобработанных данных
models_dir = "models"  # Директория для сохранения моделей

# Создание необходимых директорий
for directory in [cache_dir, preprocessed_data_dir, models_dir]:
    os.makedirs(directory, exist_ok=True)






def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)




def prepare_targets_ctc(transcripts, vocab):
    """
    Преобразует транскрипты в формат, подходящий для CTC-loss.
    
    Args:
        transcripts: список строк с транскрипциями
        vocab: словарь символов
    
    Returns:
        targets: тензор индексов символов [sum_seq_lengths]
        target_lengths: тензор длин целевых последовательностей [batch_size]
    """
    batch_size = len(transcripts)
    
    # Список индексов символов для каждой транскрипции
    target_indices = []
    # Длины целевых последовательностей
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, text in enumerate(transcripts):
        indices = []
        for char in text:
            if char in vocab:
                
                char_idx = vocab.index(char)
                indices.append(char_idx)
        
        target_indices.extend(indices)
        target_lengths[i] = len(indices)
    
    # Преобразуем список индексов в тензор
    targets = torch.tensor(target_indices, dtype=torch.long)
    
    return targets, target_lengths





class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, max_items=None, cache_dir="cache"):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.cache_dir = cache_dir
        
        # Используем только часть датасета, если указано
        if max_items is not None:
            self.indices = list(range(min(max_items, len(data_source))))
        else:
            self.indices = list(range(len(data_source)))
        
        print(f"Sorting {len(self.indices)} samples...")
        
        # Создаем кэш длин, чтобы не вызывать __getitem__ повторно
        length_cache = {}
        
        # Определяем функцию для получения длины с кэшированием
        def get_length_with_cache(idx):
            if idx not in length_cache:
                # Получаем длину аудио из датасета
                try:
                    if hasattr(self.data_source, 'get_audio_length'):
                        # Используем специальный метод, если он есть
                        length_cache[idx] = self.data_source.get_audio_length(idx)
                    else:
                        # Иначе получаем данные и измеряем длину
                        wave, *_ = self.data_source[idx]  # Получаем аудио волну
                        length_cache[idx] = wave.shape[-1]  # Длина аудио волны
                except Exception as e:
                    print(f"Error getting length for index {idx}: {e}")
                    length_cache[idx] = 0  # Используем 0 как значение по умолчанию при ошибке
            return length_cache[idx]
        
        # Сортируем индексы в пакетах для улучшения производительности
        batch_size_for_sorting = 1000
        for start_idx in range(0, len(self.indices), batch_size_for_sorting):
            end_idx = min(start_idx + batch_size_for_sorting, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Сортируем текущий пакет по длине аудио волны
            batch_indices.sort(key=get_length_with_cache)
            
            # Обновляем основной список индексов
            self.indices[start_idx:end_idx] = batch_indices
            
            print(f"Sorted {end_idx}/{len(self.indices)} samples...")
        
        print("Sorting completed.")
            
    def __iter__(self):
        """
        Итератор по батчам.
        """
        buckets = []
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if len(batch_indices) == self.batch_size or not self.drop_last:
                buckets.append(batch_indices)
        return iter(buckets)
    
    def __len__(self):
        """
        Возвращает количество батчей.
        """
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return math.ceil(len(self.indices) / self.batch_size)
        

class CommonVoiceDataset(Dataset):
    def __init__(self, root=None, lang="ru", split="train", cache_dir="cv_cache", transform=None, max_samples=None, use_preprocessed=True):
        """
        Загружает датасет Common Voice, преобразует в MelSpectrogram и кэширует.

        :param root: Корневой каталог датасета (не используется для Common Voice, но требуется совместимость)
        :param lang: Язык датасета (например, "ru" для русского).
        :param split: Часть датасета (train, test, validation).
        :param cache_dir: Каталог для кэша преобразованных данных.
        :param transform: Трансформации (например, MelSpectrogram).
        :param max_samples: Максимальное количество образцов для загрузки.
        :param use_preprocessed: Использовать предобработанные данные, если доступны.
        """
        self.split = split
        self.transform = transform
        
        self.max_samples = max_samples
        self.use_preprocessed = use_preprocessed
        self.preprocessed_file = os.path.join(preprocessed_data_dir, f"common_voice_{lang}_{split}_preprocessed.pkl")
        self.metadata_file = os.path.join(preprocessed_data_dir, f"common_voice_{lang}_{split}_metadata.json")
        

        # Загрузка и подготовка данных с нуля
        print(f"Loading dataset from scratch...")
        self.dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1", lang, split=split, trust_remote_code=True)
        
        if self.max_samples:
            print(f"Using {min(len(self.dataset),self.max_samples)} samples out of {len(self.dataset)}")
            self.effective_length = min(self.max_samples, len(self.dataset))
        else:
            self.effective_length = len(self.dataset)
        self.prepare_vocabulary(lang)



    
    def prepare_vocabulary(self, lang):
        """Подготовка словаря из текстовых данных."""
        vocab_file = os.path.join(preprocessed_data_dir, f"vocab_{lang}.txt")
        vocab_json = os.path.join(preprocessed_data_dir, f"vocab_{lang}.json")
        
        if os.path.exists(vocab_json):
            # Загрузка существующего словаря
            with open(vocab_json, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
                self.vocab = vocab_data["vocab"]
                self.vocab_size = vocab_data["vocab_size"]
                self.char_to_idx = vocab_data["char_to_idx"]
                self.idx_to_char = vocab_data["idx_to_char"]
            print(f"Loaded vocabulary with {self.vocab_size} characters")
        else:
            # Создание нового словаря
            if not os.path.exists(vocab_file):
                print("Создание нового словаря...")

                text = ""
                for i in tqdm.tqdm(range(self.effective_length), desc="Collecting text"):
                    text += self.dataset[i]["sentence"]
                
                text = re.sub(r'[^\w\s]', '', text)
                text = text.lower()
                with open(vocab_file, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                print(f"Загрузка словаря: {vocab_file}")
                with open(vocab_file, "r", encoding="utf-8") as f:
                    text = f.read()
            
            # Создаем словарь и маппинги
            self.vocab = ["<blank>"]+sorted(list(set(text))) 
            self.vocab_size = len(self.vocab)
            self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
            self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
            
            # Сохраняем словарь в JSON
            vocab_data = {
                "vocab": self.vocab,
                "vocab_size": self.vocab_size,
                "char_to_idx": self.char_to_idx,
                "idx_to_char": self.idx_to_char
            }
            with open(vocab_json, "w", encoding="utf-8") as f:
                json.dump(vocab_data, f, ensure_ascii=False, indent=2)
            
            print(f"Created vocabulary with {self.vocab_size} characters")
    
    def load_preprocessed_data(self):
        """Загрузка предобработанных данных из файла."""
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
            self.effective_length = metadata["effective_length"]
            self.vocab = metadata["vocab"]
            self.vocab_size = metadata["vocab_size"]
            self.char_to_idx = metadata["char_to_idx"]
            self.idx_to_char = metadata["idx_to_char"]
        
        print(f"Loaded metadata: {self.effective_length} samples, {self.vocab_size} vocabulary size")
    
    def save_metadata(self):
        """Сохранение метаданных датасета."""
        metadata = {
            "effective_length": self.effective_length,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char
        }
        
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to {self.metadata_file}")
        


    
    
    def __getitem__(self, idx):

        # print(type(temp[1]))
        return torch.tensor(self.dataset[idx]["audio"]["array"]), self.dataset[idx]["sentence"]

    def __len__(self):
        return self.effective_length






def collate_fn(batch):
    # Сортируем батч по длине аудио волн (в порядке убывания)
    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)
    
    # Разделяем аудио волны и транскрипции
    waves, _ = zip(*batch)
    
    # Длины входных последовательностей (без паддинга)
    input_lengths = torch.tensor([wave.shape[-1] for wave in waves], dtype=torch.long)
    
    # Паддинг аудио волн до длины самой длинной в батче
    max_length = input_lengths.max().item()
    padded_waves = []
    
    for wave in waves:
        pad_length = max_length - wave.shape[-1]
        padded_wave = torch.nn.functional.pad(wave, (0, pad_length))
        padded_waves.append(padded_wave)
    
    # Склеиваем в батч
    padded_waves = torch.stack(padded_waves)



    
    return padded_waves, _, (input_lengths - n_fft) // hop_length + 1





# Функция для сохранения метаданных датасета
def save_dataset_metadata(dataset, filename="dataset_metadata.json"):
    """
    Сохраняет метаданные датасета для последующей загрузки без повторной обработки.
    """
    metadata = {
        "vocab": dataset.vocab,
        "vocab_size": dataset.vocab_size,
        "cached_files": dataset.cached_files
    }
    
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print(f"Метаданные датасета сохранены в {filename}")

# Функция для загрузки метаданных датасета
def load_dataset_metadata(dataset, filename="dataset_metadata.json"):
    """
    Загружает метаданные датасета из файла.
    """
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        dataset.vocab = metadata["vocab"]
        dataset.vocab_size = metadata["vocab_size"]
        dataset.cached_files = metadata["cached_files"]
        
        print(f"Метаданные датасета загружены из {filename}")
        return True
    return False

# Модификация класса CommonVoiceDataset для загрузки метаданных
def initialize_dataset(load_metadata=True, lang="ru", split="train", cache_dir="cv_cache", transform=None, max_samples = 100):
    """
    Инициализирует датасет с возможностью загрузки метаданных.
    """
    dataset = CommonVoiceDataset(lang=lang, split=split, cache_dir=cache_dir, max_samples=max_samples)
    
    return dataset

# Функция для обучения модели с валидацией
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=20, patience=100, checkpoint_dir="checkpoints", vocab=None,
                grad_clip_norm=1.0):
    """
    Обучает модель с ранней остановкой, сохранением чекпоинтов и подробным логированием статистики.
    (Без использования смешанной точности)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Планировщик скорости обучения (например, по снижению при отсутствии улучшения)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, verbose=True, min_lr=1e-8, cooldown=3
    )

    train_losses = []
    val_losses = []
    grad_norms_history = []
    weight_stats_history = []
    
    best_val_loss = float('inf')
    no_improvement = 0
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Общее число параметров:", pytorch_total_params)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        epoch_grad_norms = []  # собираем нормы градиентов по батчам
        epoch_weight_stats = {}  # статистика весов по слоям
        ba_losses = []
        
        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Эпоха {epoch+1}/{epochs} (обучение)")):
            wave, sent, input_lengths = batch
            targets, target_lengths = prepare_targets_ctc(sent, vocab)
            wave = wave.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            # --- Без автокаста и масштабирования градиентов ---
            outputs = model(wave.float())
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            loss = criterion(log_probs, targets, 
                             torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.int), 
                             target_lengths)
            
            loss.backward()
            threshold = 200
            if loss.item() > threshold:
                print(f"Проблемный батч {batch_idx}, лосс: {loss.item()}")

            # Градиентное клиппирование
            total_norm = clip_grad_norm_(model.parameters(), grad_clip_norm)
            epoch_grad_norms.append(total_norm.item())
            
            optimizer.step()
            
            # Проверка на NaN в градиентах
            for name, param in model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN в градиенте у {name}")
                    param.grad = torch.zeros_like(param.grad)
            
            total_train_loss += loss.item()
            ba_losses.append(loss.item())
            
            # Сбор статистики по весам и градиентам
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param_norm = param.data.norm(2).item()
                        grad_norm = param.grad.data.norm(2).item()
                        if name not in epoch_weight_stats:
                            epoch_weight_stats[name] = {'param_norms': [], 'grad_norms': []}
                        epoch_weight_stats[name]['param_norms'].append(param_norm)
                        epoch_weight_stats[name]['grad_norms'].append(grad_norm)
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        avg_grad_norm = sum(epoch_grad_norms) / len(epoch_grad_norms)
        
        # Построение графика потерь (опционально)
        plt.figure(figsize=(10, 6))
        plt.plot(ba_losses, label='Потеря на обучении')
        plt.xlabel('Шаги обучения')
        plt.ylabel('Потеря')
        plt.title('График потерь при обучении')
        plt.legend()
        plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))
        plt.close()
    
        # Усреднение статистики весов за эпоху
        weight_stats_summary = {}
        for name, stats in epoch_weight_stats.items():
            weight_stats_summary[name] = {
                'param_norm_mean': sum(stats['param_norms']) / len(stats['param_norms']),
                'grad_norm_mean': sum(stats['grad_norms']) / len(stats['grad_norms']),
                'grad_norm_max': max(stats['grad_norms']),
                'grad_norm_min': min(stats['grad_norms']),
            }
        weight_stats_history.append(weight_stats_summary)
        grad_norms_history.append(avg_grad_norm)
        
        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Эпоха {epoch+1}/{epochs} (валидация)"):
                wave, sent, input_lengths = batch
                targets, target_lengths = prepare_targets_ctc(sent, vocab)
                wave = wave.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                
                outputs = model(wave.float())
                log_probs = outputs.log_softmax(2).permute(1, 0, 2)
                loss = criterion(log_probs, targets, 
                                 torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.int), 
                                 target_lengths)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Эпоха {epoch+1}/{epochs}:")
        print(f"  Потеря на обучении: {avg_train_loss:.4f}")
        print(f"  Потеря на валидации: {avg_val_loss:.4f}")
        print(f"  Средняя норма градиентов: {avg_grad_norm:.4f}")
        print(f"  Текущая скорость обучения: {current_lr:.6f}")
        print("  Статистика весов (средние значения):")
        for name, stats in weight_stats_summary.items():
            print(f"    {name}: param_norm_mean = {stats['param_norm_mean']:.4f}, "
                  f"grad_norm_mean = {stats['grad_norm_mean']:.4f}, "
                  f"grad_norm_max = {stats['grad_norm_max']:.4f}, "
                  f"grad_norm_min = {stats['grad_norm_min']:.4f}")
        
        # Сохранение чекпоинта
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'avg_grad_norm': avg_grad_norm,
            'weight_stats': weight_stats_summary,
        }, checkpoint_path)
        
        # Сохранение лучшей модели
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'avg_grad_norm': avg_grad_norm,
                'weight_stats': weight_stats_summary,
            }, best_model_path)
            print(f"  Лучшая модель сохранена в {best_model_path}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

        scheduler.step(avg_train_loss)
    
    # Сохранение итоговых графиков
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Потеря на обучении')
    plt.plot(val_losses, label='Потеря на валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Потеря')
    plt.title('График потерь при обучении')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))
    plt.close()
    
    torch.save({
        'train_losses': train_losses,
        'val_losses': val_losses,
        'grad_norms_history': grad_norms_history,
        'weight_stats_history': weight_stats_history,
    }, os.path.join(checkpoint_dir, 'training_stats.pth'))
    
    return model, train_losses, val_losses

# Функция для подготовки целевых значений для CTC-loss


# Функция для загрузки модели
def load_model(model, model_path, device):
    """
    Загружает модель из файла.
    """
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Модель загружена из {model_path}")
        return model, checkpoint.get('epoch', 0)
    else:
        print(f"Файл модели {model_path} не найден")
        return model, 0

# Функция для предсказания
def predict(model, audio_path, transform, device, vocab):
    """
    Делает предсказание для аудиофайла.
    
    Args:
        model: модель для предсказания
        audio_path: путь к аудиофайлу
        transform: трансформация MEL-спектрограммы
        device: устройство (cuda/cpu)
        vocab: словарь символов
    """
    model.eval()
    
    # Загрузка аудиофайла
    waveform, sr = torchaudio.load(audio_path)
    
    # Ресемплинг до нужной частоты, если необходимо
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)
    
    # Преобразование в MEL-спектрограмму
    # mel_spec = transform(waveform).unsqueeze(0)
    print(waveform.shape)
    # Предсказание
    with torch.no_grad():
        outputs = model(waveform[:,:16000*200].to(device))
    
    # Декодирование результатов (зависит от вашей задачи)
    predicted_text = decode_prediction(outputs, vocab)
    
    return predicted_text

# Функция для декодирования предсказания
def decode_prediction(outputs, vocab):
    """
    Декодирует предсказание модели обратно в текст.
    """
    # Пример: выбор символов с максимальной вероятностью
    # Это зависит от вашей конкретной модели и задачи
    predicted_indices = torch.argmax(outputs[0], dim=1)
    
    predicted_text = ""

    for idx in predicted_indices:
        
        # print(idx)
        if idx != 0:
            if idx < len(vocab):
                predicted_text += vocab[idx]
    
    return predicted_text

# Функция для запуска полного обучения
def run_training(sample_rate, n_fft, hop_length, n_mels, hid_dim, window_size, overlap_ratio, load_saved_model=False, model_path="checkpoints/best_model.pth", epochs = 20, learning_rate = 1):
    """
    Выполняет полный цикл обучения модели.
    """
    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    

    
    # Загрузка тренировочного датасета
    train_dataset = initialize_dataset(load_metadata=True, split="train", max_samples=100000000)
    
    # Загрузка валидационного датасета
    val_dataset = initialize_dataset(load_metadata=True, split="validation", max_samples=5000000)
    
    
    # Создание загрузчиков данных
    if not os.path.exists("train_busa.pth"):
        train_sampler = BucketSampler(train_dataset, batch_size=batch_size, drop_last=True)
        torch.save(train_sampler, "train_busa.pth")
    else:
        train_sampler = torch.load("train_busa.pth")
        
    
    if not os.path.exists("val_busa.pth"):
        val_sampler = BucketSampler(val_dataset, batch_size=batch_size, drop_last=False)
        torch.save(val_sampler, "val_busa.pth")
    else:
        val_sampler = torch.load("val_busa.pth")
    
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, collate_fn = collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2, collate_fn = collate_fn)

    # train_loader = [(i[0].to(device),prepare_targets_ctc(i[1], train_dataset.vocab)[0].to(device),prepare_targets_ctc(i[1], train_dataset.vocab)[1].to(device), i[2].to(device)) for i in train_loader]
    # val_loader = [(i[0].to(device),prepare_targets_ctc(i[1], train_dataset.vocab)[0].to(device),prepare_targets_ctc(i[1], train_dataset.vocab)[1].to(device), i[2].to(device)) for i in val_loader]
    
    
    # Инициализация модели
    model = ConformerSpectrogramTransformer(sample_rate, n_fft, hop_length, n_mels, n_mfcc, hid_dim, window_size, overlap_ratio, train_dataset.vocab_size).to(device)
    
    # Загрузка модели, если необходимо
    start_epoch = 0
    if load_saved_model:
        model, start_epoch = load_model(model, model_path, device)
    
    # Инициализация функции ошибки CTC
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    # Обучение модели
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs=epochs, patience=5, checkpoint_dir="checkpoints", vocab = train_dataset.vocab
    )
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), "final_model.pth")
    print("Финальная модель сохранена в final_model.pth")
    
    return model, train_losses, val_losses

# Пример использования для запуска обучения
if __name__ == "__main__":
    
    # Параметры
    batch_size = 8
    epochs = 200
    learning_rate = 1e-5
    n_fft = 512
    hop_length = 160
    hid_dim = 32
    n_mels = 512
    window_size = 100
    sample_rate = 16000
    overlap_ratio = 0.5
    n_mfcc = 512
    
    
    import argparse

    parser = argparse.ArgumentParser(description="Обучение и предсказание модели для распознавания речи")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Режим работы: train или predict")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", help="Путь к сохраненной модели")
    parser.add_argument("--audio_path", default = "assets/sample.mp3",help="Путь к аудиофайлу для предсказания")
    parser.add_argument("--load_model", default = True, action="store_true", help="Загрузить существующую модель")
    parser.add_argument("--epochs", type=int, default=200, help="Количество эпох обучения")
    parser.add_argument("--patience", type=int, default=5, help="Количество эпох для ранней остановки")
    
    args = parser.parse_args()


    
    if args.mode == "train":
        print("Запуск обучения модели...")
        model, train_losses, val_losses = run_training(
            sample_rate, n_fft, hop_length, n_mels,
            hid_dim, window_size, overlap_ratio,
            load_saved_model=args.load_model,
            model_path=args.model_path,
            epochs=args.epochs,
            learning_rate = learning_rate
        )
        print("Обучение завершено.")
    
    elif args.mode == "predict":
        if not args.audio_path:
            print("Ошибка: для режима predict необходимо указать путь к аудиофайлу (--audio_path)")
            exit(1)
        
        print(f"Выполнение предсказания для файла {args.audio_path}...")
        
        # Инициализация устройства
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Загружаем датасет только для получения словаря
        dataset = initialize_dataset(load_metadata=True, max_samples = 1000000)
        vocab = dataset.vocab
        print(vocab)

        # Инициализация модели
        model = ConformerSpectrogramTransformer(sample_rate, n_fft, hop_length, n_mels, n_mfcc, hid_dim, window_size, overlap_ratio, dataset.vocab_size).to(device)
        
        # Загрузка модели
        model, _ = load_model(model, args.model_path, device)
        
        # Выполнение предсказания
        predicted_text = predict(model, args.audio_path, lambda x:x, device, vocab)
        
        print(f"Предсказанный текст: {predicted_text}")