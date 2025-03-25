





from DyWinT3 import ConformerSpectrogramTransformer

import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from datasets import load_dataset
import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
import math
import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")  # Отключает все предупреждения

# Проверка доступности GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Параметры
batch_size = 8
epochs = 20
learning_rate = 1e-4
cache_dir = "cache"  # Директория для кэша
preprocessed_data_dir = "preprocessed_data"  # Директория для предобработанных данных
models_dir = "models"  # Директория для сохранения моделей

# Создание необходимых директорий
for directory in [cache_dir, preprocessed_data_dir, models_dir]:
    os.makedirs(directory, exist_ok=True)




from torch.optim.lr_scheduler import LambdaLR

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / 
                  float(max(1, num_training_steps - num_warmup_steps)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)










class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, max_items=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        
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
                        mel_spec, _ = self.data_source[idx]
                        length_cache[idx] = mel_spec.shape[-1]
                except Exception as e:
                    print(f"Error getting length for index {idx}: {e}")
                    length_cache[idx] = 0  # Используем 0 как значение по умолчанию при ошибке
            return length_cache[idx]
        
        # Сортируем индексы в пакетах для улучшения производительности
        batch_size_for_sorting = 1000
        for start_idx in range(0, len(self.indices), batch_size_for_sorting):
            end_idx = min(start_idx + batch_size_for_sorting, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            
            # Сортируем текущий пакет
            batch_indices.sort(key=get_length_with_cache)
            
            # Обновляем основной список индексов
            self.indices[start_idx:end_idx] = batch_indices
            
            print(f"Sorted {end_idx}/{len(self.indices)} samples...")
        
        print("Sorting completed.")
        
        # Сохраняем кэш длин для возможного использования в будущем
        with open(os.path.join(cache_dir, "length_cache.pkl"), "wb") as f:
            pickle.dump(length_cache, f)
            
    def __iter__(self):
        buckets = []
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if len(batch_indices) == self.batch_size or not self.drop_last:
                buckets.append(batch_indices)
        return iter(buckets)
    
    def __len__(self):
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
        self.cache_dir = cache_dir
        self.max_samples = max_samples
        self.use_preprocessed = use_preprocessed
        self.preprocessed_file = os.path.join(preprocessed_data_dir, f"common_voice_{lang}_{split}_preprocessed.pkl")
        self.metadata_file = os.path.join(preprocessed_data_dir, f"common_voice_{lang}_{split}_metadata.json")
        
        # Проверка наличия предобработанных данных
        if use_preprocessed and os.path.exists(self.preprocessed_file) and os.path.exists(self.metadata_file):
            print(f"Loading preprocessed data from {self.preprocessed_file}")
            self.load_preprocessed_data()
        else:
            # Загрузка и подготовка данных с нуля
            print(f"Loading dataset from scratch...")
            self.dataset = load_dataset(
                "mozilla-foundation/common_voice_16_1", lang, split=split, trust_remote_code=True)
            
            if self.max_samples:
                print(f"Using {self.max_samples} samples out of {len(self.dataset)}")
                self.effective_length = min(self.max_samples, len(self.dataset))
            else:
                self.effective_length = len(self.dataset)
            
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)

            # Создаём список файлов для кэширования
            self.cached_files = {}
            for i in tqdm.tqdm(range(self.effective_length), desc="Scanning dataset"):
                sample = self.dataset[i]
                path = sample["path"]
                self.cached_files[path] = os.path.join(
                    cache_dir, os.path.basename(path) + ".pt")
                
            # Создание или загрузка словаря
            self.prepare_vocabulary(lang)
            
            # Сохранение метаданных
            self.save_metadata()
    
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
                print("Creating vocabulary file...")
                text = ""
                for i in tqdm.tqdm(range(self.effective_length), desc="Collecting text"):
                    text += self.dataset[i]["sentence"]
                
                with open(vocab_file, "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                print(f"Loading existing vocabulary file: {vocab_file}")
                with open(vocab_file, "r", encoding="utf-8") as f:
                    text = f.read()
            
            # Создаем словарь и маппинги
            self.vocab = sorted(list(set(text)))
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
            self.cached_files = metadata["cached_files"]
        
        print(f"Loaded metadata: {self.effective_length} samples, {self.vocab_size} vocabulary size")
    
    def save_metadata(self):
        """Сохранение метаданных датасета."""
        metadata = {
            "effective_length": self.effective_length,
            "vocab": self.vocab,
            "vocab_size": self.vocab_size,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "cached_files": self.cached_files
        }
        
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to {self.metadata_file}")
    
    def preprocess_dataset(self):
        """Предобработка всего датасета и сохранение результатов."""
        if os.path.exists(self.preprocessed_file):
            print(f"Preprocessed data already exists at {self.preprocessed_file}")
            return
        
        print(f"Preprocessing {self.effective_length} samples...")
        preprocessed_data = []
        
        for idx in tqdm.tqdm(range(self.effective_length), desc="Preprocessing dataset"):
            mel_spec, transcript = self.process_item(idx)
            
            # Преобразуем текст в индексы
            transcript_indices = [self.char_to_idx.get(char, 0) for char in transcript]
            
            # Сохраняем минимальную информацию для экономии памяти
            item_data = {
                "mel_spec_path": self.cached_files.get(self.dataset[idx]["path"], ""),
                "transcript": transcript,
                "transcript_indices": transcript_indices
            }
            preprocessed_data.append(item_data)
        
        # Сохраняем предобработанные данные
        with open(self.preprocessed_file, "wb") as f:
            pickle.dump(preprocessed_data, f)
        
        print(f"Saved preprocessed data to {self.preprocessed_file}")
    
    def process_item(self, idx):
        """Обработка одного элемента датасета."""
        sample = self.dataset[idx]
        audio_path = sample["path"]
        transcript = sample["sentence"]

        # Проверяем, есть ли путь в кэше
        if audio_path in self.cached_files:
            cache_file = self.cached_files[audio_path]
            if os.path.exists(cache_file):
                # Если кэш есть, загружаем MelSpectrogram из него
                mel_spectrogram = torch.load(cache_file)
                return mel_spectrogram, transcript
        else:
            # Если пути нет в кэше, создаем его
            cache_file = os.path.join(
                self.cache_dir, os.path.basename(audio_path) + ".pt")
            self.cached_files[audio_path] = cache_file

        # Если кэша нет или путь новый, создаем MelSpectrogram
        audio_data = sample["audio"]
        # Преобразуем в torch.float32 (вместо torch.float64/double)
        waveform = torch.tensor(audio_data["array"], dtype=torch.float32).unsqueeze(0)
        
        if self.transform:
            mel_spectrogram = self.transform(waveform)
        else:
            mel_spectrogram = waveform

        # Кэшируем результат
        torch.save(mel_spectrogram, cache_file)

        return mel_spectrogram, transcript
        
    def __getitem__(self, idx):
        """
        Получение элемента датасета по индексу.
        Загружает из кэша или обрабатывает аудио, если оно еще не обработано.
        """
        if hasattr(self, 'dataset'):
            # Если датасет загружен напрямую
            return self.process_item(idx)
        else:
            # Если используем предобработанные данные
            with open(self.preprocessed_file, "rb") as f:
                preprocessed_data = pickle.load(f)
            
            item_data = preprocessed_data[idx]
            
            # Загружаем мел-спектрограмму из кэша
            mel_spec_path = item_data["mel_spec_path"]
            if os.path.exists(mel_spec_path):
                mel_spectrogram = torch.load(mel_spec_path)
            else:
                # Если файл не найден, возвращаем ошибку
                raise FileNotFoundError(f"Cached mel spectrogram not found: {mel_spec_path}")
            
            return mel_spectrogram, item_data["transcript"]

    def __len__(self):
        return self.effective_length
        
    def get_audio_length(self, idx):
        """
        Быстрый метод для получения длины аудио без полной обработки.
        Используется для BucketSampler.
        """
        if hasattr(self, 'dataset'):
            sample = self.dataset[idx]
            audio_path = sample["path"]
            
            # Проверяем наличие в кэше
            if audio_path in self.cached_files:
                cache_file = self.cached_files[audio_path]
                if os.path.exists(cache_file):
                    # Загружаем и возвращаем только форму, без копирования всего тензора
                    mel_spectrogram = torch.load(cache_file)
                    return mel_spectrogram.shape[-1]
            
            # Если нет в кэше, получаем длину из аудиомассива
            audio_data = sample["audio"]
            return len(audio_data["array"])
        else:
            # Если используем предобработанные данные
            # В этом случае просто возвращаем приблизительную длину на основе длины текста
            # Это конечно не идеально, но для предварительно обработанных данных это может работать
            with open(self.preprocessed_file, "rb") as f:
                preprocessed_data = pickle.load(f)
            return len(preprocessed_data[idx]["transcript"]) * 10  # Примерная оценка



# Определяем предобработку (MEL)
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=512,  # Увеличенное значение n_fft
    hop_length=160,
    n_mels=128
)




def collate_fn(batch):
    # Сортируем батч по длине последовательностей (в порядке убывания)
    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)
    
    # Разделяем спектрограммы и транскрипции
    mel_specs, transcripts = zip(*batch)
    
    # Длины входных последовательностей (без паддинга)
    input_lengths = torch.tensor([spec.shape[-1] for spec in mel_specs], dtype=torch.long)
    
    # Паддинг спектрограмм до длины самой длинной в батче
    max_length = input_lengths.max().item()
    padded_specs = []
    
    for spec in mel_specs:
        # Определяем, сколько нужно добавить паддинга
        pad_length = max_length - spec.shape[-1]
        # Добавляем паддинг в конец последовательности
        padded_spec = torch.nn.functional.pad(spec, (0, pad_length))
        padded_specs.append(padded_spec)
    
    # Склеиваем в батч
    padded_specs = torch.stack(padded_specs)
    
    return padded_specs, transcripts, input_lengths


import os
import json
import torch
from torch import nn, optim
import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

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
    dataset = CommonVoiceDataset(lang=lang, split=split, cache_dir=cache_dir, transform=transform, max_samples=max_samples)
    
    if load_metadata:
        if load_dataset_metadata(dataset):
            print("Метаданные датасета успешно загружены")
        else:
            print("Метаданные не найдены, создаем новые")
            save_dataset_metadata(dataset)
    
    return dataset

# Функция для обучения модели с валидацией
def train_model(model, train_loader, val_loader, criterion, optimizer, device, 
                epochs=20, patience=5, checkpoint_dir="checkpoints"):
    """
    Обучает модель с ранней остановкой и сохранением чекпоинтов.
    
    Args:
        model: модель для обучения
        train_loader: загрузчик тренировочных данных
        val_loader: загрузчик валидационных данных
        criterion: функция потерь
        optimizer: оптимизатор
        device: устройство (cuda/cpu)
        epochs: максимальное количество эпох
        patience: количество эпох для ранней остановки
        checkpoint_dir: директория для чекпоинтов
    """
    # Создаем директорию для чекпоинтов
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    total_steps = len(train_loader) * epochs  # 625 батчей * 20 эпох
    warmup_steps = int(0.1 * total_steps)  # 10% на разогрев
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    # Для хранения истории обучения
    train_losses = []
    val_losses = []
    
    # Для ранней остановки
    best_val_loss = float('inf')
    no_improvement = 0
    
    # Обучение
    for epoch in range(epochs):
        
        model.train()
        total_train_loss = 0
        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{epochs} (обучение)")):
            mel_specs, transcripts, input_lengths = batch
            mel_specs = mel_specs.to(device)
            
            # Подготовка целевых значений для CTC-loss
            targets, target_lengths = prepare_targets_ctc(transcripts, model.vocab)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            
            optimizer.zero_grad()
            
            # Получение выходных данных модели
            outputs = model(mel_specs)  # Размер [batch_size, seq_length, vocab_size]
            
            # Преобразование выходных данных для CTC-loss
            log_probs = outputs.log_softmax(2)  # log_softmax по измерению vocab_size
            log_probs = log_probs.permute(1, 0, 2)  # [seq_length, batch_size, vocab_size]
            
            # Вычисление потери
            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
            
            # Логирование для отладки каждые 100 батчей
            if batch_idx % 100 == 0:
                print(f"Батч {batch_idx}, Потеря: {loss.item():.4f}")
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Эпоха {epoch + 1}/{epochs} (валидация)"):
                mel_specs, transcripts, input_lengths = batch
                mel_specs = mel_specs.to(device)
                targets, target_lengths = prepare_targets_ctc(transcripts, model.vocab)
                
                outputs = model(mel_specs)
                
                # Преобразование выходных данных для CTC-loss
                log_probs = outputs.log_softmax(2)  # log_softmax по измерению vocab_size
                log_probs = log_probs.permute(1, 0, 2)  # [seq_length, batch_size, vocab_size]
                loss = criterion(log_probs, targets.to(device), input_lengths.to(device), target_lengths.to(device))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Эпоха {epoch + 1}/{epochs}, Потеря на обучении: {avg_train_loss:.4f}, Потеря на валидации: {avg_val_loss:.4f}")
        
        # Сохранение чекпоинта
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)
        
        # Проверка для ранней остановки
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            
            # Сохранение лучшей модели
            best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, best_model_path)
            print(f"Лучшая модель сохранена в {best_model_path}")
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Ранняя остановка на эпохе {epoch + 1}")
                break
        
    # Построение графика обучения
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Потеря на обучении')
    plt.plot(val_losses, label='Потеря на валидации')
    plt.xlabel('Эпохи')
    plt.ylabel('Потеря')
    plt.title('График потерь при обучении')
    plt.legend()
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))
    plt.close()
    
    return model, train_losses, val_losses

# Функция для подготовки целевых меток
# Функция для подготовки целевых значений для CTC-loss
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
                # Добавляем 1 к индексу, так как 0 зарезервирован для blank
                char_idx = vocab.index(char) + 1
                indices.append(char_idx)
        
        target_indices.extend(indices)
        target_lengths[i] = len(indices)
    
    # Преобразуем список индексов в тензор
    targets = torch.tensor(target_indices, dtype=torch.long)
    
    return targets, target_lengths

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
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ресемплинг до нужной частоты, если необходимо
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
    
    # Преобразование в MEL-спектрограмму
    mel_spec = transform(waveform).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(mel_spec)
    
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
        if idx < len(vocab):
            predicted_text += vocab[idx]
        elif idx == len(vocab):
            predicted_text += "<blank>"
    
    return predicted_text

# Функция для запуска полного обучения
def run_training(load_saved_model=False, model_path="checkpoints/best_model.pth"):
    """
    Выполняет полный цикл обучения модели.
    """
    # Инициализация устройства
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используемое устройство: {device}")
    
    # Определяем MEL-трансформацию
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        hop_length=160,
        n_mels=128
    )
    
    # Загрузка тренировочного датасета
    train_dataset = initialize_dataset(load_metadata=True, split="train", transform=mel_transform, max_samples=100)
    
    # Загрузка валидационного датасета
    val_dataset = initialize_dataset(load_metadata=True, split="validation", transform=mel_transform, max_samples=50)
    
    # Создание загрузчиков данных
    train_sampler = BucketSampler(train_dataset, batch_size=16, drop_last=True)
    val_sampler = BucketSampler(val_dataset, batch_size=16, drop_last=False)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=2, collate_fn = collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=2, collate_fn = collate_fn)
    
    # Инициализация модели
    model = ConformerSpectrogramTransformer(
        hid_dim=64, height=128, width=100, window_size=32, 
        overlap_ratio=0.5, vocab_size=train_dataset.vocab_size+1,
        vocab = train_dataset.vocab
    ).to(device)
    
    # Загрузка модели, если необходимо
    start_epoch = 0
    if load_saved_model:
        model, start_epoch = load_model(model, model_path, device)
    
    # Инициализация функции ошибки CTC
    criterion = nn.CTCLoss(blank=model.vocab_size-1, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Обучение модели
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, device,
        epochs=20, patience=5, checkpoint_dir="checkpoints"
    )
    
    # Сохранение финальной модели
    torch.save(model.state_dict(), "final_model.pth")
    print("Финальная модель сохранена в final_model.pth")
    
    return model, train_losses, val_losses

# Пример использования для запуска обучения
if __name__ == "__main__":
    import argparse
    import torchaudio
    
    parser = argparse.ArgumentParser(description="Обучение и предсказание модели для распознавания речи")
    parser.add_argument("--mode", choices=["train", "predict"], default="train", help="Режим работы: train или predict")
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", help="Путь к сохраненной модели")
    parser.add_argument("--audio_path", default = "C:/Users/pniki/Downloads/01_01_01_voina_i_mir.mp3",help="Путь к аудиофайлу для предсказания")
    parser.add_argument("--load_model", default = False, action="store_true", help="Загрузить существующую модель")
    parser.add_argument("--epochs", type=int, default=20, help="Количество эпох обучения")
    parser.add_argument("--patience", type=int, default=5, help="Количество эпох для ранней остановки")
    
    args = parser.parse_args()
    

    
    if args.mode == "train":
        print("Запуск обучения модели...")
        model, train_losses, val_losses = run_training(
            load_saved_model=args.load_model,
            model_path=args.model_path
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
        dataset = initialize_dataset(load_metadata=True, max_samples = 10000)
        vocab = dataset.vocab
        print(vocab)
        
        # Инициализация модели
        model = ConformerSpectrogramTransformer(
            hid_dim=64, height=128, width=100, window_size=32,
            overlap_ratio=0.5, vocab_size=dataset.vocab_size+1,
            vocab = vocab
        ).to(device)
        
        # Загрузка модели
        model, _ = load_model(model, args.model_path, device)
        
        # Выполнение предсказания
        predicted_text = predict(model, args.audio_path, mel_transform, device, vocab)
        
        print(f"Предсказанный текст: {predicted_text}")