import re
import csv
import warnings
import math
import numpy as np
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils import clip_grad_norm_
import torchaudio
import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf

from model import ConformerSpectrogramTransformer

warnings.filterwarnings("ignore")

def prepare_targets_ctc(transcripts, vocab):
    batch_size = len(transcripts)
    target_indices = []
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, text in enumerate(transcripts):
        indices = []
        for char in text:
            if char in vocab:
                char_idx = vocab.index(char)
                indices.append(char_idx)
        
        target_indices.extend(indices)
        target_lengths[i] = len(indices)
    
    targets = torch.tensor(target_indices, dtype=torch.long)
    return targets, target_lengths

class BucketSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last, max_items=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if max_items is not None:
            self.indices = list(range(min(max_items, len(data_source))))
        else:
            self.indices = list(range(len(data_source)))
        
        print(f"Sorting {len(self.indices)} samples...")
        length_cache = {}
        
        def get_length_with_cache(idx):
            if idx not in length_cache:
                try:
                    wave, *_ = self.data_source[idx]
                    length_cache[idx] = wave.shape[-1]
                except Exception as e:
                    length_cache[idx] = 0
            return length_cache[idx]
        
        batch_size_for_sorting = 1000
        for start_idx in range(0, len(self.indices), batch_size_for_sorting):
            end_idx = min(start_idx + batch_size_for_sorting, len(self.indices))
            batch_indices = self.indices[start_idx:end_idx]
            batch_indices.sort(key=get_length_with_cache)
            self.indices[start_idx:end_idx] = batch_indices
            print(f"Sorted {end_idx}/{len(self.indices)} samples...")
        
        print("Sorting completed.")
            
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

# --- Чистый локальный датасет без зависимости от datasets и pyarrow ---
class CommonVoiceDataset(Dataset):
    def __init__(self, audio_dir, tsv_path, max_samples=None, sample_rate=16000):
        self.audio_dir = Path(audio_dir)
        self.tsv_path = Path(tsv_path)
        self.sample_rate = sample_rate
        
        self.samples = []
        
        # Читаем разметку встроенной библиотекой csv (без pandas и pyarrow)
        if self.tsv_path.exists():
            with open(self.tsv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    path = self.audio_dir / row["path"]
                    sentence = row["sentence"]
                    if path.exists():
                        self.samples.append((path, sentence))
                        if max_samples and len(self.samples) >= max_samples:
                            break
        else:
            print(f"[!] Предупреждение: Файл разметки {tsv_path} не найден.")
            
        self.prepare_vocabulary()

    def prepare_vocabulary(self):
        text = ""
        for _, sentence in self.samples:
            text += sentence
        text = re.sub(r'[^\w\\s]', '', text).lower()
        self.vocab = ["<blank>"] + sorted(list(set(text)))
        self.vocab_size = len(self.vocab)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sentence = self.samples[idx]
        wave, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            wave = resampler(wave)
        return wave.squeeze(0), sentence

# --- Синтетический датасет для демонстрации ---
class MockAudioDataset(Dataset):
    def __init__(self, num_samples=16):
        self.num_samples = num_samples
        self.vocab = ["<blank>"] + [chr(i) for i in range(ord('а'), ord('я') + 1)] + [" "]
        self.vocab_size = len(self.vocab)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        wave = torch.randn(80000)
        sentence = "".join([np.random.choice(self.vocab[1:]) for _ in range(np.random.randint(10, 30))])
        return wave, sentence

def collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[-1], reverse=True)
    waves, transcripts = zip(*batch)
    
    input_lengths = torch.tensor([wave.shape[-1] for wave in waves], dtype=torch.long)
    max_length = input_lengths.max().item()
    padded_waves = []
    
    for wave in waves:
        pad_length = max_length - wave.shape[-1]
        padded_wave = torch.nn.functional.pad(wave, (0, pad_length))
        padded_waves.append(padded_wave)
    
    padded_waves = torch.stack(padded_waves)
    
    # 512 = n_fft, 160 = hop_length
    spectrogram_lengths = (input_lengths - 512) // 160 + 1
    return padded_waves, transcripts, spectrogram_lengths

def train_model(model, train_loader, val_loader, criterion, optimizer, device, config):
    checkpoint_dir = Path(config.get('save_dir', './checkpoints'))
    checkpoint_dir.mkdir(exist_ok=True)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=config.get('patience', 2)
    )

    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    no_improvement = 0
    avg_train_loss = float("inf")
    num_epochs = config.get('num_epochs', 3)
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Эпоха {epoch+1}/{num_epochs} (обучение)")):
            wave, sent, input_lengths = batch
            targets, target_lengths = prepare_targets_ctc(sent, config['vocab'])
            wave, targets, target_lengths = wave.to(device), targets.to(device), target_lengths.to(device)
            
            optimizer.zero_grad()
            outputs = model(wave.float())
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)
            
            loss = criterion(log_probs, targets, 
                             torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.int), 
                             target_lengths)
            
            if loss.item() < avg_train_loss * 10:
                loss.backward()
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_train_loss += loss.item()
            else:
                total_train_loss += avg_train_loss
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Валидация
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm.tqdm(val_loader, desc=f"Валидация"):
                wave, sent, _ = batch
                targets, target_lengths = prepare_targets_ctc(sent, config['vocab'])
                wave, targets, target_lengths = wave.to(device), targets.to(device), target_lengths.to(device)
                
                outputs = model(wave.float())
                log_probs = outputs.log_softmax(2).permute(1, 0, 2)
                loss = criterion(log_probs, targets, 
                                 torch.full((log_probs.shape[1],), log_probs.shape[0], dtype=torch.int), 
                                 target_lengths)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Эпоха {epoch+1}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1}, checkpoint_dir / "best_model.pth")
        else:
            no_improvement += 1
            if no_improvement >= config.get('patience', 2):
                print("Ранняя остановка!")
                break
                
        scheduler.step(avg_val_loss)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main_advanced_training(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = config.get('batch_size', 4)
    window_size_ms = config.get('window_size_ms', 100)
    learning_rate = config.get('learning_rate', 1e-4)
    
    dataset_cfg = config.get('dataset', {})
    if dataset_cfg.get('mock', True):
        print("[!] ВНИМАНИЕ: Запуск в MOCK-режиме (синтетический звук в ОЗУ)")
        train_dataset = MockAudioDataset(num_samples=16)
        val_dataset = MockAudioDataset(num_samples=4)
    else:
        print("Инициализация локального датасета Common Voice...")
        train_dataset = CommonVoiceDataset(
            audio_dir=dataset_cfg.get('audio_dir', 'data/clips'), 
            tsv_path=dataset_cfg.get('tsv_path', 'data/train.tsv'),
            max_samples=1000
        )
        val_dataset = CommonVoiceDataset(
            audio_dir=dataset_cfg.get('audio_dir', 'data/clips'), 
            tsv_path=dataset_cfg.get('tsv_path', 'data/val.tsv'),
            max_samples=50
        )
    
    config['vocab'] = train_dataset.vocab
    
    train_sampler = BucketSampler(train_dataset, batch_size=batch_size, drop_last=True)
    val_sampler = BucketSampler(val_dataset, batch_size=batch_size, drop_last=False)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, collate_fn=collate_fn)
    
    model = ConformerSpectrogramTransformer(
        sample_rate=16000, n_fft=512, hop_length=160, 
        n_mels=512, n_mfcc=512, hid_dim=8, 
        window_size=window_size_ms, 
        overlap_ratio=0.5, vocab_size=train_dataset.vocab_size
    ).to(device)
    
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    train_model(model, train_loader, val_loader, criterion, optimizer, device, config)

if __name__ == "__main__":
    main_advanced_training()