import numpy as np
import torch
import torch.nn as nn
import librosa

# --- 1. Упрощённый feature extractor, имитирующий wav2vec2 ---

class DummyWav2Vec2FeatureExtractor(nn.Module):
    def __init__(self):
        super(DummyWav2Vec2FeatureExtractor, self).__init__()
        # Первый сверточный слой: преобразует сигнал в 64-мерное представление
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=10, stride=5, padding=3)
        self.relu = nn.ReLU()
        # Второй сверточный слой: увеличивает размерность до 128
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, stride=4, padding=2)
        
    def forward(self, x):
        # x: [batch, time]
        x = x.unsqueeze(1)  # добавляем канал: [batch, 1, time]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        # Переставляем размерности: [batch, time, feature_dim]
        x = x.transpose(1, 2)
        return x

# --- 2. PCA ---

def my_pca(X, variance_retained=0.95):
    """
    Реализует метод главных компонент (PCA) для снижения размерности.
    
    X: входная матрица признаков (n_samples, n_features)
    variance_retained: доля дисперсии, которую требуется сохранить (например, 0.95)
    
    Возвращает:
      - X_reduced: данные, спроецированные на выбранные главные компоненты,
      - components: матрица собственных векторов, используемая для проекции.
    """
    # Центрируем данные
    X_centered = X - np.mean(X, axis=0)
    # Вычисляем ковариационную матрицу
    cov_matrix = np.cov(X_centered, rowvar=False)
    # Собственные значения и векторы
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    # Сортируем по убыванию собственных значений
    sorted_idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[sorted_idx]
    eig_vecs = eig_vecs[:, sorted_idx]
    # Вычисляем накопленное отношение дисперсии
    total_variance = np.sum(eig_vals)
    cumulative_variance = np.cumsum(eig_vals)
    ratio = cumulative_variance / total_variance
    # Определяем, сколько компонент нужно сохранить
    n_components = np.searchsorted(ratio, variance_retained) + 1
    components = eig_vecs[:, :n_components]
    # Проецируем данные на выбранные компоненты
    X_reduced = np.dot(X_centered, components)
    return X_reduced, components

# --- 3. Реализация KNN классификатора с нуля ---

def knn_predict(X_train, y_train, X_test, k=10):
    """
    Предсказывает метки для X_test, используя K ближайших соседей.
    
    X_train: обучающие признаки, shape: (n_train, n_features)
    y_train: обучающие метки, shape: (n_train,)
    X_test: тестовые признаки, shape: (n_test, n_features)
    k: число соседей
    
    Возвращает:
      - predictions: предсказанные метки для каждого примера из X_test
      - probabilities: доля голосов за выбранную метку (оценка уверенности)
    """
    predictions = []
    probabilities = []
    for x in X_test:
        # Вычисляем Эвклидовы расстояния от x до всех обучающих примеров
        distances = np.linalg.norm(X_train - x, axis=1)
        # Индексы k ближайших соседей
        nearest_idx = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_idx]
        # Подсчёт голосов
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        max_count = np.max(counts)
        pred_label = unique_labels[np.argmax(counts)]
        predictions.append(pred_label)
        probabilities.append(max_count / k)
    return np.array(predictions), np.array(probabilities)

# --- 4. Группировка последовательных кадров с одинаковыми метками ---

def group_frames(predictions, frame_duration=0.02):
    """
    Группирует последовательные кадры с одинаковыми метками.
    
    predictions: список предсказанных меток для каждого кадра
    frame_duration: длительность одного кадра в секундах (например, 20 мс)
    
    Возвращает список кортежей (метка, время_начала, время_окончания)
    """
    segments = []
    if len(predictions) == 0:
        return segments
    current_label = predictions[0]
    start_idx = 0
    for i in range(1, len(predictions)):
        if predictions[i] != current_label:
            end_idx = i - 1
            start_time = start_idx * frame_duration
            end_time = (end_idx + 1) * frame_duration
            segments.append((current_label, start_time, end_time))
            current_label = predictions[i]
            start_idx = i
    # Добавляем последний сегмент
    end_idx = len(predictions) - 1
    start_time = start_idx * frame_duration
    end_time = (end_idx + 1) * frame_duration
    segments.append((current_label, start_time, end_time))
    return segments



def load_audio(audio_path, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    return audio

# --- Основной конвейер обработки аудио ---

def main(audio_path):
    # 1. Загрузка аудио
    audio = load_audio(audio_path)  # audio: 1D массив
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)  # [1, time]
    
    # 2. Извлечение признаков с помощью dummy feature extractor
    feature_extractor = DummyWav2Vec2FeatureExtractor()
    feature_extractor.eval()
    with torch.no_grad():
        
        
        features = feature_extractor(audio_tensor)  # [1, time', feature_dim]
        print(audio_tensor.shape)
        print(features.shape)
    features = features.squeeze(0).cpu().numpy()  # [time', feature_dim]
    print("Извлечённые признаки shape:", features.shape)
    
    # 3. Снижение размерности с помощью PCA (сохраняем 95% дисперсии)
    reduced_features, _ = my_pca(features, variance_retained=0.95)
    print("Признаки после PCA shape:", reduced_features.shape)
    
    # 4. Классификация: здесь для демонстрации создаём синтетический обучающий набор
    num_train_samples = 1000
    feature_dim = reduced_features.shape[1]
    X_train = np.random.randn(num_train_samples, feature_dim)
    # Допустим, что у нас 39 классов фонем (метки от 0 до 38)
    y_train = np.random.randint(0, 39, size=num_train_samples)
    
    # Предсказание фонем для каждого кадра с помощью KNN (k=10)
    k = 10
    predicted_labels, predicted_probs = knn_predict(X_train, y_train, reduced_features, k=k)
    
    # 5. Группировка последовательных кадров с одинаковыми предсказаниями
    segments = group_frames(predicted_labels, frame_duration=800**-1)
    
    print("Результирующие сегменты фонем:")
    for seg in segments:
        label, start, end = seg
        print(f"Фонема: {label}, начало: {start:.2f} сек, конец: {end:.2f} сек")
    
    return segments

if __name__ == "__main__":
    
    print(1.2334e5)
    audio_path = "sample.mp3"  # Укажите путь к вашему аудиофайлу
    segments = main(audio_path)
    