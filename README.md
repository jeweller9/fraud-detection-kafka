# Real-Time Fraud Detection System

Сервис скоринга фродовых транзакций в реальном времени: транзакции приходят
потоком из Kafka, обрабатываются CatBoost-моделью (только CPU, только inference),
а скор и флаг фрода пишутся обратно в Kafka.

Датасеты соревнования: <https://www.kaggle.com/competitions/teta-ml-1-2025>

## Архитектура
| Сервис | Назначение |
|---|---|
| `interface` | Streamlit UI: загружает CSV формата `test.csv`, генерирует `transaction_id`, отправляет каждую строку JSON-сообщением в топик `transactions` |
| `fraud_detector` | Читает `transactions`, делает препроцессинг, скорит моделью, пишет результат в `scores` |
| `zookeeper`, `kafka` | Брокер сообщений |
| `kafka-setup` | Создаёт топики `transactions` и `scores` при старте |
| `kafka-ui` | Веб-интерфейс для просмотра топиков |

Формат входного сообщения (`transactions`):
```json
{"transaction_id": "<uuid>", "data": {"transaction_time": "...", "amount": 150.5, "...": "..."}}
```

Формат выходного сообщения (`scores`):
```json
{"transaction_id": "<uuid>", "score": 0.995, "fraud_flag": 1}
```
`fraud_flag = 1`, если `score > 0.98` (порог из соревнования, меняется переменной `FRAUD_THRESHOLD`).

## Структура проекта

```
.
├── docker-compose.yaml
├── README.md
├── sample_data/test_sample.csv      # первые 100 строк test.csv для быстрой проверки
├── fraud_detector/
│   ├── app/app.py                   # Kafka consumer/producer
│   ├── src/preprocessing.py         # препроцессинг (логика из соревнования)
│   ├── src/scorer.py                # загрузка модели и скоринг
│   ├── models/my_catboost.cbm       # предобученная модель
│   ├── train_data/                  # сюда положить train.csv (см. ниже)
│   ├── requirements.txt
│   └── Dockerfile
└── interface/
    ├── app.py                       # Streamlit UI
    ├── requirements.txt
    └── Dockerfile
```

## Запуск

Требования: Docker 20.10+ и Docker Compose v2; свободные порты 8080, 8501, 9095, 2181;
для Docker выделено не менее 4 ГБ памяти (`fraud_detector` держит train в памяти, ~0.8 ГБ).

### 1. Клонировать репозиторий
```bash
git clone <URL-вашего-репозитория>
cd <папка-репозитория>
```

### 2. Переложить `train.csv`
Препроцессингу нужен обучающий датасет (категориальные кодировки, mean-encoding,
импутация). 

1. Скачайте `train.csv` со страницы соревнования
   <https://www.kaggle.com/competitions/teta-ml-1-2025/data>.
2. Положите файл в `fraud_detector/train_data/train.csv`.

Без этого файла контейнер `fraud_detector` остановится с понятной ошибкой в логах.

### 3. Собрать и поднять контейнеры
```bash
docker compose up --build -d
docker compose ps
```

Скорость обработки — порядка 30-40 мс на транзакцию (CPU).

Готовность проверяйте по логам:
```bash
docker compose logs -f fraud_detector
```

## Проверка работоспособности

1. Откройте UI: <http://localhost:8501>.
2. Загрузите CSV формата `test.csv` из соревнования (для первого теста лучше
   взять первые 100 строк) и нажмите «Отправить». Готовый файл с первыми 100 строками
   `test.csv` лежит в `sample_data/test_sample.csv`.
3. Откройте Kafka UI: <http://localhost:8080>  Topics:
   - `transactions` — входящие сообщения;
   - `scores` — результаты скоринга (`transaction_id`, `score`, `fraud_flag`).
4. Либо прочитайте результаты из консоли:
```bash
docker compose exec kafka kafka-console-consumer \
  --bootstrap-server kafka:9092 --topic scores --from-beginning --max-messages 5
```
5. Логи сервиса: `docker compose logs fraud_detector` или
   `/app/logs/service.log` внутри контейнера.

## Остановка
```bash
docker compose down        # остановить
docker compose down -v     # остановить и удалить тома
```

## Переменные окружения `fraud_detector`

| Переменная | По умолчанию | Описание |
|---|---|---|
| `KAFKA_BOOTSTRAP_SERVERS` | `kafka:9092` | адрес брокера |
| `KAFKA_TRANSACTIONS_TOPIC` | `transactions` | входной топик |
| `KAFKA_SCORING_TOPIC` | `scores` | выходной топик |
| `FRAUD_THRESHOLD` | `0.98` | порог для `fraud_flag` |
| `TRAIN_DATA_PATH` | `./train_data/train.csv` | путь к train.csv |
| `MODEL_PATH` | `./models/my_catboost.cbm` | путь к модели |
