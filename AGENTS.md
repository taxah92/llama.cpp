# AGENTS.md

> Справочный документ для AI-агентов и инженеров: описание инфраструктуры сервера, проекта и архитектурных модификаций форка.

---

## 1. Инфраструктура и сервер

| Параметр | Значение |
| :--- | :--- |
| **Хост** | `llm` (`ssh llm`, IP: `192.168.2.253`) |
| **Пользователь** | `taxah` (доступ к `sudo` без пароля) |
| **GPU** | 1x NVIDIA Tesla V100-SXM2-16GB (Архитектура Volta, Compute Capability `sm_70`) |
| **VRAM** | 16 144 MiB (16 384 MiB физической) |
| **RAM** | 32 GB |
| **CUDA Driver** | 580.178.04, CUDA Toolkit 13.0 |
| **Путь к сборке** | `/home/taxah/build/llama.cpp` |
| **Рабочая директория сервиса** | `/opt/llama-prism-mtp` |
| **Путь к весам моделей** | `/opt/models/bonsai-2-27b/` |

---

## 2. Обзор проекта

- **Модель:** `Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf` (тернарное квантование Prism ML, 2.13 bpw, архитектура `qwen35`, Hadamard-folded, 65 слоев + 1 слой MTP).
- **Спекулятивное декодирование:** MTP (Multi-Token Prediction, 2 драфт-токена за шаг).
- **Целевой контекст:** 262 144 токена (256K).
- **KV-кэш:** 4-битный (`Q4_0`) с калибровкой центрирования средних (`--kv-mean-center`).
- **Форк llama.cpp:** [taxah92/llama.cpp](https://github.com/taxah92/llama.cpp) (форк от [PrismML-Eng/llama.cpp](https://github.com/PrismML-Eng/llama.cpp)).
- **Источник вдохновения и референс:** [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) — инженерный форк vLLM с оптимизациями низкобитного KV-кэша и внимания под архитектуру NVIDIA Volta (SM70, Tesla V100).

---

## 3. Внесенные изменения (Diff vs PrismML Upstream)

### 3.1. Исправление Shared Memory Flash Attention на Volta SM70 (`ggml-cuda/fattn-mma-f16.cuh`)
- **Проблема:** На архитектуре Volta (SM70) $cols\_per\_warp = 32$ (в отличие от 16 на Ampere+). При дефолтном значении $nbatch\_combine = 128$ объем разделяемой памяти на блок составлял $8 \times 32 \times (128+4) \times 4 = 135\text{ КБ}$, что превышало физический предел Volta (96 КБ) и вызывало фатальный сбой `cudaErrorInvalidValue` при старте Flash Attention.
- **Решение:** Добавлены специализированные конфигурации для $D_{KQ}=256, D_V=256$ и $D_{KQ}=320, D_V=256$ с $nbatch\_combine = 64$ (69 КБ $\le$ 96 КБ opt-in лимита) и $nbatch\_fa = 32$ (35 КБ $\le$ 48 КБ базового предела).

### 3.2. Векторный диспетчер Flash Attention для квантованного KV (`ggml-cuda/fattn.cu`)
- **Проблема:** Функция `ggml_cuda_get_best_fattn_kernel` на Volta умножала длину запроса на `gqa_ratio_eff`. Для шагов верификации спекулятивных драфтов ($Q \le 4$) это приводило к выбору плиточного MMA-ядра, требующего полного деквантования многогигабайтного KV-кэша в FP16 в VRAM.
- **Решение:** Для квантованных типов (`Q4_0`, `Q8_0`) при $Q_{ne[1]} \le 4$ принудительно выбирается `BEST_FATTN_KERNEL_VEC`, производящий вычисления прямо по квантованному кэшу без деквантования.

### 3.3. Устранение коллапса драфтов MTP (`common/speculative.cpp`)
- **Проблема:** Попытка оффлоада сэмплирования на GPU (`backend_sampling`) обходила процессорную подготовку массива кандидатов (`cur_p`), оставляя вероятности токенов равными $0.0\text{f}$. Функция `draft()` читала несортированный массив и на каждом шаге выдавала мусорный токен `165552 ("ansir")` со 100% отсевом драфтов (`draft acceptance = 0.00000`).
- **Решение:** Для MTP принудительно зафиксирован CPU-сэмплер (`this->params.backend_sampling = false`) с детерминированным argmax (`sparams.top_k = 1`). Принятие драфтов восстановилось до **50% – 100% (среднее ~61.4%)**, а скорость выросла с 25 до **52–70.5 токенов/сек**.

---

## 4. Окружение и флаги запуска

### Обязательный флаг `LLAMA_ATTN_ROT_DISABLE=1`
Файл смещений `/opt/models/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-kv-bias.gguf` откалиброван без вращения K-кэша (`kv_mean_center.k_rot = false`). Без флага `LLAMA_ATTN_ROT_DISABLE=1` инициализация прерывается из-за несовпадения базиса.

### Системный сервис `bonsai-server.service`
Конфигурация расположена в `/etc/systemd/system/bonsai-server.service`:
```ini
[Unit]
Description=Prism ML Bonsai 2 27B Abliterated LLM Server (256K context, Q4_0 KV, Flash Attention, Volta SM70)
After=network.target nvidia-persistenced.service

[Service]
Type=simple
User=taxah
Group=taxah
WorkingDirectory=/opt/llama-prism-mtp
Environment=LD_LIBRARY_PATH=/opt/llama-prism-mtp LLAMA_ATTN_ROT_DISABLE=1
ExecStart=/opt/llama-prism-mtp/llama-server \
  -m /opt/models/bonsai-2-27b/Ternary-Bonsai-2-27B-Abliterated-PQ2_0-MTP.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 99 \
  --ctx-size 262144 \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --kv-mean-center /opt/models/bonsai-2-27b/Ternary-Bonsai-2-27B-PQ2_0-kv-bias.gguf \
  --flash-attn on \
  --parallel 1 \
  -b 2048 \
  -ub 512 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  --jinja \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --repeat-penalty 1.0 \
  --presence-penalty 0.0 \
  --reasoning-preserve
Restart=always
RestartSec=5
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
```

---

## 5. Работа с мультимодальным проектором (Vision / `mmproj`)

Файлы в `/opt/models/bonsai-2-27b/`:
- `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` (601 МБ)
- `Ternary-Bonsai-2-27B-mmproj-BF16.gguf` (889 МБ)

### Режимы использования на 16 ГБ VRAM:
1. **Режим максимального контекста (256K) + CPU Vision:**
   - Аргументы: `--ctx-size 262144 --mmproj ... --no-mmproj-offload`
   - Веса проектора и ViT живут в системной RAM (28 ГБ свободно).
   - Обработка картинки 512×512 на CPU занимает ~4.1 сек, генерация ответа на GPU идет со скоростью ~63 ток/с.
2. **Режим полной скорости GPU (быстрое распознавание фото):**
   - Аргументы: `--ctx-size 220000 --mmproj ...` (дефолтный GPU-оффлоад)
   - Контекст 220K освобождает ~1.57 ГБ VRAM для активаций ViT.
   - Картинка 512×512 кодируется на GPU за **1.13 сек** (префилл **281 ток/с**).

---

## 6. Базовые команды управления

```bash
# Проверка статуса сервиса
ssh llm "sudo systemctl status bonsai-server.service --no-pager"

# Просмотр логов
ssh llm "journalctl -u bonsai-server -f"

# Перезапуск сервиса
ssh llm "sudo systemctl restart bonsai-server.service"

# Проверка VRAM
ssh llm "nvidia-smi"

# Тестовый запрос
curl -s http://192.168.2.253:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```
