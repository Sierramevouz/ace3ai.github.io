# API Reference

Complete reference for the ACE3 SDK and REST API.

## Client Initialization

### ace3.Client

The main entry point for ACE3 SDK.

```python
import ace3

client = ace3.Client(
    api_key="your-api-key",
    endpoint="https://api.ace3.ai",
    timeout=30,
    retry_attempts=3
)
```

**Parameters:**
- `api_key` (str): Your ACE3 API key
- `endpoint` (str, optional): API endpoint URL
- `timeout` (int, optional): Request timeout in seconds
- `retry_attempts` (int, optional): Number of retry attempts

## Authentication

### API Key Authentication

All API requests require a valid API key:

```python
# Method 1: Pass directly to client
client = ace3.Client(api_key="your-api-key")

# Method 2: Set environment variable
import os
os.environ['ACE3_API_KEY'] = "your-api-key"
client = ace3.Client()

# Method 3: Use config file
client = ace3.Client.from_config("ace3.config.json")
```

### Token Refresh

API tokens are automatically refreshed:

```python
# Check token status
print(client.token_info())

# Manually refresh token
client.refresh_token()
```

## Model Management

### Loading Models

#### From Local File

```python
# ONNX model
model = client.load_model("model.onnx")

# PyTorch model
model = client.load_model("model.pt", format="pytorch")

# TensorFlow model
model = client.load_model("saved_model/", format="tensorflow")
```

#### From Hugging Face Hub

```python
# Load by model name
model = client.load_model("huggingface:bert-base-uncased")

# Load specific revision
model = client.load_model(
    "huggingface:microsoft/DialoGPT-medium",
    revision="main"
)
```

#### From URL

```python
model = client.load_model("https://example.com/model.onnx")
```

### Model Information

```python
# Get model metadata
info = model.info()
print(f"Model name: {info.name}")
print(f"Input shape: {info.input_shape}")
print(f"Output shape: {info.output_shape}")

# Check if model is optimized
print(f"Optimized: {model.is_optimized()}")

# Get model size
print(f"Model size: {model.size_mb()} MB")
```

### Model Optimization

```python
# Basic optimization
optimized_model = client.optimize(model)

# Specify target
optimized_model = client.optimize(
    model, 
    target="inference",  # or "training"
    level="standard"     # "basic", "standard", "aggressive"
)

# Advanced optimization options
optimized_model = client.optimize(
    model,
    target="inference",
    options={
        "use_fp16": True,
        "batch_size": 32,
        "max_sequence_length": 512,
        "enable_kernel_fusion": True
    }
)
```

## Inference

### Synchronous Inference

```python
import numpy as np

# Single prediction
input_data = np.random.randn(1, 224, 224, 3)
result = model.predict(input_data)

# Batch prediction
batch_data = np.random.randn(8, 224, 224, 3)
results = model.batch_predict(batch_data)

# With custom batch size
results = model.batch_predict(batch_data, batch_size=4)
```

### Asynchronous Inference

```python
import asyncio

async def async_inference():
    # Single async prediction
    result = await model.async_predict(input_data)
    
    # Batch async prediction
    results = await model.async_batch_predict(batch_data)
    
    return results

# Run async inference
results = asyncio.run(async_inference())
```

### Streaming Inference

```python
# For real-time applications
stream = model.create_stream()

for data_chunk in data_stream:
    result = stream.predict(data_chunk)
    process_result(result)

stream.close()
```

## Training

### Distributed Training Setup

```python
# Configure distributed training
config = ace3.DistributedConfig(
    num_gpus=4,
    num_nodes=2,
    backend="nccl",  # or "gloo", "mpi"
    master_addr="localhost",
    master_port=29500
)

# Initialize trainer
trainer = ace3.DistributedTrainer(config)
```

### Training Configuration

```python
# Training parameters
train_config = ace3.TrainingConfig(
    batch_size=32,
    learning_rate=1e-4,
    epochs=10,
    optimizer="adam",
    loss_function="cross_entropy",
    mixed_precision=True,
    gradient_checkpointing=True
)

# Start training
trainer.train(
    model=model,
    dataset=train_dataset,
    config=train_config,
    validation_dataset=val_dataset
)
```

### Training Callbacks

```python
# Built-in callbacks
callbacks = [
    ace3.callbacks.EarlyStopping(patience=5),
    ace3.callbacks.ModelCheckpoint(save_best_only=True),
    ace3.callbacks.ReduceLROnPlateau(factor=0.5),
    ace3.callbacks.TensorBoard(log_dir="./logs")
]

trainer.train(model, dataset, callbacks=callbacks)
```

## Monitoring and Profiling

### Performance Profiling

```python
# Start profiler
profiler = ace3.Profiler()
profiler.start()

# Your code here
result = model.predict(input_data)

# Get profiling results
stats = profiler.stop()
print(stats.summary())

# Save detailed report
stats.save_report("profile_report.html")
```

### Memory Monitoring

```python
# Monitor memory usage
monitor = ace3.MemoryMonitor()
monitor.start()

# Your code here
result = model.predict(input_data)

# Get memory stats
memory_stats = monitor.stop()
print(f"Peak memory: {memory_stats.peak_mb} MB")
print(f"Current memory: {memory_stats.current_mb} MB")
```

### Real-time Metrics

```python
# Enable metrics collection
client.enable_metrics()

# Get real-time metrics
metrics = client.get_metrics()
print(f"Requests per second: {metrics.rps}")
print(f"Average latency: {metrics.avg_latency_ms} ms")
print(f"Error rate: {metrics.error_rate}%")
```

## Error Handling

### Exception Types

```python
try:
    model = client.load_model("invalid_model.onnx")
except ace3.ModelLoadError as e:
    print(f"Failed to load model: {e}")
except ace3.AuthenticationError as e:
    print(f"Authentication failed: {e}")
except ace3.APIError as e:
    print(f"API error: {e}")
except ace3.NetworkError as e:
    print(f"Network error: {e}")
```

### Retry Logic

```python
from ace3.utils import retry

@retry(max_attempts=3, delay=1.0, backoff=2.0)
def robust_inference(model, data):
    return model.predict(data)

# Usage
result = robust_inference(model, input_data)
```

## Utilities

### Data Preprocessing

```python
# Image preprocessing
preprocessor = ace3.ImagePreprocessor(
    resize=(224, 224),
    normalize=True,
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

processed_image = preprocessor(raw_image)

# Text preprocessing
text_processor = ace3.TextPreprocessor(
    tokenizer="bert-base-uncased",
    max_length=512,
    padding=True,
    truncation=True
)

processed_text = text_processor(raw_text)
```

### Model Conversion

```python
# Convert between formats
converter = ace3.ModelConverter()

# PyTorch to ONNX
onnx_model = converter.pytorch_to_onnx(
    pytorch_model,
    input_shape=(1, 3, 224, 224),
    output_path="model.onnx"
)

# TensorFlow to ONNX
onnx_model = converter.tensorflow_to_onnx(
    tf_model,
    output_path="model.onnx"
)
```

### Benchmarking

```python
# Benchmark model performance
benchmark = ace3.Benchmark()

results = benchmark.run(
    model=model,
    input_data=test_data,
    num_runs=100,
    warmup_runs=10
)

print(f"Average latency: {results.avg_latency_ms} ms")
print(f"Throughput: {results.throughput_rps} RPS")
print(f"P95 latency: {results.p95_latency_ms} ms")
```

## REST API

### Base URL

```
https://api.ace3.ai/v1
```

### Authentication

Include your API key in the Authorization header:

```bash
curl -H "Authorization: Bearer your-api-key" \
     https://api.ace3.ai/v1/models
```

### Endpoints

#### List Models

```bash
GET /models
```

Response:
```json
{
  "models": [
    {
      "id": "model-123",
      "name": "my-model",
      "status": "ready",
      "created_at": "2023-01-01T00:00:00Z"
    }
  ]
}
```

#### Upload Model

```bash
POST /models
Content-Type: multipart/form-data

{
  "name": "my-model",
  "file": <model-file>,
  "format": "onnx"
}
```

#### Run Inference

```bash
POST /models/{model_id}/predict
Content-Type: application/json

{
  "inputs": [[1.0, 2.0, 3.0]],
  "options": {
    "batch_size": 1
  }
}
```

Response:
```json
{
  "outputs": [[0.1, 0.9]],
  "latency_ms": 15.2,
  "request_id": "req-456"
}
```

## Rate Limits

API rate limits by plan:

- **Free**: 100 requests/hour
- **Pro**: 10,000 requests/hour  
- **Enterprise**: Unlimited

Rate limit headers:
```
X-RateLimit-Limit: 10000
X-RateLimit-Remaining: 9999
X-RateLimit-Reset: 1640995200
```

## SDK Configuration

### Global Settings

```python
# Set global configuration
ace3.set_config({
    "default_timeout": 30,
    "max_retries": 3,
    "log_level": "INFO",
    "cache_enabled": True,
    "cache_ttl": 3600
})

# Get current configuration
config = ace3.get_config()
print(config)
```

### Logging

```python
import logging

# Enable ACE3 logging
ace3.set_log_level("DEBUG")

# Use custom logger
logger = logging.getLogger("my_app")
ace3.set_logger(logger)
```

