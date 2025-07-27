# SDK References

Complete reference for ACE3 SDKs across different programming languages and platforms.

## Python SDK

### Installation

```bash
# Install from PyPI
pip install ace3-sdk

# Install with optional dependencies
pip install ace3-sdk[all]

# Install development version
pip install git+https://github.com/ace3-ai/ace3-python.git
```

### Quick Start

```python
import ace3

# Initialize client
client = ace3.Client(api_key="your-api-key")

# Load and optimize model
model = client.load_model("path/to/model.onnx")
optimized_model = client.optimize(model, target="inference")

# Run inference
result = optimized_model.predict(input_data)
```

### Core Classes

#### ace3.Client

Main client for interacting with ACE3 services.

```python
class Client:
    def __init__(self, api_key=None, endpoint=None, timeout=30):
        """Initialize ACE3 client"""
        
    def load_model(self, path, format="auto"):
        """Load a model from file or URL"""
        
    def optimize(self, model, target="inference", level="standard", options=None):
        """Optimize a model for specific target"""
        
    def list_models(self):
        """List available models"""
        
    def get_model_info(self, model_id):
        """Get information about a model"""
```

#### ace3.Model

Represents a loaded model.

```python
class Model:
    def predict(self, input_data):
        """Run single prediction"""
        
    def batch_predict(self, batch_data, batch_size=None):
        """Run batch prediction"""
        
    async def async_predict(self, input_data):
        """Run asynchronous prediction"""
        
    def info(self):
        """Get model information"""
        
    def is_optimized(self):
        """Check if model is optimized"""
        
    def save(self, path):
        """Save model to file"""
```

#### ace3.DistributedTrainer

For distributed training across multiple GPUs/nodes.

```python
class DistributedTrainer:
    def __init__(self, config):
        """Initialize distributed trainer"""
        
    def train(self, model, dataset, epochs=10, callbacks=None):
        """Train model with distributed setup"""
        
    def evaluate(self, model, dataset):
        """Evaluate model performance"""
        
    def save_checkpoint(self, path):
        """Save training checkpoint"""
        
    def load_checkpoint(self, path):
        """Load training checkpoint"""
```

### Configuration

#### Environment Variables

```bash
export ACE3_API_KEY="your-api-key"
export ACE3_ENDPOINT="https://api.ace3.ai"
export ACE3_LOG_LEVEL="INFO"
export ACE3_CACHE_DIR="/tmp/ace3_cache"
```

#### Configuration File

Create `ace3.config.json`:

```json
{
  "api_key": "your-api-key",
  "endpoint": "https://api.ace3.ai",
  "timeout": 30,
  "retry_attempts": 3,
  "cache_enabled": true,
  "cache_ttl": 3600,
  "log_level": "INFO"
}
```

Load configuration:

```python
client = ace3.Client.from_config("ace3.config.json")
```

### Advanced Features

#### Custom Operators

```python
@ace3.custom_operator("my_op")
def my_custom_operator(input_tensor, param1, param2):
    # Custom implementation
    return output_tensor

# Register and use
client.register_operator(my_custom_operator)
model = client.optimize(model, custom_operators=["my_op"])
```

#### Model Callbacks

```python
class CustomCallback(ace3.Callback):
    def on_prediction_start(self, inputs):
        print("Starting prediction...")
    
    def on_prediction_end(self, outputs):
        print("Prediction completed!")

model.add_callback(CustomCallback())
```

#### Profiling

```python
# Profile inference
with ace3.profile() as profiler:
    result = model.predict(input_data)

print(profiler.summary())
profiler.save_report("profile.html")
```

## JavaScript SDK

### Installation

```bash
# Install via npm
npm install ace3-sdk

# Install via yarn
yarn add ace3-sdk
```

### Quick Start

```javascript
import ACE3 from 'ace3-sdk';

// Initialize client
const client = new ACE3.Client({
  apiKey: 'your-api-key',
  endpoint: 'https://api.ace3.ai'
});

// Load model
const model = await client.loadModel('path/to/model.onnx');

// Optimize model
const optimizedModel = await client.optimize(model, {
  target: 'inference',
  precision: 'fp16'
});

// Run inference
const result = await optimizedModel.predict(inputData);
```

### Core Classes

#### ACE3.Client

```javascript
class Client {
  constructor(options = {}) {
    // Initialize client with options
  }
  
  async loadModel(path, options = {}) {
    // Load model from path or URL
  }
  
  async optimize(model, options = {}) {
    // Optimize model for target
  }
  
  async listModels() {
    // List available models
  }
}
```

#### ACE3.Model

```javascript
class Model {
  async predict(inputData) {
    // Run single prediction
  }
  
  async batchPredict(batchData, options = {}) {
    // Run batch prediction
  }
  
  getInfo() {
    // Get model information
  }
  
  isOptimized() {
    // Check optimization status
  }
}
```

### Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/ace3-sdk@latest/dist/ace3.min.js"></script>
</head>
<body>
  <script>
    const client = new ACE3.Client({
      apiKey: 'your-api-key'
    });
    
    async function runInference() {
      const model = await client.loadModel('model.onnx');
      const result = await model.predict(inputData);
      console.log(result);
    }
    
    runInference();
  </script>
</body>
</html>
```

### Node.js Usage

```javascript
const ACE3 = require('ace3-sdk');
const fs = require('fs');

const client = new ACE3.Client({
  apiKey: process.env.ACE3_API_KEY
});

async function processFile(filePath) {
  try {
    const model = await client.loadModel('classifier.onnx');
    const data = fs.readFileSync(filePath);
    const result = await model.predict(data);
    return result;
  } catch (error) {
    console.error('Error processing file:', error);
  }
}
```

### TypeScript Support

```typescript
import ACE3, { Model, Client, OptimizationOptions } from 'ace3-sdk';

interface PredictionResult {
  outputs: number[][];
  latency: number;
  requestId: string;
}

const client: Client = new ACE3.Client({
  apiKey: 'your-api-key'
});

const options: OptimizationOptions = {
  target: 'inference',
  precision: 'fp16',
  batchSize: 32
};

async function runTypedInference(): Promise<PredictionResult> {
  const model: Model = await client.loadModel('model.onnx');
  const optimizedModel: Model = await client.optimize(model, options);
  return await optimizedModel.predict(inputData);
}
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
     -H "Content-Type: application/json" \
     https://api.ace3.ai/v1/models
```

### Endpoints

#### Models

**List Models**
```http
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
      "created_at": "2023-01-01T00:00:00Z",
      "size_mb": 45.2,
      "format": "onnx"
    }
  ],
  "total": 1,
  "page": 1,
  "per_page": 10
}
```

**Upload Model**
```http
POST /models
Content-Type: multipart/form-data

{
  "name": "my-model",
  "file": <model-file>,
  "format": "onnx",
  "description": "My custom model"
}
```

**Get Model Info**
```http
GET /models/{model_id}
```

**Delete Model**
```http
DELETE /models/{model_id}
```

#### Optimization

**Optimize Model**
```http
POST /models/{model_id}/optimize
Content-Type: application/json

{
  "target": "inference",
  "level": "standard",
  "options": {
    "precision": "fp16",
    "batch_size": 32
  }
}
```

#### Inference

**Run Inference**
```http
POST /models/{model_id}/predict
Content-Type: application/json

{
  "inputs": [[1.0, 2.0, 3.0]],
  "options": {
    "batch_size": 1,
    "timeout": 30
  }
}
```

Response:
```json
{
  "outputs": [[0.1, 0.9]],
  "latency_ms": 15.2,
  "request_id": "req-456",
  "model_version": "v1.0"
}
```

**Batch Inference**
```http
POST /models/{model_id}/batch_predict
Content-Type: application/json

{
  "inputs": [
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0],
    [7.0, 8.0, 9.0]
  ],
  "batch_size": 3
}
```

### Error Handling

Error responses follow this format:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Input shape mismatch",
    "details": {
      "expected_shape": [1, 3, 224, 224],
      "received_shape": [1, 224, 224, 3]
    }
  },
  "request_id": "req-789"
}
```

Common error codes:
- `INVALID_API_KEY`: Invalid or missing API key
- `MODEL_NOT_FOUND`: Model does not exist
- `INVALID_INPUT`: Input data format error
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Mobile SDKs

### iOS SDK

#### Installation

Add to your `Podfile`:

```ruby
pod 'ACE3SDK', '~> 1.0'
```

Or using Swift Package Manager:

```swift
dependencies: [
    .package(url: "https://github.com/ace3-ai/ace3-ios.git", from: "1.0.0")
]
```

#### Quick Start

```swift
import ACE3SDK

// Initialize client
let client = ACE3Client(apiKey: "your-api-key")

// Load model
client.loadModel(path: "model.onnx") { result in
    switch result {
    case .success(let model):
        // Optimize model
        client.optimize(model: model, target: .inference) { optimizedResult in
            switch optimizedResult {
            case .success(let optimizedModel):
                // Run inference
                optimizedModel.predict(input: inputData) { prediction in
                    print("Result: \(prediction)")
                }
            case .failure(let error):
                print("Optimization failed: \(error)")
            }
        }
    case .failure(let error):
        print("Model loading failed: \(error)")
    }
}
```

#### Core Classes

```swift
public class ACE3Client {
    public init(apiKey: String, endpoint: String = "https://api.ace3.ai")
    
    public func loadModel(path: String, completion: @escaping (Result<ACE3Model, Error>) -> Void)
    
    public func optimize(model: ACE3Model, target: OptimizationTarget, completion: @escaping (Result<ACE3Model, Error>) -> Void)
}

public class ACE3Model {
    public func predict(input: Data, completion: @escaping (Result<Data, Error>) -> Void)
    
    public func batchPredict(inputs: [Data], completion: @escaping (Result<[Data], Error>) -> Void)
    
    public func getInfo() -> ModelInfo
}
```

### Android SDK

#### Installation

Add to your `build.gradle`:

```gradle
dependencies {
    implementation 'ai.ace3:ace3-android:1.0.0'
}
```

#### Quick Start

```kotlin
import ai.ace3.sdk.ACE3Client
import ai.ace3.sdk.ACE3Model
import ai.ace3.sdk.OptimizationTarget

// Initialize client
val client = ACE3Client(apiKey = "your-api-key")

// Load model
client.loadModel("model.onnx") { result ->
    result.onSuccess { model ->
        // Optimize model
        client.optimize(model, OptimizationTarget.INFERENCE) { optimizedResult ->
            optimizedResult.onSuccess { optimizedModel ->
                // Run inference
                optimizedModel.predict(inputData) { prediction ->
                    println("Result: $prediction")
                }
            }
        }
    }.onFailure { error ->
        println("Model loading failed: $error")
    }
}
```

#### Core Classes

```kotlin
class ACE3Client(
    private val apiKey: String,
    private val endpoint: String = "https://api.ace3.ai"
) {
    suspend fun loadModel(path: String): Result<ACE3Model>
    
    suspend fun optimize(
        model: ACE3Model, 
        target: OptimizationTarget
    ): Result<ACE3Model>
}

class ACE3Model {
    suspend fun predict(input: ByteArray): Result<ByteArray>
    
    suspend fun batchPredict(inputs: List<ByteArray>): Result<List<ByteArray>>
    
    fun getInfo(): ModelInfo
}
```

## CLI Tools

### Installation

```bash
# Install CLI tools
pip install ace3-cli

# Or download binary
curl -L https://github.com/ace3-ai/cli/releases/latest/download/ace3-cli-linux -o ace3
chmod +x ace3
```

### Commands

#### Model Management

```bash
# List models
ace3 models list

# Upload model
ace3 models upload --name "my-model" --file model.onnx

# Download model
ace3 models download --id model-123 --output model.onnx

# Delete model
ace3 models delete --id model-123
```

#### Optimization

```bash
# Optimize model
ace3 optimize --input model.onnx --output optimized.onnx --target inference

# Optimize with options
ace3 optimize --input model.onnx --output optimized.onnx \
  --target inference --precision fp16 --batch-size 32
```

#### Inference

```bash
# Run inference
ace3 predict --model model.onnx --input data.json

# Batch inference
ace3 batch-predict --model model.onnx --input batch_data.json --batch-size 16
```

#### Benchmarking

```bash
# Benchmark model
ace3 benchmark --model model.onnx --input sample_data.json --runs 100

# Compare models
ace3 compare --model1 original.onnx --model2 optimized.onnx --input data.json
```

### Configuration

Create `~/.ace3/config.yaml`:

```yaml
api_key: your-api-key
endpoint: https://api.ace3.ai
timeout: 30
log_level: INFO
cache_dir: ~/.ace3/cache
```

## SDK Comparison

| Feature | Python | JavaScript | iOS | Android | CLI |
|---------|--------|------------|-----|---------|-----|
| Model Loading | ✅ | ✅ | ✅ | ✅ | ✅ |
| Optimization | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inference | ✅ | ✅ | ✅ | ✅ | ✅ |
| Batch Processing | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async Support | ✅ | ✅ | ✅ | ✅ | ❌ |
| Training | ✅ | ❌ | ❌ | ❌ | ❌ |
| Profiling | ✅ | ✅ | ✅ | ✅ | ✅ |
| Custom Operators | ✅ | ✅ | ❌ | ❌ | ❌ |

## Version Compatibility

| SDK Version | API Version | Python | Node.js | iOS | Android |
|-------------|-------------|--------|---------|-----|---------|
| 1.0.x | v1 | 3.7+ | 14+ | 13+ | API 21+ |
| 1.1.x | v1 | 3.7+ | 14+ | 13+ | API 21+ |
| 2.0.x | v2 | 3.8+ | 16+ | 14+ | API 23+ |

## Migration Guides

### Upgrading from v1.x to v2.x

**Python SDK Changes:**

```python
# v1.x
client = ace3.Client("your-api-key")
model = client.load_model("model.onnx")

# v2.x
client = ace3.Client(api_key="your-api-key")
model = client.load_model("model.onnx")
```

**JavaScript SDK Changes:**

```javascript
// v1.x
const client = new ACE3.Client("your-api-key");

// v2.x
const client = new ACE3.Client({ apiKey: "your-api-key" });
```

## Support and Resources

- **Documentation**: [https://docs.ace3.ai](https://docs.ace3.ai)
- **GitHub**: [https://github.com/ace3-ai](https://github.com/ace3-ai)
- **Discord**: [https://discord.gg/ace3](https://discord.gg/ace3)
- **Stack Overflow**: Tag questions with `ace3`
- **Email Support**: support@ace3.ai

