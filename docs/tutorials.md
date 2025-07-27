# Tutorials

Step-by-step tutorials to help you get the most out of ACE3.

## Getting Started Tutorials

### Tutorial 1: Your First ACE3 Model

Learn the basics of loading and optimizing a model with ACE3.

**What you'll learn:**
- How to load a model
- Basic optimization techniques
- Running inference
- Measuring performance improvements

**Prerequisites:**
- Python 3.7+
- ACE3 SDK installed
- A sample ONNX model

**Step 1: Setup**

```python
import ace3
import numpy as np
import time

# Initialize ACE3 client
client = ace3.Client(api_key="your-api-key")
print("ACE3 client initialized successfully!")
```

**Step 2: Load Your Model**

```python
# Load a pre-trained model (replace with your model path)
model_path = "resnet50.onnx"
model = client.load_model(model_path)

print(f"Model loaded: {model.info().name}")
print(f"Input shape: {model.info().input_shape}")
print(f"Output shape: {model.info().output_shape}")
```

**Step 3: Prepare Test Data**

```python
# Create sample input data
batch_size = 1
input_shape = (batch_size, 3, 224, 224)  # Adjust based on your model
test_input = np.random.randn(*input_shape).astype(np.float32)

print(f"Test input shape: {test_input.shape}")
```

**Step 4: Baseline Performance**

```python
# Measure baseline performance
def measure_performance(model, input_data, num_runs=100):
    # Warmup
    for _ in range(10):
        _ = model.predict(input_data)
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        result = model.predict(input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    return avg_time, result

baseline_time, baseline_result = measure_performance(model, test_input)
print(f"Baseline average inference time: {baseline_time:.2f} ms")
```

**Step 5: Optimize the Model**

```python
# Optimize for inference
optimized_model = client.optimize(model, target="inference")
print("Model optimization completed!")

# Measure optimized performance
optimized_time, optimized_result = measure_performance(optimized_model, test_input)
print(f"Optimized average inference time: {optimized_time:.2f} ms")

# Calculate speedup
speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x")
```

**Step 6: Verify Results**

```python
# Verify that results are similar
import numpy as np

# Calculate difference
diff = np.abs(baseline_result - optimized_result)
max_diff = np.max(diff)
mean_diff = np.mean(diff)

print(f"Maximum difference: {max_diff:.6f}")
print(f"Mean difference: {mean_diff:.6f}")

# Check if results are close enough
tolerance = 1e-3
if max_diff < tolerance:
    print("✅ Optimization successful - results are consistent!")
else:
    print("⚠️ Large difference detected - consider using higher precision")
```

### Tutorial 2: Batch Processing for High Throughput

Learn how to process multiple inputs efficiently.

**What you'll learn:**
- Batch processing techniques
- Optimal batch size selection
- Throughput optimization
- Memory management

**Step 1: Prepare Batch Data**

```python
import ace3
import numpy as np

client = ace3.Client(api_key="your-api-key")
model = client.load_model("your-model.onnx")
optimized_model = client.optimize(model, target="inference")

# Create batch data
batch_sizes = [1, 4, 8, 16, 32]
input_shape = (3, 224, 224)  # Single input shape

batch_data = {}
for batch_size in batch_sizes:
    batch_data[batch_size] = np.random.randn(batch_size, *input_shape).astype(np.float32)
    print(f"Created batch data for batch size {batch_size}: {batch_data[batch_size].shape}")
```

**Step 2: Measure Batch Performance**

```python
def measure_batch_performance(model, batch_data, num_runs=50):
    # Warmup
    for _ in range(5):
        _ = model.batch_predict(batch_data)
    
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        results = model.batch_predict(batch_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_runs * 1000  # ms
    avg_time_per_sample = avg_time_per_batch / len(batch_data)  # ms per sample
    throughput = len(batch_data) * num_runs / total_time  # samples per second
    
    return avg_time_per_batch, avg_time_per_sample, throughput

# Test different batch sizes
results = {}
for batch_size in batch_sizes:
    batch_time, sample_time, throughput = measure_batch_performance(
        optimized_model, batch_data[batch_size]
    )
    results[batch_size] = {
        'batch_time': batch_time,
        'sample_time': sample_time,
        'throughput': throughput
    }
    print(f"Batch size {batch_size:2d}: {batch_time:6.2f} ms/batch, "
          f"{sample_time:6.2f} ms/sample, {throughput:6.1f} samples/sec")
```

**Step 3: Find Optimal Batch Size**

```python
# Find the batch size with highest throughput
optimal_batch_size = max(results.keys(), key=lambda x: results[x]['throughput'])
optimal_throughput = results[optimal_batch_size]['throughput']

print(f"\nOptimal batch size: {optimal_batch_size}")
print(f"Maximum throughput: {optimal_throughput:.1f} samples/sec")

# Plot results (optional)
try:
    import matplotlib.pyplot as plt
    
    batch_sizes_list = list(results.keys())
    throughputs = [results[bs]['throughput'] for bs in batch_sizes_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes_list, throughputs, 'bo-')
    plt.xlabel('Batch Size')
    plt.ylabel('Throughput (samples/sec)')
    plt.title('Throughput vs Batch Size')
    plt.grid(True)
    plt.show()
except ImportError:
    print("Install matplotlib to see the throughput plot")
```

### Tutorial 3: Mixed Precision Training

Learn how to use mixed precision to accelerate training.

**What you'll learn:**
- Mixed precision concepts
- Automatic mixed precision (AMP)
- Loss scaling
- Memory savings

**Step 1: Setup Training Environment**

```python
import ace3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model and data
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Create dummy dataset
X = torch.randn(1000, 784)
y = torch.randint(0, 10, (1000,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

**Step 2: Standard Training (FP32)**

```python
import time

def train_epoch(model, dataloader, optimizer, criterion, use_amp=False):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    scaler = ace3.GradientScaler() if use_amp else None
    
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        
        if use_amp:
            with ace3.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    end_time = time.time()
    avg_loss = total_loss / len(dataloader)
    epoch_time = end_time - start_time
    
    return avg_loss, epoch_time

# Train with FP32
print("Training with FP32...")
fp32_loss, fp32_time = train_epoch(model, dataloader, optimizer, criterion, use_amp=False)
print(f"FP32 - Loss: {fp32_loss:.4f}, Time: {fp32_time:.2f}s")
```

**Step 3: Mixed Precision Training**

```python
# Reset model for fair comparison
model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train with mixed precision
print("Training with Mixed Precision...")
amp_loss, amp_time = train_epoch(model, dataloader, optimizer, criterion, use_amp=True)
print(f"AMP - Loss: {amp_loss:.4f}, Time: {amp_time:.2f}s")

# Calculate speedup
speedup = fp32_time / amp_time
print(f"Speedup: {speedup:.2f}x")
```

**Step 4: Memory Usage Comparison**

```python
def measure_memory_usage(model, dataloader, use_amp=False):
    import torch
    
    model.train()
    torch.cuda.empty_cache()
    
    # Measure peak memory
    torch.cuda.reset_peak_memory_stats()
    
    scaler = ace3.GradientScaler() if use_amp else None
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx >= 5:  # Only measure first few batches
            break
            
        optimizer.zero_grad()
        
        if use_amp:
            with ace3.autocast():
                output = model(data)
                loss = criterion(output, target)
            scaler.scale(loss).backward()
        else:
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
    
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    return peak_memory

# Compare memory usage (if CUDA available)
if torch.cuda.is_available():
    model = model.cuda()
    
    fp32_memory = measure_memory_usage(model, dataloader, use_amp=False)
    amp_memory = measure_memory_usage(model, dataloader, use_amp=True)
    
    print(f"FP32 peak memory: {fp32_memory:.1f} MB")
    print(f"AMP peak memory: {amp_memory:.1f} MB")
    print(f"Memory savings: {(fp32_memory - amp_memory) / fp32_memory * 100:.1f}%")
```

## Computer Vision Tutorials

### Tutorial 4: Image Classification with ACE3

Build an optimized image classification pipeline.

**What you'll learn:**
- Image preprocessing
- Model optimization for vision tasks
- Batch processing for images
- Performance benchmarking

**Step 1: Setup and Data Preparation**

```python
import ace3
import numpy as np
from PIL import Image
import requests
from io import BytesIO

# Initialize ACE3
client = ace3.Client(api_key="your-api-key")

# Load a pre-trained image classification model
model = client.load_model("resnet50.onnx")  # or your model
optimized_model = client.optimize(model, target="inference")

# Download sample images
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Sample image URLs (replace with your own)
image_urls = [
    "https://example.com/cat.jpg",
    "https://example.com/dog.jpg",
    "https://example.com/bird.jpg"
]

# For demo, create random images
images = [Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)) 
          for _ in range(3)]
```

**Step 2: Image Preprocessing**

```python
def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model input"""
    # Resize
    image = image.resize(target_size)
    
    # Convert to array
    img_array = np.array(image).astype(np.float32)
    
    # Normalize (ImageNet normalization)
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255
    img_array = (img_array - mean) / std
    
    # Change from HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    
    return img_array

# Preprocess all images
preprocessed_images = [preprocess_image(img) for img in images]
batch_input = np.stack(preprocessed_images)

print(f"Batch input shape: {batch_input.shape}")
```

**Step 3: Run Inference**

```python
# Single image inference
single_result = optimized_model.predict(batch_input[0:1])
print(f"Single image result shape: {single_result.shape}")

# Batch inference
batch_results = optimized_model.batch_predict(batch_input)
print(f"Batch results shape: {batch_results.shape}")

# Get top predictions
def get_top_predictions(predictions, top_k=5):
    """Get top-k predictions"""
    top_indices = np.argsort(predictions)[-top_k:][::-1]
    top_scores = predictions[top_indices]
    return list(zip(top_indices, top_scores))

# Process results
for i, result in enumerate(batch_results):
    top_preds = get_top_predictions(result)
    print(f"Image {i+1} top predictions:")
    for class_id, score in top_preds:
        print(f"  Class {class_id}: {score:.4f}")
```

### Tutorial 5: Object Detection Pipeline

Build an optimized object detection system.

**What you'll learn:**
- Object detection preprocessing
- Post-processing techniques
- Non-maximum suppression
- Visualization

**Step 1: Load Detection Model**

```python
import ace3
import numpy as np
import cv2

# Load object detection model (e.g., YOLOv5)
client = ace3.Client(api_key="your-api-key")
detection_model = client.load_model("yolov5s.onnx")
optimized_detector = client.optimize(detection_model, target="inference")

print(f"Model input shape: {detection_model.info().input_shape}")
print(f"Model output shape: {detection_model.info().output_shape}")
```

**Step 2: Preprocessing for Detection**

```python
def preprocess_for_detection(image, input_size=640):
    """Preprocess image for object detection"""
    # Get original dimensions
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(input_size / h, input_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # Normalize and transpose
    padded = padded.astype(np.float32) / 255.0
    padded = np.transpose(padded, (2, 0, 1))
    padded = np.expand_dims(padded, 0)
    
    return padded, scale

# Load and preprocess image
image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # Demo image
preprocessed, scale = preprocess_for_detection(image)

print(f"Preprocessed shape: {preprocessed.shape}")
print(f"Scale factor: {scale}")
```

**Step 3: Post-processing**

```python
def postprocess_detections(predictions, scale, conf_threshold=0.5, iou_threshold=0.45):
    """Post-process detection results"""
    # Extract predictions
    boxes = predictions[0][:, :4]  # x, y, w, h
    scores = predictions[0][:, 4]  # confidence
    class_probs = predictions[0][:, 5:]  # class probabilities
    
    # Filter by confidence
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_probs = class_probs[mask]
    
    # Get class predictions
    class_ids = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)
    
    # Convert box format (center_x, center_y, w, h) to (x1, y1, x2, y2)
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    # Scale back to original image size
    x1 /= scale
    y1 /= scale
    x2 /= scale
    y2 /= scale
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), conf_threshold, iou_threshold
    )
    
    if len(indices) > 0:
        indices = indices.flatten()
        return {
            'boxes': np.column_stack([x1[indices], y1[indices], x2[indices], y2[indices]]),
            'scores': scores[indices],
            'class_ids': class_ids[indices]
        }
    else:
        return {'boxes': np.array([]), 'scores': np.array([]), 'class_ids': np.array([])}

# Run detection
predictions = optimized_detector.predict(preprocessed)
detections = postprocess_detections(predictions, scale)

print(f"Found {len(detections['boxes'])} objects")
for i, (box, score, class_id) in enumerate(zip(
    detections['boxes'], detections['scores'], detections['class_ids']
)):
    print(f"Object {i+1}: Class {class_id}, Score {score:.3f}, Box {box}")
```

## Natural Language Processing Tutorials

### Tutorial 6: Text Classification with BERT

Optimize BERT models for text classification.

**What you'll learn:**
- BERT model optimization
- Text preprocessing
- Sequence length optimization
- Batch processing for text

**Step 1: Setup BERT Model**

```python
import ace3
import numpy as np
from transformers import AutoTokenizer

# Load BERT model
client = ace3.Client(api_key="your-api-key")
bert_model = client.load_model("huggingface:bert-base-uncased")

# Optimize for inference
optimized_bert = client.optimize(
    bert_model,
    target="inference",
    options={
        "max_sequence_length": 512,
        "batch_size": 32,
        "precision": "fp16"
    }
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

**Step 2: Text Preprocessing**

```python
def preprocess_texts(texts, tokenizer, max_length=512):
    """Preprocess texts for BERT"""
    # Tokenize texts
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"
    )
    
    return {
        'input_ids': encoded['input_ids'],
        'attention_mask': encoded['attention_mask']
    }

# Sample texts
texts = [
    "This movie is absolutely fantastic!",
    "I didn't like this product at all.",
    "The service was okay, nothing special.",
    "Amazing quality and fast delivery!",
    "Terrible experience, would not recommend."
]

# Preprocess
inputs = preprocess_texts(texts, tokenizer)
print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")
```

**Step 3: Run BERT Inference**

```python
# Run inference
results = optimized_bert.predict([inputs['input_ids'], inputs['attention_mask']])

# Extract embeddings (last hidden state)
embeddings = results[0]  # Shape: (batch_size, seq_length, hidden_size)
pooled_embeddings = np.mean(embeddings, axis=1)  # Average pooling

print(f"Embeddings shape: {embeddings.shape}")
print(f"Pooled embeddings shape: {pooled_embeddings.shape}")

# Use embeddings for classification (example)
# You would typically add a classification head here
for i, text in enumerate(texts):
    embedding = pooled_embeddings[i]
    print(f"Text: '{text[:50]}...'")
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
```

## Best Practices and Tips

### Performance Optimization Tips

1. **Choose the Right Batch Size**
   ```python
   # Find optimal batch size
   optimal_batch_size = ace3.find_optimal_batch_size(model, sample_input)
   print(f"Optimal batch size: {optimal_batch_size}")
   ```

2. **Use Appropriate Precision**
   ```python
   # For maximum speed with minimal accuracy loss
   model = client.optimize(model, options={"precision": "fp16"})
   
   # For maximum accuracy
   model = client.optimize(model, options={"precision": "fp32"})
   ```

3. **Profile Your Code**
   ```python
   # Profile inference
   profiler = ace3.InferenceProfiler()
   with profiler:
       result = model.predict(input_data)
   
   print(profiler.summary())
   ```

### Memory Management Tips

1. **Monitor Memory Usage**
   ```python
   # Check memory usage
   memory_info = ace3.get_memory_info()
   print(f"GPU memory used: {memory_info.gpu_used_mb} MB")
   print(f"GPU memory available: {memory_info.gpu_available_mb} MB")
   ```

2. **Clean Up Resources**
   ```python
   # Free model memory when done
   del model
   ace3.clear_cache()
   ```

### Debugging Tips

1. **Enable Debug Mode**
   ```python
   # Enable detailed logging
   ace3.set_debug_mode(True)
   ace3.set_log_level("DEBUG")
   ```

2. **Validate Optimization**
   ```python
   # Compare original vs optimized results
   original_result = original_model.predict(test_input)
   optimized_result = optimized_model.predict(test_input)
   
   diff = np.abs(original_result - optimized_result)
   print(f"Max difference: {np.max(diff)}")
   print(f"Mean difference: {np.mean(diff)}")
   ```

## Next Steps

After completing these tutorials, you should:

1. **Explore Advanced Features**: Try model ensembling, A/B testing, and custom operators
2. **Optimize for Production**: Implement proper error handling, monitoring, and scaling
3. **Join the Community**: Share your experiences and learn from others
4. **Read the Documentation**: Dive deeper into specific topics that interest you

## Additional Resources

- **[API Reference](api-reference.md)**: Complete API documentation
- **[Best Practices Guide](best-practices.md)**: Production deployment tips
- **[Troubleshooting Guide](troubleshooting.md)**: Common issues and solutions
- **[Community Forum](https://community.ace3.ai)**: Get help from other users
- **[GitHub Examples](https://github.com/ace3-ai/examples)**: More code examples

