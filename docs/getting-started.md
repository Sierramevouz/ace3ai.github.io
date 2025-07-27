# Getting Started with ACE3

Welcome to ACE3! This guide will help you get up and running with our AI acceleration platform in just a few minutes.

## What is ACE3?

ACE3 is a comprehensive AI acceleration platform that helps you:

- **Speed up model inference** by up to 5x
- **Accelerate training** with distributed computing
- **Reduce costs** while maintaining high performance
- **Scale seamlessly** from development to production

## Prerequisites

Before you begin, make sure you have:

- **Python 3.7+** installed on your system
- **CUDA-compatible GPU** (optional but recommended)
- **Valid ACE3 account** with API key

## Quick Installation

### Step 1: Install the ACE3 SDK

```bash
pip install ace3-sdk
```

### Step 2: Get Your API Key

1. Sign up at [ACE3 Dashboard](https://dashboard.ace3.ai)
2. Navigate to API Keys section
3. Generate a new API key
4. Copy and save it securely

### Step 3: Verify Installation

```python
import ace3

# Test your installation
print(ace3.__version__)

# Initialize client with your API key
client = ace3.Client(api_key="your-api-key-here")
print("ACE3 client initialized successfully!")
```

## Your First ACE3 Project

Let's create a simple example that demonstrates ACE3's inference acceleration:

```python
import ace3
import numpy as np

# Initialize the client
client = ace3.Client(api_key="your-api-key")

# Load a sample model (replace with your model path)
model = client.load_model("path/to/your/model.onnx")

# Accelerate the model for inference
accelerated_model = client.accelerate(model, target="inference")

# Create sample input data
input_data = np.random.randn(1, 224, 224, 3)

# Run inference
result = accelerated_model.predict(input_data)
print(f"Inference result shape: {result.shape}")
```

## Basic Concepts

### Models

ACE3 supports various model formats:
- **ONNX** - Open Neural Network Exchange format
- **PyTorch** - Native PyTorch models
- **TensorFlow** - TensorFlow SavedModel format
- **Hugging Face** - Models from Hugging Face Hub

### Acceleration Targets

Choose the right acceleration target for your use case:

- **inference** - Optimize for fast prediction
- **training** - Optimize for efficient training
- **mixed** - Balance between inference and training

### Optimization Levels

ACE3 offers different optimization levels:

- **basic** - Quick optimizations with minimal changes
- **standard** - Balanced optimization (recommended)
- **aggressive** - Maximum optimization with potential accuracy trade-offs

## Configuration

### Environment Variables

Set up your environment for easier development:

```bash
export ACE3_API_KEY="your-api-key"
export ACE3_ENDPOINT="https://api.ace3.ai"
export ACE3_LOG_LEVEL="INFO"
```

### Configuration File

Create a `ace3.config.json` file in your project:

```json
{
  "api_key": "your-api-key",
  "endpoint": "https://api.ace3.ai",
  "timeout": 30,
  "retry_attempts": 3,
  "optimization_level": "standard"
}
```

## Next Steps

Now that you have ACE3 set up, explore these topics:

1. **[API Reference](api-reference.md)** - Detailed API documentation
2. **[Inference Acceleration](inference.md)** - Optimize your models for production
3. **[Training Acceleration](training.md)** - Speed up model training
4. **[Tutorials](tutorials.md)** - Step-by-step guides for common use cases

## Need Help?

- 📚 **Documentation**: Browse our comprehensive docs
- 💬 **Community**: Join our [Discord server](https://discord.gg/ace3)
- 📧 **Support**: Email us at support@ace3.ai
- 🐛 **Issues**: Report bugs on [GitHub](https://github.com/ace3-ai/ace3-sdk)

## Common Issues

### Installation Problems

If you encounter installation issues:

```bash
# Update pip first
pip install --upgrade pip

# Install with verbose output
pip install -v ace3-sdk

# For development installation
pip install ace3-sdk[dev]
```

### API Key Issues

If your API key isn't working:

1. Check that you copied the key correctly
2. Ensure your account is active
3. Verify the key hasn't expired
4. Contact support if issues persist

### GPU Detection

To check if ACE3 can detect your GPU:

```python
import ace3

client = ace3.Client(api_key="your-api-key")
print(f"Available devices: {client.list_devices()}")
print(f"GPU available: {client.is_gpu_available()}")
```

