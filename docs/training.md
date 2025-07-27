# Training Acceleration

Accelerate your model training with ACE3's distributed computing and optimization techniques.

## Overview

ACE3's training acceleration provides:

- **Distributed training** across multiple GPUs and nodes
- **Mixed precision training** for faster training and reduced memory usage
- **Gradient optimization** techniques for better convergence
- **Memory optimization** to train larger models
- **Automatic hyperparameter tuning** for optimal performance

## Quick Start

```python
import ace3

# Initialize client
client = ace3.Client(api_key="your-api-key")

# Load your model and dataset
model = client.load_model("path/to/model.py")
dataset = client.load_dataset("path/to/dataset")

# Configure distributed training
config = ace3.TrainingConfig(
    num_gpus=4,
    batch_size=32,
    learning_rate=1e-4,
    mixed_precision=True
)

# Start accelerated training
trainer = ace3.DistributedTrainer(config)
trainer.train(model, dataset)
```

## Distributed Training

### Single-Node Multi-GPU

Train on multiple GPUs within a single machine:

```python
# Configure single-node training
config = ace3.DistributedConfig(
    num_gpus=4,
    backend="nccl",  # NVIDIA GPUs
    # backend="gloo",  # CPU or mixed
)

# Initialize trainer
trainer = ace3.DistributedTrainer(config)

# Train the model
trainer.train(
    model=model,
    train_dataset=train_data,
    validation_dataset=val_data,
    epochs=10
)
```

### Multi-Node Training

Scale training across multiple machines:

```python
# Configure multi-node training
config = ace3.DistributedConfig(
    num_gpus=4,
    num_nodes=4,
    node_rank=0,  # Current node rank (0, 1, 2, 3)
    master_addr="192.168.1.100",
    master_port=29500,
    backend="nccl"
)

# Initialize trainer
trainer = ace3.DistributedTrainer(config)

# Train across nodes
trainer.train(model, dataset)
```

### Data Parallel Training

Distribute data across multiple devices:

```python
# Data parallel configuration
config = ace3.TrainingConfig(
    parallelism="data",
    num_gpus=8,
    batch_size=256,  # Total batch size across all GPUs
    gradient_accumulation_steps=4
)

trainer = ace3.DataParallelTrainer(config)
trainer.train(model, dataset)
```

### Model Parallel Training

Split large models across multiple devices:

```python
# Model parallel configuration
config = ace3.TrainingConfig(
    parallelism="model",
    num_gpus=4,
    model_sharding_strategy="layer"  # or "tensor", "pipeline"
)

trainer = ace3.ModelParallelTrainer(config)
trainer.train(model, dataset)
```

### Pipeline Parallel Training

Use pipeline parallelism for very large models:

```python
# Pipeline parallel configuration
config = ace3.TrainingConfig(
    parallelism="pipeline",
    num_stages=4,
    micro_batch_size=8,
    gradient_accumulation_steps=16
)

trainer = ace3.PipelineParallelTrainer(config)
trainer.train(model, dataset)
```

## Mixed Precision Training

### Automatic Mixed Precision

Enable automatic mixed precision for faster training:

```python
# Enable AMP
config = ace3.TrainingConfig(
    mixed_precision=True,
    amp_level="O1",  # O0, O1, O2, O3
    loss_scaling="dynamic"  # or "static", value
)

trainer = ace3.Trainer(config)
trainer.train(model, dataset)
```

### Manual Mixed Precision

Fine-tune mixed precision settings:

```python
# Manual AMP configuration
amp_config = ace3.AMPConfig(
    enabled=True,
    opt_level="O2",
    keep_batchnorm_fp32=True,
    loss_scale="dynamic",
    min_loss_scale=1.0,
    max_loss_scale=65536.0
)

trainer = ace3.Trainer(amp_config=amp_config)
trainer.train(model, dataset)
```

### Gradient Scaling

Handle gradient scaling for mixed precision:

```python
# Custom gradient scaling
scaler = ace3.GradientScaler(
    init_scale=65536.0,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000
)

# Use in training loop
for batch in dataloader:
    with ace3.autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

## Memory Optimization

### Gradient Checkpointing

Trade computation for memory:

```python
# Enable gradient checkpointing
config = ace3.TrainingConfig(
    gradient_checkpointing=True,
    checkpoint_segments=4  # Number of segments
)

trainer = ace3.Trainer(config)
trainer.train(model, dataset)
```

### CPU Offloading

Offload parameters and gradients to CPU:

```python
# Enable CPU offloading
config = ace3.TrainingConfig(
    cpu_offload=True,
    offload_optimizer=True,
    offload_parameters=True
)

trainer = ace3.Trainer(config)
trainer.train(model, dataset)
```

### Memory-Efficient Optimizers

Use memory-efficient optimizers:

```python
# Use memory-efficient Adam
optimizer = ace3.optimizers.MemoryEfficientAdam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)

# Use 8-bit Adam
optimizer = ace3.optimizers.Adam8bit(
    model.parameters(),
    lr=1e-4
)
```

## Advanced Training Techniques

### Gradient Accumulation

Simulate larger batch sizes:

```python
# Configure gradient accumulation
config = ace3.TrainingConfig(
    batch_size=32,  # Per-device batch size
    gradient_accumulation_steps=8,  # Effective batch size: 32 * 8 = 256
    max_grad_norm=1.0  # Gradient clipping
)

trainer = ace3.Trainer(config)
trainer.train(model, dataset)
```

### Learning Rate Scheduling

Optimize learning rate schedules:

```python
# Built-in schedulers
scheduler = ace3.schedulers.CosineAnnealingLR(
    optimizer,
    T_max=1000,
    eta_min=1e-6
)

# Warmup scheduler
scheduler = ace3.schedulers.WarmupCosineScheduler(
    optimizer,
    warmup_steps=1000,
    total_steps=10000,
    max_lr=1e-3,
    min_lr=1e-6
)

# Use in training
trainer = ace3.Trainer(scheduler=scheduler)
trainer.train(model, dataset)
```

### Dynamic Loss Scaling

Automatically adjust loss scaling:

```python
# Dynamic loss scaling
loss_scaler = ace3.DynamicLossScaler(
    init_scale=2**16,
    scale_factor=2.0,
    scale_window=2000
)

# Use in training loop
for batch in dataloader:
    with ace3.autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)
    
    scaled_loss = loss_scaler.scale(loss)
    scaled_loss.backward()
    
    loss_scaler.step(optimizer)
    loss_scaler.update()
```

## Hyperparameter Optimization

### Automatic Hyperparameter Tuning

Let ACE3 find optimal hyperparameters:

```python
# Define search space
search_space = {
    "learning_rate": ace3.hp.loguniform(1e-5, 1e-2),
    "batch_size": ace3.hp.choice([16, 32, 64, 128]),
    "weight_decay": ace3.hp.uniform(0.0, 0.1),
    "dropout": ace3.hp.uniform(0.1, 0.5)
}

# Configure hyperparameter search
hp_config = ace3.HPConfig(
    search_space=search_space,
    algorithm="bayesian",  # or "random", "grid"
    max_trials=50,
    objective="val_accuracy",
    direction="maximize"
)

# Run hyperparameter optimization
best_params = ace3.hyperparameter_search(
    model=model,
    dataset=dataset,
    config=hp_config
)

print(f"Best parameters: {best_params}")
```

### Manual Hyperparameter Search

Implement custom search strategies:

```python
# Grid search
param_grid = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "batch_size": [32, 64, 128],
    "weight_decay": [0.01, 0.05, 0.1]
}

best_score = 0
best_params = None

for params in ace3.utils.grid_search(param_grid):
    config = ace3.TrainingConfig(**params)
    trainer = ace3.Trainer(config)
    
    results = trainer.train(model, dataset)
    score = results.best_val_accuracy
    
    if score > best_score:
        best_score = score
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

## Monitoring and Logging

### Training Metrics

Monitor training progress in real-time:

```python
# Enable comprehensive logging
logger = ace3.TrainingLogger(
    log_dir="./logs",
    log_interval=100,  # Log every 100 steps
    save_checkpoints=True,
    checkpoint_interval=1000
)

# Add custom metrics
logger.add_metric("custom_loss", lambda: compute_custom_loss())

trainer = ace3.Trainer(logger=logger)
trainer.train(model, dataset)
```

### TensorBoard Integration

Visualize training with TensorBoard:

```python
# Enable TensorBoard logging
tensorboard = ace3.TensorBoardLogger(
    log_dir="./tensorboard_logs",
    log_graph=True,
    log_images=True,
    log_histograms=True
)

trainer = ace3.Trainer(logger=tensorboard)
trainer.train(model, dataset)

# View in TensorBoard
# tensorboard --logdir=./tensorboard_logs
```

### Weights & Biases Integration

Track experiments with W&B:

```python
# Initialize W&B logging
wandb_logger = ace3.WandBLogger(
    project="my-project",
    name="experiment-1",
    config={
        "learning_rate": 1e-4,
        "batch_size": 32,
        "model": "resnet50"
    }
)

trainer = ace3.Trainer(logger=wandb_logger)
trainer.train(model, dataset)
```

## Callbacks and Hooks

### Built-in Callbacks

Use pre-built callbacks for common tasks:

```python
# Early stopping
early_stopping = ace3.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)

# Model checkpointing
checkpoint = ace3.callbacks.ModelCheckpoint(
    filepath="./checkpoints/model-{epoch:02d}-{val_loss:.2f}.pt",
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False
)

# Learning rate reduction
lr_reducer = ace3.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-7
)

# Use callbacks
trainer = ace3.Trainer(callbacks=[early_stopping, checkpoint, lr_reducer])
trainer.train(model, dataset)
```

### Custom Callbacks

Create custom callbacks for specific needs:

```python
class CustomCallback(ace3.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting epoch {epoch}")
    
    def on_epoch_end(self, epoch, logs=None):
        val_acc = logs.get("val_accuracy", 0)
        if val_acc > 0.95:
            print("Target accuracy reached!")
            self.model.stop_training = True
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Processed batch {batch}")

# Use custom callback
custom_callback = CustomCallback()
trainer = ace3.Trainer(callbacks=[custom_callback])
trainer.train(model, dataset)
```

## Model Checkpointing

### Automatic Checkpointing

Save model checkpoints automatically:

```python
# Configure checkpointing
checkpoint_config = ace3.CheckpointConfig(
    save_dir="./checkpoints",
    save_interval=1000,  # Save every 1000 steps
    max_checkpoints=5,   # Keep only 5 latest checkpoints
    save_optimizer=True,
    save_scheduler=True
)

trainer = ace3.Trainer(checkpoint_config=checkpoint_config)
trainer.train(model, dataset)
```

### Resume Training

Resume training from a checkpoint:

```python
# Resume from checkpoint
trainer = ace3.Trainer()
trainer.load_checkpoint("./checkpoints/checkpoint-5000.pt")
trainer.train(model, dataset, resume=True)
```

### Manual Checkpointing

Save checkpoints manually:

```python
# Save checkpoint
checkpoint = {
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
    "accuracy": accuracy
}

ace3.save_checkpoint(checkpoint, "./checkpoints/manual_checkpoint.pt")

# Load checkpoint
checkpoint = ace3.load_checkpoint("./checkpoints/manual_checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

## Performance Optimization

### Profiling Training

Profile training performance:

```python
# Enable training profiler
profiler = ace3.TrainingProfiler(
    activities=[
        ace3.ProfilerActivity.CPU,
        ace3.ProfilerActivity.CUDA
    ],
    record_shapes=True,
    profile_memory=True
)

with profiler:
    trainer.train(model, dataset)

# Analyze results
print(profiler.key_averages().table(sort_by="cuda_time_total"))
profiler.export_chrome_trace("training_trace.json")
```

### Memory Profiling

Monitor memory usage during training:

```python
# Memory profiler
memory_profiler = ace3.MemoryProfiler()
memory_profiler.start()

trainer.train(model, dataset)

memory_stats = memory_profiler.stop()
print(f"Peak memory usage: {memory_stats.peak_memory_mb} MB")
print(f"Memory efficiency: {memory_stats.efficiency}%")
```

## Best Practices

### Data Loading

1. **Use efficient data loaders**: Optimize data pipeline
2. **Prefetch data**: Load next batch while training current batch
3. **Use multiple workers**: Parallelize data loading
4. **Cache preprocessed data**: Avoid redundant preprocessing

```python
# Optimized data loader
dataloader = ace3.DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
```

### Training Stability

1. **Use gradient clipping**: Prevent exploding gradients
2. **Monitor loss curves**: Detect training issues early
3. **Use appropriate learning rates**: Start with proven values
4. **Implement early stopping**: Prevent overfitting

### Resource Management

1. **Monitor GPU utilization**: Ensure efficient hardware usage
2. **Balance batch size and memory**: Maximize throughput
3. **Use mixed precision**: Reduce memory and increase speed
4. **Clean up resources**: Free memory when not needed

## Troubleshooting

### Common Issues

#### Out of Memory Errors

```python
# Reduce batch size
config.batch_size = 16

# Enable gradient checkpointing
config.gradient_checkpointing = True

# Use CPU offloading
config.cpu_offload = True

# Use mixed precision
config.mixed_precision = True
```

#### Slow Training

```python
# Check data loading
profiler = ace3.DataLoaderProfiler()
profiler.profile(dataloader)

# Optimize data pipeline
dataloader = ace3.optimize_dataloader(dataloader)

# Use multiple GPUs
config.num_gpus = 4

# Enable mixed precision
config.mixed_precision = True
```

#### Training Instability

```python
# Use gradient clipping
config.max_grad_norm = 1.0

# Reduce learning rate
config.learning_rate = 1e-5

# Use learning rate warmup
scheduler = ace3.schedulers.WarmupScheduler(
    optimizer, warmup_steps=1000
)

# Add regularization
config.weight_decay = 0.01
config.dropout = 0.1
```

## Performance Benchmarks

### Training Speedups

| Configuration | Baseline (hours) | Accelerated (hours) | Speedup |
|---------------|------------------|---------------------|---------|
| Single GPU    | 24.0            | 24.0               | 1.0x    |
| 4 GPUs        | 24.0            | 6.2                | 3.9x    |
| 8 GPUs        | 24.0            | 3.1                | 7.7x    |
| 8 GPUs + AMP  | 24.0            | 2.1                | 11.4x   |

### Memory Efficiency

| Technique | Memory Usage | Model Size Increase |
|-----------|--------------|-------------------|
| Baseline  | 16 GB       | 1.0x              |
| Mixed Precision | 8 GB | 2.0x              |
| Gradient Checkpointing | 12 GB | 1.3x |
| CPU Offloading | 4 GB | 4.0x |

*Benchmarks based on training BERT-Large on 8x V100 GPUs*

