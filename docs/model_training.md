# Model Training Guide

This guide explains how to train custom face detection and recognition models for the Face Scan Project.

## Overview

The Face Scan Project supports multiple model types:
- **Face Detection Models**: Detect faces in images
- **Face Recognition Models**: Recognize and identify faces
- **Custom Models**: Train models on your specific dataset

## Prerequisites

### Data Requirements
- Minimum 100 images per person for recognition training
- Diverse lighting conditions and angles
- High-quality images (minimum 224x224 pixels)
- Balanced dataset (similar number of images per person)

### Hardware Requirements
- GPU with CUDA support (recommended)
- 16GB+ RAM
- 50GB+ free disk space
- Multi-core CPU

### Software Requirements
- Python 3.8+
- TensorFlow 2.x
- OpenCV
- scikit-learn
- matplotlib

## Data Preparation

### 1. Dataset Structure

Organize your dataset in the following structure:
```
data/
├── raw/
│   ├── training/
│   │   ├── person1/
│   │   │   ├── image1.jpg
│   │   │   ├── image2.jpg
│   │   │   └── ...
│   │   ├── person2/
│   │   │   ├── image1.jpg
│   │   │   └── ...
│   │   └── ...
│   └── validation/
│       ├── person1/
│       ├── person2/
│       └── ...
```

### 2. Data Preprocessing

Run the data preprocessing script:
```bash
python scripts/data_preprocessing.py \
    --input_dir data/raw/training \
    --output_dir data/processed/training \
    --resize 224 \
    --augment
```

### 3. Data Validation

Validate your dataset:
```bash
python scripts/validate_dataset.py \
    --dataset_dir data/processed/training \
    --min_images_per_person 50
```

## Training Face Detection Models

### 1. Using Pre-trained Models

Fine-tune a pre-trained face detection model:
```bash
python scripts/train_model.py \
    --model_type detection \
    --base_model mobilenet_v2 \
    --dataset_dir data/processed/training \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### 2. Training from Scratch

Train a custom face detection model:
```bash
python scripts/train_model.py \
    --model_type detection \
    --architecture custom_cnn \
    --dataset_dir data/processed/training \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.01
```

### 3. Model Configuration

Create a custom model configuration:
```python
# config/model_config.py
DETECTION_MODEL_CONFIG = {
    'input_shape': (224, 224, 3),
    'num_classes': 1,  # face/no-face
    'architecture': 'mobilenet_v2',
    'pretrained': True,
    'freeze_layers': 10,
    'dropout_rate': 0.5,
    'regularization': 'l2'
}
```

## Training Face Recognition Models

### 1. Embedding-based Recognition

Train a face embedding model:
```bash
python scripts/train_model.py \
    --model_type recognition \
    --architecture facenet \
    --dataset_dir data/processed/training \
    --epochs 200 \
    --batch_size 64 \
    --embedding_size 128
```

### 2. Classification-based Recognition

Train a face classification model:
```bash
python scripts/train_model.py \
    --model_type recognition \
    --architecture resnet50 \
    --dataset_dir data/processed/training \
    --epochs 100 \
    --batch_size 32 \
    --num_classes 100  # number of people
```

### 3. Triplet Loss Training

Train with triplet loss for better embeddings:
```bash
python scripts/train_model.py \
    --model_type recognition \
    --loss_function triplet \
    --dataset_dir data/processed/training \
    --epochs 300 \
    --batch_size 32 \
    --margin 0.5
```

## Training Configuration

### 1. Hyperparameters

Configure training hyperparameters:
```python
# config/training_config.py
TRAINING_CONFIG = {
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'learning_rate_schedule': 'cosine_decay',
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.2,
    'early_stopping': {
        'patience': 10,
        'monitor': 'val_loss'
    },
    'data_augmentation': {
        'rotation_range': 20,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'brightness_range': [0.8, 1.2]
    }
}
```

### 2. Callbacks

Configure training callbacks:
```python
CALLBACKS = [
    'model_checkpoint',
    'early_stopping',
    'learning_rate_scheduler',
    'tensorboard_logging'
]
```

## Model Evaluation

### 1. Validation Metrics

Evaluate model performance:
```bash
python scripts/evaluate_model.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --test_dataset data/processed/validation \
    --metrics accuracy precision recall f1
```

### 2. Performance Analysis

Analyze model performance:
```bash
python scripts/analyze_performance.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --test_dataset data/processed/validation \
    --output_dir results/analysis
```

### 3. Confusion Matrix

Generate confusion matrix:
```bash
python scripts/generate_confusion_matrix.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --test_dataset data/processed/validation \
    --output_file results/confusion_matrix.png
```

## Model Optimization

### 1. Quantization

Quantize model for faster inference:
```bash
python scripts/quantize_model.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --output_path data/models/face_recognition/quantized_model.tflite \
    --quantization_type int8
```

### 2. Pruning

Prune model to reduce size:
```bash
python scripts/prune_model.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --output_path data/models/face_recognition/pruned_model.h5 \
    --pruning_ratio 0.3
```

### 3. Model Conversion

Convert to different formats:
```bash
# Convert to TensorFlow Lite
python scripts/convert_model.py \
    --input_path data/models/face_recognition/best_model.h5 \
    --output_path data/models/face_recognition/model.tflite \
    --format tflite

# Convert to ONNX
python scripts/convert_model.py \
    --input_path data/models/face_recognition/best_model.h5 \
    --output_path data/models/face_recognition/model.onnx \
    --format onnx
```

## Training Monitoring

### 1. TensorBoard

Monitor training with TensorBoard:
```bash
tensorboard --logdir logs/training
```

### 2. Custom Metrics

Track custom metrics:
```python
# Add custom metrics to training
custom_metrics = {
    'face_detection_accuracy': face_detection_accuracy,
    'recognition_precision': recognition_precision,
    'inference_time': inference_time
}
```

### 3. Model Checkpointing

Configure model checkpointing:
```python
CHECKPOINT_CONFIG = {
    'save_best_only': True,
    'monitor': 'val_accuracy',
    'mode': 'max',
    'save_freq': 'epoch',
    'filepath': 'data/models/face_recognition/best_model.h5'
}
```

## Advanced Training Techniques

### 1. Transfer Learning

Use transfer learning for better performance:
```bash
python scripts/train_model.py \
    --model_type recognition \
    --base_model vgg16 \
    --transfer_learning \
    --freeze_base_layers \
    --fine_tune_layers 3
```

### 2. Multi-task Learning

Train models for multiple tasks:
```bash
python scripts/train_model.py \
    --model_type multitask \
    --tasks detection recognition \
    --dataset_dir data/processed/training \
    --epochs 150
```

### 3. Adversarial Training

Use adversarial training for robustness:
```bash
python scripts/train_model.py \
    --model_type recognition \
    --adversarial_training \
    --adversarial_ratio 0.1 \
    --dataset_dir data/processed/training
```

## Model Deployment

### 1. Model Packaging

Package trained model:
```bash
python scripts/package_model.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --output_path data/models/face_recognition/package.zip \
    --include_preprocessing
```

### 2. Model Testing

Test deployed model:
```bash
python scripts/test_deployed_model.py \
    --model_path data/models/face_recognition/package.zip \
    --test_images data/samples/test_images
```

### 3. Performance Benchmarking

Benchmark model performance:
```bash
python scripts/benchmark_model.py \
    --model_path data/models/face_recognition/best_model.h5 \
    --benchmark_dataset data/benchmark \
    --output_file results/benchmark.json
```

## Best Practices

### 1. Data Quality
- Use high-quality, diverse images
- Ensure proper labeling
- Balance your dataset
- Augment data for better generalization

### 2. Training Strategy
- Start with pre-trained models
- Use appropriate learning rates
- Monitor overfitting
- Use validation sets

### 3. Model Selection
- Choose appropriate architecture
- Consider inference speed requirements
- Balance accuracy vs. performance
- Test on real-world data

### 4. Evaluation
- Use multiple metrics
- Test on unseen data
- Perform cross-validation
- Analyze failure cases

## Troubleshooting

### Common Training Issues

**1. Overfitting**
- Increase regularization
- Use data augmentation
- Reduce model complexity
- Increase training data

**2. Underfitting**
- Increase model complexity
- Reduce regularization
- Increase training time
- Check data quality

**3. Slow Training**
- Use GPU acceleration
- Increase batch size
- Optimize data pipeline
- Use mixed precision

**4. Memory Issues**
- Reduce batch size
- Use gradient checkpointing
- Optimize data loading
- Use model parallelism

## Resources

- [TensorFlow Training Guide](https://www.tensorflow.org/guide/training)
- [Face Recognition Research Papers](https://paperswithcode.com/task/face-recognition)
- [Computer Vision Best Practices](https://github.com/microsoft/ComputerVision)
- [Model Optimization Techniques](https://www.tensorflow.org/lite/performance)

## Support

For training support:
- Email: training@facescan.com
- GitHub Issues: https://github.com/yourusername/face-scan-project/issues
- Documentation: https://docs.facescan.com
