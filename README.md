# Personality-trait-prediction

## Performance Table

| Model                         | Backcbone                        | Method             | 1 - MAE | Input Data | Resolution | Frame |
|-------------------------------|----------------------------------|--------------------|---------|------------|------------|-------|
| Vgg16                         | -                                | -                  | 0.8819  | Full-Shot  | 224x224    | 15    |
| ResNet101                     | -                                | -                  | 0.8836  | Full-Shot  | 224x224    | 15    |
| InCeption V2                  | -                                | -                  | 0.8831  | Full-Shot  | 224x224    | 15    |
| Vgg16-LSTM                    | -                                | -                  | 0.8824  | Full-Shot  | 224x224    | 15    |
| ResNet101-LSTM                | -                                | -                  | 0.8826  | Full-Shot  | 224x224    | 15    |
| InCeption V2-LSTM             | -                                | -                  | 0.8831  | Full-Shot  | 224x224    | 15    |
| Video Swin Transformer        | -                                | -                  | 0.898   | Full-Shot  | 224x224    | 15    |
|                               |                                  | -                  | 0.9005  | Face       | 128x128    | 15    |
|                               | R(2+1)D Pretrained with ImageNet |                    | 0.9136  | Full-Shot  | 224x224    | 15    |
|                               | R(2+1)D Pretrained with IG-65M   |                    | 0.9095  | Face       | 128x128    | 15    |
|                               | R(2+1)D Pretrained with ImageNet | 2D Patch Partition | 0.9127  | Full-Shot  | 224x224    | 15    |
|                               | R(2+1)D Pretrained with ImageNet |                    | 0.9127  | Face       | 224x224    | 15    |
|                               | R(2+1)D Pretrained with IG-65M   | Forced Attention   |         | Full-Shot  | 224x224    | 15    |
|                               | R(2+1)D Pretrained with IG-65M   |                    |         | Face       | 128x128    | 15    |
|                               |                                  |                    |         |            |            |       |
| Audio Spectrogram Transformer | -                                | -                  | 0.889   | Audio      | -          | -     |