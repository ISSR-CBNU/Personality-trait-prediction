# Overview of Personality Trait Prediction Using Single-Modality or Multi-Modality Approaches
This repository and project presents the implementation code and performance evaluation results of an OCEAN recognition model utilizing various AI models.

## Performance Table

|Model                        |Backcbone                       |Method                                  |1 - MAE|Input Data                       |Resolution         |Frame|
|-----------------------------|--------------------------------|----------------------------------------|-------|---------------------------------|-------------------|-----|
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|Cross-Attention (Main branch: Full-Shot)|0.914  |Query: Full-Shot Key, Value: Audio|Full-Shot : 224x224|15   |
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|Cross-Attention (Main branch: Full-Shot)|0.9139 |Query: Audio Key, Value: Full-Shot|Full-Shot : 224x224|15   |
|FAt Transformer              |R(2+1)D Pretrained with ImageNet|-                                       |0.9136 |Full-Shot                        |224x224            |15   |
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|-                                       |0.9136 |Full-Shot                        |224x224            |15   |
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|Cross-Attention (Main branch: Full-Shot)|0.9132 |Query: Text Key, Value: Full-Shot |Full-Shot : 224x224|15   |
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|2D Patch Partition                      |0.9127 |Full-Shot                        |224x224            |15   |
|Video Swin Transformer       |R(2+1)D Pretrained with ImageNet|2D Patch Partition                      |0.9127 |Face                             |224x224            |15   |
|FAt Transformer              |R(2+1)D Pretrained with ImageNet|Late Fusion                             |0.9091 |Full-Shot, Text                  |Full-Shot : 224x224|15   |
|Video Swin Transformer       |R(2+1)D Pretrained with IG-65M  |-                                       |0.9095 |Face                             |128x128            |15   |
|Video Swin Transformer       |-                               |-                                       |0.9005 |Face                             |128x128            |15   |
|Video Swin Transformer       |-                               |-                                       |0.898  |Full-Shot                        |224x224            |15   |
|Audio Spectrogram Transformer|-                               |-                                       |0.8879 |Audio                            |-                  |-    |
|ResNet101                    |-                               |-                                       |0.8836 |Full-Shot                        |224x224            |15   |
|InCeption V2                 |-                               |-                                       |0.8831 |Full-Shot                        |224x224            |15   |
|InCeption V2-LSTM            |-                               |-                                       |0.8831 |Full-Shot                        |224x224            |15   |
|ResNet101-LSTM               |-                               |-                                       |0.8826 |Full-Shot                        |224x224            |15   |
|Vgg16-LSTM                   |-                               |-                                       |0.8824 |Full-Shot                        |224x224            |15   |
|Vgg16                        |-                               |-                                       |0.8819 |Full-Shot                        |224x224            |15   |
