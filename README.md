# FHENet-PyTorch

### The official pytorch implementation of FHENet:**Lightweight Feature Hierarchical Exploration Network for Real-Time Rail SurfaceDefect Inspection in RGB-D Images**.[[PDF](https://ieeexplore.ieee.org/document/10019291)].The model structure is as follows：

<img decodeing="async" src="https://sky-image-1302500563.cos.ap-nanjing.myqcloud.com/image/22.png" width="100%">


 
 
# Feature Maps 
Baidu [RGB-D](https://pan.baidu.com/s/1xcK303N9WScaOHdVFqsHIg?pwd=na4e)  提取码: na4e 




# Comparison of results table
Table I Evaluation metrics obtained from compared methods. The best results are shown in bold.

| Models |    Sm↑    |  maxEm↑   |  maxFm↑   |   MAE↓    |                         
| :----: | :-------: | :-------: | :-------: | :-------: |
|  DCMC  |   0.484   |   0.595   |   0.498   |   0.287   |
|  ACSD  |   0.556   |   0.670   |   0.575   |   0.360   |
|   DF   |   0.564   |   0.713   |   0.636   |   0.241   |
|  CDCP  |   0.574   |   0.694   |   0.591   |   0.236   |
|  DMRA  |   0.736   |   0.834   |   0.783   |   0.141   |
|  HAI   |   0.718   |   0.829   |   0.803   |   0.171   |
|  S2MA  |   0.775   |   0.864   |   0.817   |   0.141   |
| CONET  |   0.786   |   0.878   |   0.834   |   0.101   |
|  EMI   |   0.800   |   0.876   |   0.850   |   0.104   |
|  CSEP  |   0.814   |   0.899   |   0.866   |   0.085   |
|  EDR   |   0.811   |   0.893   |   0.850   |   0.082   |
|  BBS   |   0.828   |   0.909   |   0.867   |   0.074   |
|  DAC   |   0.824   |   0.911   |   0.875   |   0.071   |
|  CLA   |   0.835   |   0.920   |   0.878   |   0.069   |
|  Ours  | **0.836** | **0.926** | **0.881** | **0.064** |

Table II Test results of the performance of the relevant methods. The best results are shown in bold.

|  Models  |  DCMC  |  ACSD  |   DF   |  CDCP  |  DMRA  |    HAI     |  S2MA  | CONET  |  EMI   |  CSEP  |  EDR   |  BBS   |  DAC   | CLA        |    Ours    |
| :------: | :----: | :----: | :----: | :----: | :----: | :--------: | :----: | ------ | :----: | :----: | :----: | :----: | :----: | ---------- | :--------: |
| **Pre↑** | 66.16% | 55.93% | 78.88% | 73.07% | 80.36% |   73.90%   | 76.91% | 86.85% | 82.65% | 85.29% | 85.32% | 86.27% | 86.71% | **87.27%** |   87.22%   |
| **Rec↑** | 25.46% | 63.88% | 31.02% | 36.14% | 74.18% | **91.67%** | 82.83% | 78.61% | 87.76% | 87.61% | 86.60% | 87.31% | 88.09% | 86.59%     |   88.34%   |
| **F1↑**  | 33.36% | 55.65% | 42.12% | 44.98% | 74.84% |   78.98%   | 78.20% | 80.55% | 83.31% | 85.14% | 84.12% | 85.63% | 86.23% | 86.07%     | **87.01%** |
| **IOU↑** | 19.23% | 40.63% | 22.41% | 27.86% | 62.96% |   68.91%   | 70.39% | 70.57% | 74.82% | 76.65% | 75.39% | 77.27% | 77.77% | 77.87%     | **78.93%** |




# Citation

If you use CEKD in your academic work, please cite:
'''
@article{zhou2023fhenet,
  title={FHENet: Lightweight Feature Hierarchical Exploration Network for Real-Time Rail Surface Defect Inspection in RGB-D Images},
  author={Zhou, Wujie and Hong, Jiankang},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2023},
  publisher={IEEE}
}
'''
