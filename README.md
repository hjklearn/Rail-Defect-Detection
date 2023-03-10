# FHENet-PyTorch

### The official pytorch implementation of FHENet:**Lightweight Feature Hierarchical Exploration Network for Real-Time Rail SurfaceDefect Inspection in RGB-D Images**.[PDF](https://ieeexplore.ieee.org/document/10019291).

### The model structure is as follows：

<img decodeing="async" src="https://sky-image-1302500563.cos.ap-nanjing.myqcloud.com/image/22.png" width="100%">


# Feature Maps 

### Baidu [RGB-D](https://pan.baidu.com/s/1xcK303N9WScaOHdVFqsHIg?pwd=na4e)  提取码: na4e 


# Comparison of results table

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

| Models |    Pre↑    |    Rec↑    |    F1↑     |    IOU↑    |
| :----: | :--------: | :--------: | :--------: | :--------: |
|  DCMC  |   66.16%   |   25.46%   |   33.36%   |   19.23%   |
|  ACSD  |   55.93%   |   63.88%   |   55.65%   |   40.63%   |
|   DF   |   78.88%   |   31.02%   |   42.12%   |   22.41%   |
|  CDCP  |   73.07%   |   36.14%   |   44.98%   |   27.86%   |
|  DMRA  |   80.36%   |   74.18%   |   74.84%   |   62.96%   |
|  HAI   |   73.90%   | **91.67%** |   78.98%   |   68.91%   |
|  S2MA  |   76.91%   |   82.83%   |   78.20%   |   70.39%   |
| CONET  |   86.85%   |   78.61%   |   80.55%   |   70.57%   |
|  EMI   |   82.65%   |   87.76%   |   83.31%   |   74.82%   |
|  CSEP  |   85.29%   |   87.61%   |   85.14%   |   76.65%   |
|  EDR   |   85.32%   |   86.60%   |   84.12%   |   75.39%   |
|  BBS   |   86.27%   |   87.31%   |   85.63%   |   77.27%   |
|  DAC   |   86.71%   |   88.09%   |   86.23%   |   77.77%   |
|  CLA   | **87.27%** |   86.59%   |   86.07%   |   77.87%   |
|  Ours  |   87.22%   |   88.34%   | **87.01%** | **78.93%** |

