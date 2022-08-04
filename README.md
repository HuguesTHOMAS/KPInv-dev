
List of upgrades compared to KPConv-Pytorch

- New pytorch implementation of grid subsampling on GPU tensors using `torch.unique` function.
- New pytorch implementation of cropped radius neighbor on GPU tensors using Keops.
- New network blocks, simplified and more standalone
- Separate block for KPConv, KPDef and KPInv blocks
- More standalone blocks for easir plug and play in other networks and tasks
- Configuration as an easyDict and save as json for readability
- Option to choose from two input pipelines:
    1. Simpler input pipeline that does not preload neighborhoods.
        * With it, a network that computes conv neighbors on the fly on GPU.
        * Faster for inference on streamed data (like segmenting lidar frames in real time).
        * Easier to use as plug and play in other types of networks that do not compute neighbors during data preparation.
        * Smaller use of CPU.
    2. Old pipeline precomputing neighbors in parallel on GPU.
        * The neighbors are computed in advance by the dataloader on CPU.
        * Much faster training or test when data is already on disk and input preparation can be parallelized.
        * Possible heavy CPU usage.
- New simplified and faster calibration for batch limit and neighbor limits.
- Additional options like feature grouping.
- New definition of architecture layers and blocks.
- Still the option
