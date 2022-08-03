
List of upgrades compared to KPConv-Pytorch

- New network blocks, simplified and more standalone
- Separate block for KPConv, KPDef and KPInv blocks
- More standalone blocks for easir plug and play in other networks and tasks
- Configuration as an easyDict and save as json for readability
- Option to choose from two input pipelines:
    1. Simpler input pipeline that does not preload neighborhoods.
    2. With it, a network that computes conv neighbors on the fly on GPU
    3. faster for 
- Still the option
