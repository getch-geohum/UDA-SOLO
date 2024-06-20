This repostory is used for [**Unsupervised Domain Adaptation for Instance Segmentation: Extracting Dwellings in Temporary Settlements Across Various Geographical Settings**](https://ieeexplore.ieee.org/document/10363437). The repository uses [SOLO](https://github.com/WXinlong/SOLO) as a base model which is based on [1](https://arxiv.org/pdf/1912.04488) and [2](https://arxiv.org/pdf/2003.10152) research works. We highly apprecitae authors for their work and opensource resurce. For the installation, follow the procedure provided in the [original imlamentation](https://github.com/WXinlong/SOLO/blob/master/docs/INSTALL.md). Please note that, images from all geographical areas are first converted to smaller chips using ESRI ArcGIS pro data preparation tool(see documentation [here](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm)). 
## USAGE
After converting function parameters inside each files, the scripts could run as:
To make spatial sampling of image chips
```
python spatial_sampling.py
```
To prepare samples to coco annotation format use

```
python label_to_coco_annotation.py
```
To generate deep feature space embeding with deep feature space simiarity, image lavel simmiarity and obect level simmiarity metrics, run the following independently

```
python generate features.py
pthon automate_feature_space_plot.py
pthon image_simmiarity.py
```

To train a domain adaptation, run the following independently
```
python dann_runner.py
pthon mmd_runner.py
python ot_runner.py
```
To mae inference and evaluate detections
```
python coco_evaluate.py
```
