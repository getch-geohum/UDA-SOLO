This repostory is used for [**Unsupervised Domain Adaptation for Instance Segmentation: Extracting Dwellings in Temporary Settlements Across Various Geographical Settings**](https://ieeexplore.ieee.org/document/10363437). The repository uses [SOLO](https://github.com/WXinlong/SOLO) as a base model which is based on [1](https://arxiv.org/pdf/1912.04488) and [2](https://arxiv.org/pdf/2003.10152) research works. We highly appreciate authors for their work and opensourcing their codebase. For the installation, follow the procedure provided in the [original imlamentation](https://github.com/WXinlong/SOLO/blob/master/docs/INSTALL.md). Please note that, images from all geographical areas are first converted to smaller chips using ESRI ArcGIS pro data preparation tool(see documentation [here](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm)). 
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
python automate_feature_space_plot.py
python image_simmiarity.py
```

To train a domain adaptation, run the following independently
```
python dann_runner.py
python mmd_runner.py
python ot_runner.py
python base_runner.py # or alternatively pthon ./SOLO/tools/train.py 
```
To mae inference and evaluate detections
```
python ./SOLO/tools/test_ins.py config './SOLO/configs/models/solov2train_config.py' checkpoint 'path to trained checkpoint' --out 'output path' --eval segm 
python coco_evaluate.py
```
