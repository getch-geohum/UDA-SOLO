This repostory is used for [**Unsupervised Domain Adaptation for Instance Segmentation: Extracting Dwellings in Temporary Settlements Across Various Geographical Settings**](https://ieeexplore.ieee.org/document/10363437). The repository uses [SOLO](https://github.com/WXinlong/SOLO) as a base model which is based on [1](https://arxiv.org/pdf/1912.04488) and [2](https://arxiv.org/pdf/2003.10152) research works. We highly appreciate authors for their work and opensourcing their codebase. For the installation, follow the procedure provided in the [original imlamentation](https://github.com/WXinlong/SOLO/blob/master/docs/INSTALL.md). Please note that, images from all geographical areas are first converted to smaller chips using ESRI ArcGIS pro image analyst deep learning data preparation tool(see documentation [here](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm)). 
## USAGE 
After converting function parameters inside each files, the scripts could run as: <br />
To make spatial sampling of image chips <br />
```
python /scripts/spatial_sampling.py
```
To prepare samples to coco annotation format use
```
python /scripts/label_to_coco_annotation.py 
```
To generate deep feature space embeding with deep feature space simiarity, image lavel simmiarity and obect level simmiarity metrics, run the following independently

```
python /scripts/generate_features.py
python /scripts/automate_feature_space_plot.py
python /scripts/image_simmiarity.py   # final outputs need summary and visualization
python /scripts/object_simmilarity # final outputs need summary and visualization
```

To train a domain adaptation, run the following independently
```
python /scripts/dann_runner.py
python /scripts/mmd_runner.py
python /scripts/ot_runner.py
python /scripts/base_runner.py # or alternatively pthon ./SOLO/tools/train.py  # During base training shared encoder should be chnaged to sinle encoder
```
To make inference and evaluate detections
```
python ./SOLO/tools/test_ins.py config './SOLO/configs/models/solov2train_config.py' checkpoint 'path to trained checkpoint' --out 'output path' --eval segm 
python /scripts/coco_evaluate.py or generate accurac results together with "python ./SOLO/tools/test_ins.py"
```
