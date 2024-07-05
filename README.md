This repostory is used for [**Unsupervised Domain Adaptation for Instance Segmentation: Extracting Dwellings in Temporary Settlements Across Various Geographical Settings**](https://ieeexplore.ieee.org/document/10363437). The repository implements three unsupervised domain adaptation techniques[Maximum Mean Discrepancy (Borgwardt et al.(2006))](), [Domain Adversarial training of Neural Network Ganin et al.(2016)](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf) and [Deep Joint Distribution Optimal Transport(Damodaran et al. (2018)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Bharath_Bhushan_Damodaran_DeepJDOT_Deep_Joint_ECCV_2018_paper.pdf). The repository uses [SOLO](https://github.com/WXinlong/SOLO) as a base model which is based on [Wang et al.(2020)](https://arxiv.org/pdf/1912.04488) and [Wang et al.(2022)](https://arxiv.org/pdf/2003.10152) research works(see also their source code [here](https://github.com/WXinlong/SOLO). We highly appreciate authors for their work and opensourcing their codebase. For the installation, follow the procedure provided in the [original imlementation](https://github.com/WXinlong/SOLO/blob/master/docs/INSTALL.md). Please note that, images from all geographical areas are first converted to smaller chips using ESRI ArcGIS pro image analyst deep learning data preparation tool(see documentation [here](https://pro.arcgis.com/en/pro-app/latest/tool-reference/image-analyst/export-training-data-for-deep-learning.htm)). 
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
### Citation 
```
@article{gella2023unsupervised,
  title={Unsupervised Domain Adaptation for Instance Segmentation: Extracting Dwellings in Temporary Settlements Across Various Geographical Settings},
  author={Gella, Getachew Workineh and Pelletier, Charlotte and Lef{\`e}vre, S{\'e}bastien and Wendt, Lorenz and Tiede, Dirk and Lang, Stefan},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2023},
  publisher={IEEE}
}
```
