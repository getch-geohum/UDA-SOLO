import os
import mmcv
from tqdm import tqdm

def changer(file):
    fls = mmcv.load(file)
    for i in tqdm(range(len(fls['annotations']))):
        fls['annotations'][i]['category_id'] = 1
    fls['categories'] = [{'supercategory': 'camp', 'id': 0, 'name': 'background'},
                         {'supercategory': 'camp', 'id': 1, 'name': 'dwelling'}]
    mmcv.dump(fls, file)
    print('done')
root = 'path to dataset with coco format root'
for fold in os.listdir(root):
    if fold == 'minawa_june_2016':
        pass
    else:
        print('Processing for {}'.format(fold))
        fold_ = root + '/' + fold
        for sub_fold in os.listdir(fold_):
            file = fold_ + f'/{sub_fold}/{sub_fold}_annotations.json'
            changer(file=file)
