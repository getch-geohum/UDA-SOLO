import os
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import numpy as np

def re_evaluate(gt_json, dt_json, outfile, eval_type='segm'):
    assert eval_type in ['segm','bbox','keypoints'], 'the evaluation type shoul be either of "segm", "bbox" or "keypoints" '
    prefix = 'person_keypoints' if eval_type=='keypoints' else 'instances'
    print('Running demo for checking the results.')
    cocoGt = COCO(gt_json)
    cocoDt = cocoGt.loadRes(dt_json)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt,cocoDt,eval_type)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    print('saving metris to the file')
    print(cocoEval.stats)
    keys = ["segm_mAP", "segm_mAP_50", "segm_mAP_75", "segm_mAP_s", "segm_mAP_m", "segm_mAP_l"]
    vals = [round(cocoEval.stats[i], 4) for i in range(6)]
    metrics =  {}
    for i in range(len(keys)):
        metrics[keys[i]] = vals[i]
    metrics['copy_paste'] = vals

    np.save(outfile, metrics)   #### save as numpy file


def argumentParser():
    parser = argparse.ArgumentParser(description="batch COCO revaluation using detected and ground truth json files")
    parser.add_argument("--data_root", help="Root path that contained root data",type=str, required=True)
    parser.add_argument('--eval', help="Evaluation type to generat matrices either bbox, segm or keypoints", type=str, default='segm')
    vals = parser.parse_args()
    return vals

if __name__ == "__main__":
    args = argumentParser()
    source = os.listdir(args.data_root)
    for sfold in source:
        for tfold in target:
            if sfold =!tfold:
                pred = f'{data_root}/{sfold}/{tfold}/outs/result_segm.json'
                src = args.data_root + f'/{tfold}/test_annotations.json'
                out_dir = f'{data_root}/{sfold}/{tfold}/outs/summary_metric'
    
                if os.path.exists(out_dir):
                    pass
                else:
                    if (os.path.exists(pred) and os.path.exists(src)):
                        try:
                            re_evaluate(src, pred, out_dir, args.eval)
                        except:
                            pass
                    else:
                        print('either: {src} or {pred} not exist') 

