import mmcv
import argparse

def changeWeight(conf, weight):
    print('weight obtained: {}'.format(weight))
    print(mmcv.__version__)
    yy = mmcv.Config.fromfile(conf)
    yy.resume_from = weight
    print(dir(yy))
    mmcv.Config.dump(yy, conf)
    print('Weight changed to: {}'.format(weight))
    
def argumentParser():
    parser = argparse.ArgumentParser(description = 'on the flight weight change')
    parser.add_argument('--conf', help='path to config file', type=str)
    parser.add_argument('--weight', help = 'trained weight file', type = str)
    arg = parser.parse_args()
    return arg

if __name__ == "__main__":
    args = argumentParser()
    changeWeight(conf=args.conf, weight=args.weight)
