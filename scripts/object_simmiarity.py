import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
import geopandas as gpd
from shapely.geometry import box
import json

def SHI(a=15, p=70):
    '''Computes shape index of an object'''
    return round((0.25*p/(math.sqrt(a)+1e-6)), 3)

def FRD(a=15, p=70):
    '''Conputes fractal dimension of an object, '''
    return round((2*math.log(0.25*p))/(math.log(a)+1e-6), 3)

def MinDist(gdf):
    '''gdf: is the geopandas geoseries of all polygons per patch'''
    try:
        mdist = [min([gdf[j].distance(gdf[i]) for i in range(len(gdf)) if i != j]) for j in range(len(gdf)) ]
    except:
        mdist = [0]
    return mdist
  
def objectMetric(file):
    with rasterio.open(file) as src:
        profile = src.profile
        bounds = src.bounds
        
    im = imread(file)
    img_msk = im.astype(np.uint8)
    contours = measure.find_contours(img_msk, 0.5)
    
    polys = []
    areas = []
    lengths = []
    if len(contours)>=1:
        for cont in contours:
            if cont.shape[0]<3:
                pass
            else:
                poly = Polygon(cont)
                poly = poly.simplify(1.0, preserve_topology=False)
                if poly.is_empty or not poly.is_valid:
                    pass
                else:
                    polys.append(poly)
                    areas.append(poly.area)
                    lengths.append(poly.length)
                
    if len(polys)>=1:
        chip_area = box(*bounds).area
        density = len(areas)/chip_area
        numb_obj = len(areas)
        ngdf = gpd.GeoSeries(polys, crs=profile['crs'])
        mndist = MinDist(ngdf)  # minimum distance to neighbour 
        shape_index = [SHI(a=P[0], p=P[1]) for P in list(zip(areas, lengths))]
#         frac_dim = [FRD(a=P[0], p=P[1]) for P in list(zip(areas, lengths))]
        return {'are':areas,'per':lengths, 'den':density, 'num':numb_obj,'mnd':mndist, 'shi':shape_index} # 'frd':frac_dim
    else:
        return {'are':[],'per':[], 'den':0, 'num':0,'mnd':[], 'shi':[]}  # 'frd':[]


def datasetObjMetric(files):
    are = []
    per = []
    num = []
    den = []
    mnd = []
    shi = []
#     frd = []
    for file in files:
        R = objectMetric(file)
        are += R['are']
        per += R['per']
        num.append(R['num'])
        den.append(R['den'])
        mnd += R['mnd']
        shi += R['shi']
#         frd += R['frd']
    return {'are':are,'per':per, 'den':den, 'num':num,'mnd':mnd, 'shi':shi} # 'frd':frd

if name == "__main__":
  file_path = input("Diretor where the image labels exist:")
  out_path = input("folder to save obect metric results: ")
  root = '{file_path}/{}/labels/*.tif'
  main_data = {}
  for fold in os.listdir(file_path):
      files = glob(root.format(fold))
      res = datasetObjMetric(files=files)
      main_data[fold] = res

  with open(f"out_path/raw_data.json","w") as f:
    json.dump(main_data,f)
