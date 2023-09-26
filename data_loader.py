import os
import sys
import time
import math
import h5py
import random
import logging
import pickle
import pandas as pd
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import Polygon, shape

import geopandas as gpd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import Config
import tools



def read_datasets():
    _time = time.time()
    gpd_file = Config.data_dir + '/' + Config.training_region_file
    df_lakes_poly = gpd.read_file(gpd_file)
    
    df_lakesout = df_lakes_poly[0:0] # remove all rows
    
    lst_img, lst_label = [], [] # list of tensor~[c, h, w]

    for tif_filename, lst_regions in Config.training_datasets.items():
        for region_meta in lst_regions:
            lakes_poly = df_lakes_poly[(df_lakes_poly['region_num'] == region_meta[0]) & \
                                        (df_lakes_poly['image'] == tif_filename)]['geometry'].tolist()
            lakes_poly = [list(p.exterior.coords) for p in lakes_poly] # [[[x1,y1], [x2, y2], ..], [], ...]
            
            _img, _lakes_label, lakes_out = read_dataset(tif_filename, region_meta[0], region_meta[1:5], lakes_poly)

            lst_img.extend(list(_img))
            lst_label.extend(list(_lakes_label))
            
            if False and tif_filename == 'Greenland26X_22W_Sentinel2_2019-08-25_29.tif' and region_meta[0] == 5:
                for p in lakes_out: # lakes_out: list of Polygon
                    df_lakesout = df_lakesout.append({'image': tif_filename, 
                                                        'region_num': region_meta[0], 
                                                        'geometry': p}, ignore_index=True)
                    df_lakesout.to_file(gpd_file[:-5]+'_out.gpkg', layer='lakes', driver="GPKG")
    
    # split and shuffle
    pairs = list(zip(lst_img, lst_label))
    random.shuffle(  pairs  )
    lst_img, lst_label = zip(*pairs)
    lst_img, lst_label = list(lst_img), list(lst_label)
    
    logging.info("[Done-Read all datasets] @={:.2f}, #img={}, #label={}".format( \
                    time.time() - _time, len(lst_img), len(lst_label)))
    return lst_img, lst_label
    
    

# read each specific region -- its raw tif, and labelled images with lake polygons
def read_dataset(dataset_file, region_id, region_range, lakes_poly):
    # lakes_poly = [ [(x1,y1), (x2, y2), ...] ]
    _time = time.time()
    logging.debug('[Start] @={:.0f}'.format(_time))
    
    dataset_dir = Config.data_dir + '/' + dataset_file # tif file
    x_min, y_min, x_max, y_max = region_range # lonlat
    lst_lakes = []
    
    with rasterio.open(dataset_dir) as tif:
        logging.debug("[Read tif] {}, ({},{},{}), {}, {}, {}".format( \
                        dataset_file, tif.count, tif.height, tif.width, tif.crs, tif.res, tif.bounds))

        region_rows, region_cols = rasterio.transform.rowcol(tif.transform, [x_min, x_max], [y_max, y_min])
        # img.shape = [3, height, width], ndarray
        img = tif.read(window = Window(region_cols[0], region_rows[0], \
                                        region_cols[1]-region_cols[0]+1, \
                                        region_rows[1]-region_rows[0]+1), fill_value = 255)
        logging.debug("[Read region] {}, xy_range=({},{},{},{}), rows={}, cols={}, img.shape={}".format( \
                        region_id, x_min, x_max, y_min, y_max, region_rows, region_cols, img.shape))
        
        # Prepare lake data
        # 1. convert lake's polys to rasters; 
        # 2. fill rasters (i.e., labels) to empty canvas
        # lakes_label = np.zeros((img.shape[1], img.shape[2]), dtype = np.uint8) # [height, width]
        lakes_label = np.zeros((tif.height, tif.width), dtype = np.uint8) # [height, width]
        
        for poly in lakes_poly:
            _xs, _ys = zip(*poly)
            _rowcol = rasterio.transform.rowcol(tif.transform, _xs, _ys)
            # TODO: assert valid rows or cols
            _row_min, _row_max = min(_rowcol[0]), max(_rowcol[0]) # rowcol in the whole canvas
            _col_min, _col_max = min(_rowcol[1]), max(_rowcol[1])
            _row_size = _row_max - _row_min + 1
            _col_size = _col_max - _col_min + 1
            _arr = np.array(_rowcol) - [[_row_min], [_col_min]] # [2, n]
            _arr = np.fliplr(_arr.T) # [n, 2]
            _rast = rasterize([Polygon(_arr)], out_shape = (_row_size, _col_size))
            logging.debug('{} {} {} {}'.format(_row_min, _row_max, _col_min, _col_max))
            lakes_label[_row_min: _row_max+1, _col_min: _col_max+1] = _rast

        if False and dataset_file == 'Greenland26X_22W_Sentinel2_2019-08-25_29.tif' and region_id == 5:
            for c, v in rasterio.features.shapes(lakes_label, connectivity = 4, transform = tif.transform):
                g = shape(c)
                lst_lakes.append(g)
            lst_lakes.pop(-1)
        
        lakes_label = lakes_label[region_rows[0]: region_rows[1]+1, region_cols[0]: region_cols[1]+1]
        
    # split img and lakes_label into patches
    side = Config.image_side_len
    img = torch.from_numpy(img).float().unsqueeze(0) # [1, 3, h, w]
    img = tools.unfold4D_with_padding(img, side, 255)
    img = tools.image_norm(img)
    
    lakes_label = torch.from_numpy(lakes_label).long().unsqueeze(0).unsqueeze(0) # [1, 1, h, w]
    lakes_label = tools.unfold4D_with_padding(lakes_label, side, 0)

    logging.info('[Done] @={:.2f}, dataset={}, region_id={}, #lakes_poly={}, #lst_lakes={}, img.shape={}, lakes_label.shape={}'.format( \
                time.time() - _time, dataset_file, region_id, len(lakes_poly), len(lst_lakes), img.shape, lakes_label.shape))
    
    logging.debug('weights -- {:.0f}'.format( np.prod(lakes_label.shape)/lakes_label.sum() ))

    return img, lakes_label, lst_lakes # Tensors; image=[batch, 3, height, width], lakes_label=[batch, 1, height, width] 


def read_test_dataset():
    _time = time.time()
    
    dic_test = {}
    
    for tif_filename, lst_regions in Config.test_datasets.items():
        dic_test[tif_filename] = {}
        for region_meta in lst_regions:
            region_id = region_meta[0]
            region_range = region_meta[1:5]
            
            tif_dir = Config.data_dir + '/' + tif_filename # tif file
            x_min, y_min, x_max, y_max = region_range # lonlat
        
            with rasterio.open(tif_dir) as tif:
                logging.debug("[Read tif] {}, ({},{},{}), {}, {}, {}".format( \
                                tif_dir, tif.count, tif.height, tif.width, tif.crs, tif.res, tif.bounds))

                region_rows, region_cols = rasterio.transform.rowcol(tif.transform, [x_min, x_max], [y_max, y_min])
                img = tif.read(window = Window(region_cols[0], region_rows[0], \
                                                region_cols[1]-region_cols[0]+1, \
                                                region_rows[1]-region_rows[0]+1), fill_value = 255)
                
                img = torch.from_numpy(img).float().unsqueeze(0) # [1, 3, h, w]
                img = tools.unfold4D_with_padding(img, Config.image_side_len, 255) # [n, 3, sidelen, sidelen]
                img = tools.image_norm(img)
                lst_img = list(img) # list of [3, sidelen, sidelen]
                
                dic_test[tif_filename][region_id] = (lst_img, region_rows, region_cols)
                logging.debug("[Read region] {}, xy_range=({},{},{},{}), rows={}, cols={}, img.shape={}".format( \
                                region_id, x_min, x_max, y_min, y_max, region_rows, region_cols, img.shape))

    logging.info("[Done-Read test datasets]. @={:.2f}, #regions={}, #imgs={}".format( \
                    time.time() - _time, sum([len(v) for v in dic_test.values()]),
                    sum([len(vv[0]) for v in dic_test.values() for vv in v.values()]) ))
    return dic_test



def output_testresults(dic_preds):
    # lst_preds: dict of tensor~[1, 1, realh, realw], each tensor is for one region
    _time = time.time()
    logging.debug('[Start] @={:.0f}'.format(_time))
    # 1. read a dummy img, for making use of its space
    # 2. read training gpk, for making use of its file format
    # 3. read 6 region polygons, for verifying outputs
    
    gpd_file = Config.data_dir + '/' + Config.training_region_file
    df_lakes_poly = gpd.read_file(gpd_file)
    df_lakes_poly = df_lakes_poly[0:0] # remove all rows
    
    gpd_file = Config.data_dir + '/' + Config.region_file
    df_regions_poly = gpd.read_file(gpd_file)
    
    tif, tif_h, tif_w = __read_dummy_tif()
    
    for tif_filename, lst_regions in Config.test_datasets.items():
        for region_meta in lst_regions:
            region_id = region_meta[0]
            region_range = region_meta[1:5]
            region_poly = df_regions_poly[df_regions_poly['region_num'] == region_id]['geometry'].item()
            
            # tif_dir = Config.data_dir + '/' + tif_filename # tif file
            # x_min, y_min, x_max, y_max = region_range # lonlat
            
            pred, region_rows, region_cols = dic_preds[tif_filename][region_id]
            
            whole_img = torch.zeros((tif_h, tif_w), dtype = torch.uint8)
            whole_img[region_rows[0]: region_rows[1]+1, region_cols[0]: region_cols[1]+1] += pred
            # whole_img[whole_img>=1] = 1
            whole_img = whole_img.numpy()
                
            lst_lakes = []
            for c, v in rasterio.features.shapes(whole_img, connectivity = 4, transform = tif.transform):
                g = shape(c)
                lst_lakes.append(g)
            lst_lakes.pop(-1)
        
            for p in lst_lakes: # lakes_out: list of Polygon
                if p.within(region_poly):
                    df_lakes_poly = df_lakes_poly.append({'image': tif_filename, 
                                                                'region_num': region_id, 
                                                                'geometry': p}, ignore_index=True)
            n_df_lakes_poly_ = df_lakes_poly[(df_lakes_poly['image']==tif_filename) & (df_lakes_poly['region_num']==region_id)].shape[0]
            logging.info("#lst_lake={}, #df_lakes_poly={}".format(len(lst_lakes), n_df_lakes_poly_))
    
    if df_lakes_poly.size:
        df_lakes_poly.to_file(Config.data_dir + '/lake_polygons_test{}.gpkg'.format(Config.dumpfile_uniqueid), layer='lakes', driver="GPKG")
        logging.info("[Output testresult] Done. @={:.2f}, #lakes_poly={}".format(time.time() - _time, df_lakes_poly.shape[0]))
    else:  
        logging.error("[Output testresult] Fail. No output lake results. @={:.2f}".format(time.time() - _time))
        
    return 


def __read_dummy_tif():
    tif_filename = next(iter(Config.test_datasets.keys()))
    tif_dir = Config.data_dir + '/' + tif_filename # tif file
    
    # region_meta = lst_regions[0]
    # region_id = region_meta[0]
    # region_range = region_meta[1:5]
            
    tif_dir = Config.data_dir + '/' + tif_filename # tif file
    # x_min, y_min, x_max, y_max = region_range # lonlat
    
    tif = rasterio.open(tif_dir)
    logging.debug("[Read tif] {}, ({},{},{}), {}, {}, {}".format( \
                    tif_dir, tif.count, tif.height, tif.width, tif.crs, tif.res, tif.bounds))
    return tif, tif.height, tif.width
    


class PairDataset(Dataset):
    def __init__(self, a, b):
        assert len(a) == len(b)
        self.a = a
        self.b = b

    def __getitem__(self, index):
        return self.a[index], self.b[index]

    def __len__(self):
        return len(self.a)

   
   

if __name__ == '__main__':
    logging.basicConfig(level = logging.DEBUG if Config.debug else logging.INFO,
            format = "[%(filename)s:%(lineno)s %(funcName)s()] -> %(message)s",
            handlers = [logging.FileHandler(Config.root_dir+'/exp/log/'+tools.log_file_name(), mode = 'w'), 
                        logging.StreamHandler()] )

    logging.info('python ' + ' '.join(sys.argv))
    logging.info('=================================')
    logging.info(Config.to_str())
    logging.info('=================================')

    read_datasets()