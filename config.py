import os
import torch

class Config:
    debug = False
    dumpfile_uniqueid = ''
    seed = 2000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    root_dir = os.path.dirname(os.path.abspath(__file__)) # ./
    data_dir = os.path.join(root_dir, 'data') # ./data
    snapshot_dir = os.path.join(root_dir, 'exp', 'snapshot')
    checkpoint_file = None # see post_updates
    
    training_datasets = {
        'Greenland26X_22W_Sentinel2_2019-06-03_05.tif': [
                        # (region, x_min, y_min, x_max, y_max)
                        (2, -5582514, 10239607, -5406608, 10548173),
                        (4, -5609072,  9571010, -5475605,  9905488),
                        (6, -2769904, 14052249, -2558152, 14319413)],
        'Greenland26X_22W_Sentinel2_2019-06-19_20.tif': [
                        (1, -5518971, 10542174, -5261090, 10875150),
                        (3, -5571762,  9917687, -5295918, 10241221),
                        (5, -2984687, 14729079, -2364804, 15625820)],
        'Greenland26X_22W_Sentinel2_2019-07-31_25.tif': [
                        (2, -5574184, 10238082, -5227924, 10543693),
                        (4, -5603308,  9565915, -5367395,  9904206),
                        (6, -3244541, 14034351, -2456623, 14717213)],
        'Greenland26X_22W_Sentinel2_2019-08-25_29.tif': [
                        (1, -5491758, 10557969, -5194109, 10864973),
                        (3, -5597210,  9915898, -5261292, 10233107),
                        (5, -3158592, 14824447, -2380378, 15515087)],
    }
    # training_datasets = {
    #     'Greenland26X_22W_Sentinel2_2019-06-03_05.tif': [
    #                     (2, -5582514, 10239607, -5406608, 10548173)]}

    
    test_datasets = {
        'Greenland26X_22W_Sentinel2_2019-06-03_05.tif': [
                        (1, -5585309, 10550998, -5221373, 10858855),
                        (3, -5632853,  9919099, -5323000, 10233766),
                        (5, -2824945, 14725163, -2372058, 15608608)],
        'Greenland26X_22W_Sentinel2_2019-06-19_20.tif': [
                        (2, -5538547, 10227706, -5242379, 10555077),
                        (4, -5559968,  9558244, -5302362,  9911194),
                        (6, -3090348, 14057036, -2449158, 14735713)],
        'Greenland26X_22W_Sentinel2_2019-07-31_25.tif': [
                        (1, -5597354, 10543191, -5173758, 10867901),
                        (3, -5661397,  9914958, -5198364, 10238170),
                        (5, -3298944, 14743477, -2389087, 15559535)],
        'Greenland26X_22W_Sentinel2_2019-08-25_29.tif': [
                        (2, -5516302, 10228305, -5180994, 10551862),
                        (4, -5531740,  9553416, -5263277,  9902308),
                        (6, -3231836, 14047125, -2444219, 14809233)],
    }
    # test_datasets = {
    #     'Greenland26X_22W_Sentinel2_2019-06-03_05.tif': [
    #                     (1, -5585309, 10550998, -5221373, 10858855),
    #                     (3, -5632853,  9919099, -5323000, 10233766)],
    #     'Greenland26X_22W_Sentinel2_2019-06-19_20.tif': [
    #                     (2, -5538547, 10227706, -5242379, 10555077),
    #                     (4, -5559968,  9558244, -5302362,  9911194)]
    # }
    
    region_file = 'lakes_regions.gpkg'
    training_region_file = 'lake_polygons_training.gpkg'
    
    backbone_str = 'u2net'
    load_checkpoint = False
    
    training_lr = 0.0005
    training_weight_decay = 0.0001
    training_momentum = 0.999
    training_lr_degrade_step = 5
    training_lr_degrade_gamma = 0.5
    training_bad_patience = 5
    training_cel_weight = 300 # crossentropyloss weight
    # training_loss = 'bce' # cel: CrossEntropyLoss, bce: BCEEithLogitLoss+DiceLoss
    training_celdice_loss_weight = 0.4 # [0, 1], bce weights, while dice weight = 1-this_value
    training_u2net_threshold = 0.9
    
    training_epochs = 20
    training_batch_size = 16
    
    image_augment = False
    image_side_len = 256
    area_threshold = 1e5 # m^2 Cartesian. confirmed


    @classmethod
    def update(cls, dic: dict):
        for k, v in dic.items():
            if k in cls.__dict__:
                assert type(getattr(Config, k)) == type(v)
            setattr(Config, k, v)
        cls.post_updates()
        return    
    
    
    @classmethod
    def post_updates(cls):
        # cls.checkpoint_file = '{}/{}_best.pt'.format(cls.snapshot_dir, cls.backbone_str)
        cls.checkpoint_file = '{}/{}_best{}.pt'.format(cls.snapshot_dir, cls.backbone_str, cls.dumpfile_uniqueid)
        return


    @classmethod
    def to_str(cls): # __str__, self
        dic = cls.__dict__.copy()
        lst = list(filter( \
                        lambda p: (not p[0].startswith('__')) and type(p[1]) != classmethod, \
                        dic.items() \
                        ))
        return '\n'.join([str(k) + ' = ' + str(v) for k, v in lst])

