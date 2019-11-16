# coding:utf8
# author: Gofinge


class DefaultConfig(object):
    data_root = '/Users/gofinge/Documents/Data/ILD_DB'
    lung_mask = 'ILD_DB_lungMasks'
    roi_mask = 'ILD_DB_volumeROIs'
    roi_txt = 'ILD_DB_txtROIs'

    labels_list = ['healthy', 'emphysema', 'ground_glass', 'fibrosis', 'micronodules', 'consolidation',
                   'bronchial_wall_thickening', 'reticulation', 'macronodules', 'cysts', 'peripheral_micronodules',
                   'bronchiectasis', 'air_trapping', 'early_fibrosis', 'increased_attenuation', 'tuberculosis', 'pcp']

    labels = [1, 3, 4, 5, 6, 8]
    colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [139, 0, 0], [0, 0, 139], [139, 0, 139], [144, 238, 144],
              [0, 139, 139], [155, 48, 288], [255, 62, 150], [255, 165, 0], [255, 211, 155], [255, 193, 37],
              [255, 255, 0], [192, 255, 62], [0, 255, 255], [153, 50, 204], [255, 162, 0], [50, 205, 50], [0, 255, 255],
              [47, 79, 79], [119, 136, 153], [25, 25, 112], [123, 104, 238], [135, 206, 250], [0, 100, 0],
              [173, 255, 47], [188, 143, 143], [250, 128, 114], [205, 102, 29], [205, 51, 51], [205, 16, 118]]

    CUDA_VISIBLE_DEVICES = 0


if __name__ == '__main__':
    opt = DefaultConfig()
    print(opt.labels)
    print(opt.CUDA_VISIBLE_DEVICES)


