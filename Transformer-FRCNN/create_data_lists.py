from utils import create_data_lists

if __name__ == '__main__':
    create_data_lists(voc07_path='/nvme/scratch/sweta/pascal_voc/VOCdevkit/VOC2007',
                      voc12_path='/nvme/scratch/sweta/pascal_voc/VOCdevkit/VOC2012',
                      output_folder='./')