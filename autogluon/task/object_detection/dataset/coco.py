from .base import DatasetBase
import autogluon as ag

from gluoncv import data as gdata

@ag.autogluon_object(
    nstage1=ag.Int(2, 4),
    nstage2=ag.Int(2, 4),
)
class COCO(DatasetBase):
    def __init__(self):
        super(COCO. self).__init__()
        self.train_dataset = gdata.COCODetection(splits='instances_train2017')
        self.val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        self.val_metric = COCODetectionMetric(
                                self.val_dataset, args.save_prefix + '_eval', cleanup=True,
                                data_shape=(args.data_shape, args.data_shape))
        
        #TODO: whether to use the code below
        """
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
        """
    
    def get_train_val_metric(self):
        return (self.train_dataset, self.val_dataset, self.val_metric)
    
    def get_dataset_name(self):
        return 'coco'






