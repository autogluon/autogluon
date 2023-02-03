import torchmetrics.detection.mean_ap as map

class MeanAveragePrecision(map.MeanAveragePrecision):
    is_ag_copy = True