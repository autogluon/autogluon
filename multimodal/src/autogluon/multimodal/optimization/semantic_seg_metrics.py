# -*- coding: utf-8 -*-

import numpy as np
import torch
import torchmetrics
from scipy.ndimage import convolve
from scipy.ndimage import distance_transform_edt as bwdist

_EPS = np.spacing(1)  # the different implementation of epsilon (extreme min value) between numpy and matlab
_TYPE = np.float64


def _prepare_data(pred: np.ndarray, gt: np.ndarray) -> tuple:
    """
    A numpy-based function for preparing ``pred`` and ``gt``.
    - for ``pred``, it looks like ``mapminmax(im2double(...))`` of matlab;
    - ``gt`` will be binarized by 128.
    :param pred: prediction
    :param gt: mask
    :return: pred, gt
    """
    gt = gt > 128
    # im2double, mapminmax
    pred = pred / 255
    if pred.max() != pred.min():
        pred = (pred - pred.min()) / (pred.max() - pred.min())
    return pred, gt


def _get_adaptive_threshold(matrix: np.ndarray, max_value: float = 1) -> float:
    """
    Return an adaptive threshold, which is equal to twice the mean of ``matrix``.
    :param matrix: a data array
    :param max_value: the upper limit of the threshold
    :return: min(2 * matrix.mean(), max_value)
    """
    return min(2 * matrix.mean(), max_value)


class Fmeasure(object):
    def __init__(self, beta: float = 1.0):
        """
        F-measure for SOD.
        ::
            @inproceedings{Fmeasure,
                title={Frequency-tuned salient region detection},
                author={Achanta, Radhakrishna and Hemami, Sheila and Estrada, Francisco and S{\"u}sstrunk, Sabine},
                booktitle=CVPR,
                number={CONF},
                pages={1597--1604},
                year={2009}
            }
        :param beta: the weight of the precision
        """
        self.beta = beta
        self.precisions = []
        self.recalls = []
        self.adaptive_fms = []
        self.changeable_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        adaptive_fm = self.cal_adaptive_fm(pred=pred, gt=gt)
        self.adaptive_fms.append(adaptive_fm)

        precisions, recalls, changeable_fms = self.cal_pr(pred=pred, gt=gt)
        self.precisions.append(precisions)
        self.recalls.append(recalls)
        self.changeable_fms.append(changeable_fms)

    def cal_adaptive_fm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive F-measure.
        :return: adaptive_fm
        """
        # ``np.count_nonzero`` is faster and better
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        binary_predcition = pred >= adaptive_threshold
        area_intersection = binary_predcition[gt].sum()
        if area_intersection == 0:
            adaptive_fm = 0
        else:
            pre = area_intersection / np.count_nonzero(binary_predcition)
            rec = area_intersection / np.count_nonzero(gt)
            adaptive_fm = (1 + self.beta) * pre * rec / (self.beta * pre + rec)
        return adaptive_fm

    def cal_pr(self, pred: np.ndarray, gt: np.ndarray) -> tuple:
        """
        Calculate the corresponding precision and recall when the threshold changes from 0 to 255.
        These precisions and recalls can be used to obtain the mean F-measure, maximum F-measure,
        precision-recall curve and F-measure-threshold curve.
        For convenience, ``changeable_fms`` is provided here, which can be used directly to obtain
        the mean F-measure, maximum F-measure and F-measure-threshold curve.
        :return: precisions, recalls, changeable_fms
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_hist, _ = np.histogram(pred[gt], bins=bins)
        bg_hist, _ = np.histogram(pred[~gt], bins=bins)

        fg_w_thrs = np.cumsum(np.flip(fg_hist), axis=0)
        bg_w_thrs = np.cumsum(np.flip(bg_hist), axis=0)

        TPs = fg_w_thrs
        Ps = fg_w_thrs + bg_w_thrs

        Ps[Ps == 0] = 1
        T = max(np.count_nonzero(gt), 1)

        precisions = TPs / Ps
        recalls = TPs / T

        numerator = (1 + self.beta) * precisions * recalls
        denominator = np.where(numerator == 0, 1, self.beta * precisions + recalls)
        changeable_fms = numerator / denominator
        return precisions, recalls, changeable_fms

    def get_results(self) -> dict:
        """
        Return the results about F-measure.
        :return: dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))
        """
        adaptive_fm = np.mean(np.array(self.adaptive_fms, _TYPE))
        changeable_fm = np.mean(np.array(self.changeable_fms, dtype=_TYPE), axis=0)
        precision = np.mean(np.array(self.precisions, dtype=_TYPE), axis=0)  # N, 256
        recall = np.mean(np.array(self.recalls, dtype=_TYPE), axis=0)  # N, 256
        return dict(fm=dict(adp=adaptive_fm, curve=changeable_fm), pr=dict(p=precision, r=recall))


class MAE_SOD(object):
    def __init__(self):
        """
        MAE(mean absolute error) for SOD.
        ::
            @inproceedings{MAE,
                title={Saliency filters: Contrast based filtering for salient region detection},
                author={Perazzi, Federico and Kr{\"a}henb{\"u}hl, Philipp and Pritch, Yael and Hornung, Alexander},
                booktitle=CVPR,
                pages={733--740},
                year={2012}
            }
        """
        self.maes = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred, gt)

        mae = self.cal_mae(pred, gt)
        # mae = np.sum(cv2.absdiff(gt.astype(float), pred.astype(float))) / (pred.shape[1] * pred.shape[0])
        self.maes.append(mae)

    def cal_mae(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the mean absolute error.
        :return: mae
        """
        mae = np.mean(np.abs(pred - gt))
        return mae

    def get_results(self) -> dict:
        """
        Return the results about MAE.
        :return: dict(mae=mae)
        """
        mae = np.mean(np.array(self.maes, _TYPE))
        return dict(mae=mae)


class Smeasure(object):
    def __init__(self, alpha: float = 0.5):
        """
        S-measure(Structure-measure) of SOD.
        ::
            @inproceedings{Smeasure,
                title={Structure-measure: A new way to eval foreground maps},
                author={Fan, Deng-Ping and Cheng, Ming-Ming and Liu, Yun and Li, Tao and Borji, Ali},
                booktitle=ICCV,
                pages={4548--4557},
                year={2017}
            }
        :param alpha: the weight for balancing the object score and the region score
        """
        self.sms = []
        self.alpha = alpha

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        sm = self.cal_sm(pred, gt)
        self.sms.append(sm)

    def cal_sm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the S-measure.
        :return: s-measure
        """
        y = np.mean(gt)
        if y == 0:
            sm = 1 - np.mean(pred)
        elif y == 1:
            sm = np.mean(pred)
        else:
            sm = self.alpha * self.object(pred, gt) + (1 - self.alpha) * self.region(pred, gt)
            sm = max(0, sm)
        return sm

    def object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the object score.
        """
        fg = pred * gt
        bg = (1 - pred) * (1 - gt)
        u = np.mean(gt)
        object_score = u * self.s_object(fg, gt) + (1 - u) * self.s_object(bg, 1 - gt)
        return object_score

    def s_object(self, pred: np.ndarray, gt: np.ndarray) -> float:
        x = np.mean(pred[gt == 1])
        sigma_x = np.std(pred[gt == 1], ddof=1)
        score = 2 * x / (np.power(x, 2) + 1 + sigma_x + _EPS)
        return score

    def region(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the region score.
        """
        x, y = self.centroid(gt)
        part_info = self.divide_with_xy(pred, gt, x, y)
        w1, w2, w3, w4 = part_info["weight"]
        # assert np.isclose(w1 + w2 + w3 + w4, 1), (w1 + w2 + w3 + w4, pred.mean(), gt.mean())

        pred1, pred2, pred3, pred4 = part_info["pred"]
        gt1, gt2, gt3, gt4 = part_info["gt"]
        score1 = self.ssim(pred1, gt1)
        score2 = self.ssim(pred2, gt2)
        score3 = self.ssim(pred3, gt3)
        score4 = self.ssim(pred4, gt4)

        return w1 * score1 + w2 * score2 + w3 * score3 + w4 * score4

    def centroid(self, matrix: np.ndarray) -> tuple:
        """
        To ensure consistency with the matlab code, one is added to the centroid coordinate,
        so there is no need to use the redundant addition operation when dividing the region later,
        because the sequence generated by ``1:X`` in matlab will contain ``X``.
        :param matrix: a bool data array
        :return: the centroid coordinate
        """
        h, w = matrix.shape
        area_object = np.count_nonzero(matrix)
        if area_object == 0:
            x = np.round(w / 2)
            y = np.round(h / 2)
        else:
            # More details can be found at: https://www.yuque.com/lart/blog/gpbigm
            y, x = np.argwhere(matrix).mean(axis=0).round()
        return int(x) + 1, int(y) + 1

    def divide_with_xy(self, pred: np.ndarray, gt: np.ndarray, x: int, y: int) -> dict:
        """
        Use (x,y) to divide the ``pred`` and the ``gt`` into four submatrices, respectively.
        """
        h, w = gt.shape
        area = h * w

        gt_LT = gt[0:y, 0:x]
        gt_RT = gt[0:y, x:w]
        gt_LB = gt[y:h, 0:x]
        gt_RB = gt[y:h, x:w]

        pred_LT = pred[0:y, 0:x]
        pred_RT = pred[0:y, x:w]
        pred_LB = pred[y:h, 0:x]
        pred_RB = pred[y:h, x:w]

        w1 = x * y / area
        w2 = y * (w - x) / area
        w3 = (h - y) * x / area
        w4 = 1 - w1 - w2 - w3

        return dict(
            gt=(gt_LT, gt_RT, gt_LB, gt_RB),
            pred=(pred_LT, pred_RT, pred_LB, pred_RB),
            weight=(w1, w2, w3, w4),
        )

    def ssim(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the ssim score.
        """
        h, w = pred.shape
        N = h * w

        x = np.mean(pred)
        y = np.mean(gt)

        sigma_x = np.sum((pred - x) ** 2) / (N - 1)
        sigma_y = np.sum((gt - y) ** 2) / (N - 1)
        sigma_xy = np.sum((pred - x) * (gt - y)) / (N - 1)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x + sigma_y)

        if alpha != 0:
            score = alpha / (beta + _EPS)
        elif alpha == 0 and beta == 0:
            score = 1
        else:
            score = 0
        return score

    def get_results(self) -> dict:
        """
        Return the results about S-measure.
        :return: dict(sm=sm)
        """
        sm = np.mean(np.array(self.sms, dtype=_TYPE))
        return dict(sm=sm)


class Emeasure(object):
    def __init__(self):
        """
        E-measure(Enhanced-alignment Measure) for SOD.
        More details about the implementation can be found in https://www.yuque.com/lart/blog/lwgt38
        ::
            @inproceedings{Emeasure,
                title="Enhanced-alignment Measure for Binary Foreground Map Evaluation",
                author="Deng-Ping {Fan} and Cheng {Gong} and Yang {Cao} and Bo {Ren} and Ming-Ming {Cheng} and Ali {Borji}",
                booktitle=IJCAI,
                pages="698--704",
                year={2018}
            }
        """
        self.adaptive_ems = []
        self.changeable_ems = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        self.gt_fg_numel = np.count_nonzero(gt)
        self.gt_size = gt.shape[0] * gt.shape[1]

        changeable_ems = self.cal_changeable_em(pred, gt)
        self.changeable_ems.append(changeable_ems)
        adaptive_em = self.cal_adaptive_em(pred, gt)
        self.adaptive_ems.append(adaptive_em)

    def cal_adaptive_em(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the adaptive E-measure.
        :return: adaptive_em
        """
        adaptive_threshold = _get_adaptive_threshold(pred, max_value=1)
        adaptive_em = self.cal_em_with_threshold(pred, gt, threshold=adaptive_threshold)
        return adaptive_em

    def cal_changeable_em(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the changeable E-measure, which can be used to obtain the mean E-measure,
        the maximum E-measure and the E-measure-threshold curve.
        :return: changeable_ems
        """
        changeable_ems = self.cal_em_with_cumsumhistogram(pred, gt)
        return changeable_ems

    def cal_em_with_threshold(self, pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
        """
        Calculate the E-measure corresponding to the specific threshold.
        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        binarized_pred = pred >= threshold
        fg_fg_numel = np.count_nonzero(binarized_pred & gt)
        fg_bg_numel = np.count_nonzero(binarized_pred & ~gt)

        fg___numel = fg_fg_numel + fg_bg_numel
        bg___numel = self.gt_size - fg___numel

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel
        else:
            parts_numel, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel,
                fg_bg_numel=fg_bg_numel,
                pred_fg_numel=fg___numel,
                pred_bg_numel=bg___numel,
            )

            results_parts = []
            for i, (part_numel, combination) in enumerate(zip(parts_numel, combinations)):
                align_matrix_value = (
                    2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts.append(enhanced_matrix_value * part_numel)
            enhanced_matrix_sum = sum(results_parts)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def cal_em_with_cumsumhistogram(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Calculate the E-measure corresponding to the threshold that varies from 0 to 255..
        Variable naming rules within the function:
        ``[pred attribute(foreground fg, background bg)]_[gt attribute(foreground fg, background bg)]_[meaning]``
        If only ``pred`` or ``gt`` is considered, another corresponding attribute location is replaced with '``_``'.
        """
        pred = (pred * 255).astype(np.uint8)
        bins = np.linspace(0, 256, 257)
        fg_fg_hist, _ = np.histogram(pred[gt], bins=bins)
        fg_bg_hist, _ = np.histogram(pred[~gt], bins=bins)
        fg_fg_numel_w_thrs = np.cumsum(np.flip(fg_fg_hist), axis=0)
        fg_bg_numel_w_thrs = np.cumsum(np.flip(fg_bg_hist), axis=0)

        fg___numel_w_thrs = fg_fg_numel_w_thrs + fg_bg_numel_w_thrs
        bg___numel_w_thrs = self.gt_size - fg___numel_w_thrs

        if self.gt_fg_numel == 0:
            enhanced_matrix_sum = bg___numel_w_thrs
        elif self.gt_fg_numel == self.gt_size:
            enhanced_matrix_sum = fg___numel_w_thrs
        else:
            parts_numel_w_thrs, combinations = self.generate_parts_numel_combinations(
                fg_fg_numel=fg_fg_numel_w_thrs,
                fg_bg_numel=fg_bg_numel_w_thrs,
                pred_fg_numel=fg___numel_w_thrs,
                pred_bg_numel=bg___numel_w_thrs,
            )

            results_parts = np.empty(shape=(4, 256), dtype=np.float64)
            for i, (part_numel, combination) in enumerate(zip(parts_numel_w_thrs, combinations)):
                align_matrix_value = (
                    2 * (combination[0] * combination[1]) / (combination[0] ** 2 + combination[1] ** 2 + _EPS)
                )
                enhanced_matrix_value = (align_matrix_value + 1) ** 2 / 4
                results_parts[i] = enhanced_matrix_value * part_numel
            enhanced_matrix_sum = results_parts.sum(axis=0)

        em = enhanced_matrix_sum / (self.gt_size - 1 + _EPS)
        return em

    def generate_parts_numel_combinations(self, fg_fg_numel, fg_bg_numel, pred_fg_numel, pred_bg_numel):
        bg_fg_numel = self.gt_fg_numel - fg_fg_numel
        bg_bg_numel = pred_bg_numel - bg_fg_numel

        parts_numel = [fg_fg_numel, fg_bg_numel, bg_fg_numel, bg_bg_numel]

        mean_pred_value = pred_fg_numel / self.gt_size
        mean_gt_value = self.gt_fg_numel / self.gt_size

        demeaned_pred_fg_value = 1 - mean_pred_value
        demeaned_pred_bg_value = 0 - mean_pred_value
        demeaned_gt_fg_value = 1 - mean_gt_value
        demeaned_gt_bg_value = 0 - mean_gt_value

        combinations = [
            (demeaned_pred_fg_value, demeaned_gt_fg_value),
            (demeaned_pred_fg_value, demeaned_gt_bg_value),
            (demeaned_pred_bg_value, demeaned_gt_fg_value),
            (demeaned_pred_bg_value, demeaned_gt_bg_value),
        ]
        return parts_numel, combinations

    def get_results(self) -> dict:
        """
        Return the results about E-measure.
        :return: dict(em=dict(adp=adaptive_em, curve=changeable_em))
        """
        adaptive_em = np.mean(np.array(self.adaptive_ems, dtype=_TYPE))
        changeable_em = np.mean(np.array(self.changeable_ems, dtype=_TYPE), axis=0)
        return dict(em=dict(adp=adaptive_em, curve=changeable_em))


class WeightedFmeasure(object):
    def __init__(self, beta: float = 0.3):
        """
        Weighted F-measure for SOD.
        ::
            @inproceedings{wFmeasure,
                title={How to eval foreground maps?},
                author={Margolin, Ran and Zelnik-Manor, Lihi and Tal, Ayellet},
                booktitle=CVPR,
                pages={248--255},
                year={2014}
            }
        :param beta: the weight of the precision
        """
        self.beta = beta
        self.weighted_fms = []

    def step(self, pred: np.ndarray, gt: np.ndarray):
        pred, gt = _prepare_data(pred=pred, gt=gt)

        if np.all(~gt):
            wfm = 0
        else:
            wfm = self.cal_wfm(pred, gt)
        self.weighted_fms.append(wfm)

    def cal_wfm(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """
        Calculate the weighted F-measure.
        """
        # [Dst,IDXT] = bwdist(dGT);
        Dst, Idxt = bwdist(gt == 0, return_indices=True)

        # %Pixel dependency
        # E = abs(FG-dGT);
        E = np.abs(pred - gt)
        # Et = E;
        # Et(~GT)=Et(IDXT(~GT)); %To deal correctly with the edges of the foreground region
        Et = np.copy(E)
        Et[gt == 0] = Et[Idxt[0][gt == 0], Idxt[1][gt == 0]]

        # K = fspecial('gaussian',7,5);
        # EA = imfilter(Et,K);
        K = self.matlab_style_gauss2D((7, 7), sigma=5)
        EA = convolve(Et, weights=K, mode="constant", cval=0)
        # MIN_E_EA = E;
        # MIN_E_EA(GT & EA<E) = EA(GT & EA<E);
        MIN_E_EA = np.where(gt & (EA < E), EA, E)

        # %Pixel importance
        # B = ones(size(GT));
        # B(~GT) = 2-1*exp(log(1-0.5)/5.*Dst(~GT));
        # Ew = MIN_E_EA.*B;
        B = np.where(gt == 0, 2 - np.exp(np.log(0.5) / 5 * Dst), np.ones_like(gt))
        Ew = MIN_E_EA * B

        # TPw = sum(dGT(:)) - sum(sum(Ew(GT)));
        # FPw = sum(sum(Ew(~GT)));
        TPw = np.sum(gt) - np.sum(Ew[gt == 1])
        FPw = np.sum(Ew[gt == 0])

        # R = 1- mean2(Ew(GT)); %Weighed Recall
        # P = TPw./(eps+TPw+FPw); %Weighted Precision
        # 注意这里使用mask索引矩阵的时候不可使用Ew[gt]，这实际上仅在索引Ew的0维度
        R = 1 - np.mean(Ew[gt == 1])
        P = TPw / (TPw + FPw + _EPS)

        # % Q = (1+Beta^2)*(R*P)./(eps+R+(Beta.*P));
        Q = (1 + self.beta) * R * P / (R + self.beta * P + _EPS)

        return Q

    def matlab_style_gauss2D(self, shape: tuple = (7, 7), sigma: int = 5) -> np.ndarray:
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1) / 2 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def get_results(self) -> dict:
        """
        Return the results about weighted F-measure.
        :return: dict(wfm=weighted_fm)
        """
        weighted_fm = np.mean(np.array(self.weighted_fms, dtype=_TYPE))
        return dict(wfm=weighted_fm)


class Multiclass_IoU(torchmetrics.Metric):
    """
    Compute the IoU for multi-class semantic segmentation based on https://github.com/xieenze/Trans2Seg/blob/master/segmentron/utils/score.py.
    The direct use of torchmetrics for large dataset will lead to issues such as high CPU usage or insufficient memory.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.add_state("total_inter", default=torch.zeros(num_classes), dist_reduce_fx=None)
        self.add_state("total_union", default=torch.zeros(num_classes), dist_reduce_fx=None)
        self.num_classes = num_classes

    def update(self, logits, labels):
        inter, union = self.batch_intersection_union(logits, labels)
        self.total_inter += inter
        self.total_union += union

    def compute(self):
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        return torch.tensor(IoU.mean().item())

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = self.num_classes
        nbins = self.num_classes
        predict = torch.argmax(output, 1) + 1
        target = target.float() + 1

        predict = predict.float() * (target > 0).float()
        intersection = predict * (predict == target).float()
        # areas of intersection and union
        area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        return area_inter.float(), area_union.float()


class Binary_IoU(torchmetrics.Metric):
    """
    Compute the IoU for binary semantic segmentation. The direct use of torchmetrics to calculate IoU for multiple samples does not yield accurate results.
    So we iteratively calculate metric values and then take the average.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("logits", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        logits = torch.cat(self.logits).cpu()
        labels = torch.cat(self.labels).cpu()

        res_list = []
        metric = torchmetrics.JaccardIndex(task="binary")
        for logit, label in zip(logits, labels):
            res_list.append(metric(logit, label))
        return torch.mean(torch.tensor(res_list))


class Balanced_Error_Rate(torchmetrics.Metric):
    """
    Compute the balanced error rate.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.add_state("logits", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        logits = torch.cat(self.logits).cpu()
        labels = torch.cat(self.labels).cpu()

        labels = (labels * 255) > 125
        logits = (logits * 255) > 125
        ber = 1 - torchmetrics.Accuracy(
            task="multiclass", num_classes=2, average="macro", multidim_average="samplewise"
        )(logits, labels)
        return torch.mean(ber)


class COD(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("logits", default=[], dist_reduce_fx=None)
        self.add_state("labels", default=[], dist_reduce_fx=None)

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        pass


class SM(COD):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_SM = Smeasure()

        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255
            metric_SM.step(pred=pred, gt=true)

        return torch.tensor(metric_SM.get_results()["sm"])


class FM(COD):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_WFM = WeightedFmeasure()
        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_WFM.step(pred=pred, gt=true)

        return torch.tensor(metric_WFM.get_results()["wfm"])


class EM(COD):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_EM = Emeasure()

        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_EM.step(pred=pred, gt=true)

        return torch.tensor(metric_EM.get_results()["em"]["curve"].mean())


class MAE(COD):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_MAE = MAE_SOD()
        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_MAE.step(pred=pred, gt=true)

        return torch.tensor(metric_MAE.get_results()["mae"])


COD_METRICS_NAMES = {"sm": SM(), "fm": FM(), "em": EM(), "mae": MAE()}


# TODO: Modify multi-gpu evaluation error. Maybe there will be a more elegant way.
class Multiclass_IoU_Pred:
    """
    Compute the IoU for multi-class semantic segmentation based on https://github.com/xieenze/Trans2Seg/blob/master/segmentron/utils/score.py.
    The direct use of torchmetrics for large dataset will lead to issues such as high CPU usage or insufficient memory.
    """

    def __init__(self, num_classes):
        super().__init__()
        self.total_inter = torch.zeros(num_classes)
        self.total_union = torch.zeros(num_classes)
        self.num_classes = num_classes

    def update(self, logits, labels):
        inter, union = self.batch_intersection_union(logits, labels)
        self.total_inter += inter
        self.total_union += union

    def compute(self):
        IoU = 1.0 * self.total_inter / (2.220446049250313e-16 + self.total_union)
        return torch.tensor(IoU.mean().item())

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = self.num_classes
        nbins = self.num_classes
        predict = torch.argmax(output, 1) + 1
        target = target.float() + 1

        predict = predict.float() * (target > 0).float()
        intersection = predict * (predict == target).float()
        # areas of intersection and union
        area_inter = torch.histc(intersection, bins=nbins, min=mini, max=maxi)
        area_pred = torch.histc(predict, bins=nbins, min=mini, max=maxi)
        area_lab = torch.histc(target, bins=nbins, min=mini, max=maxi)
        area_union = area_pred + area_lab - area_inter
        assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
        return area_inter.float(), area_union.float()


class Binary_IoU_Pred:
    """
    Compute the IoU for binary semantic segmentation. The direct use of torchmetrics to calculate IoU for multiple samples does not yield accurate results.
    So we iteratively calculate metric values and then take the average.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.logits = []
        self.labels = []

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        logits = torch.cat(self.logits).cpu()
        labels = torch.cat(self.labels).cpu()

        res_list = []
        metric = torchmetrics.JaccardIndex(task="binary")
        for logit, label in zip(logits, labels):
            res_list.append(metric(logit, label))
        return torch.mean(torch.tensor(res_list))


class Balanced_Error_Rate_Pred:
    """
    Compute the balanced error rate.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.logits = []
        self.labels = []

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        logits = torch.cat(self.logits).cpu()
        labels = torch.cat(self.labels).cpu()

        labels = (labels * 255) > 125
        logits = (logits * 255) > 125
        ber = 1 - torchmetrics.Accuracy(
            task="multiclass", num_classes=2, average="macro", multidim_average="samplewise"
        )(logits, labels)
        return torch.mean(ber)


class COD_Pred:
    def __init__(self):
        super().__init__()
        self.logits = []
        self.labels = []

    def update(self, logits, labels):
        self.logits.append(logits)
        self.labels.append(labels)

    def compute(self):
        pass


class SM_Pred(COD_Pred):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_SM = Smeasure()

        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255
            metric_SM.step(pred=pred, gt=true)

        return torch.tensor(metric_SM.get_results()["sm"])


class FM_Pred(COD_Pred):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_WFM = WeightedFmeasure()
        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_WFM.step(pred=pred, gt=true)

        return torch.tensor(metric_WFM.get_results()["wfm"])


class EM_Pred(COD_Pred):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_EM = Emeasure()

        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_EM.step(pred=pred, gt=true)

        return torch.tensor(metric_EM.get_results()["em"]["curve"].mean())


class MAE_Pred(COD_Pred):
    def compute(self):
        logits = torch.cat(self.logits)
        labels = torch.cat(self.labels)
        assert logits.shape == labels.shape
        batchsize = labels.shape[0]

        metric_MAE = MAE_SOD()
        for i in range(batchsize):
            true, pred = labels[i, 0].cpu().data.numpy() * 255, logits[i, 0].cpu().data.numpy() * 255

            metric_MAE.step(pred=pred, gt=true)

        return torch.tensor(metric_MAE.get_results()["mae"])


COD_METRICS_NAMES_Pred = {"sm": SM_Pred(), "fm": FM_Pred(), "em": EM_Pred(), "mae": MAE_Pred()}
