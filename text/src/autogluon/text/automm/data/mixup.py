from timm.data.mixup import Mixup

class MixupModule(Mixup):
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        super().__init__(mixup_alpha,cutmix_alpha,cutmix_minmax,prob,switch_prob,mode,correct_lam,label_smoothing,num_classes)

    def __call__(self, batch, image_key, label_key):
        assert len(batch[image_key]) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(batch[image_key])
        elif self.mode == 'pair':
            lam = self._mix_pair(batch[image_key])
        else:
            lam = self._mix_batch(batch[image_key])
        return batch, batch[label_key], batch[label_key].flip(0), lam
