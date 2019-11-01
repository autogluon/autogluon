import os
import collections
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

__all__ = ['WireResNet', 'get_wire_resnet50', 'get_wire_resnet101']

class WireResNet(HybridBlock):
    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=nn.BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', **kwargs):
        super(WireResNet, self).__init__(**kwargs)
        self.inplanes = stem_width*2 if deep_stem else 64
        super(ResNetV1b, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width*2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False))
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width*2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma)
            if dilated:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                if avg_down:
                    if dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))

        layers = []
        if dilation in (1, 2):
            layers.append(block(planes, strides, dilation=1,
                                downsample=downsample, previous_dilation=dilation,
                                norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                last_gamma=last_gamma))
        elif dilation == 4:
            layers.append(block(planes, strides, dilation=2,
                                downsample=downsample, previous_dilation=dilation,
                                norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                last_gamma=last_gamma))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, dilation=dilation,
                                previous_dilation=dilation, norm_layer=norm_layer,
                                norm_kwargs=self.norm_kwargs, last_gamma=last_gamma))

        return Wire_Sequential(layers)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def get_wire_resnet50(**kwargs):
    from gluoncv.model_zoo.resnetv1b import BottleneckV1b
    model = WireResNet(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)
    return model

def get_wire_resnet101(**kwargs):
    from gluoncv.model_zoo.resnetv1b import BottleneckV1b
    model = WireResNet(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True, **kwargs)
    return model
