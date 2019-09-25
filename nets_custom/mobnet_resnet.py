from keras import Input, Model
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
from nets_custom import net_factory


class mobnet_resnet(net_factory.NetworkFactory):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


