from keras import Input, Model
from keras.applications import MobileNet
from keras.layers import SeparableConv2D, BatchNormalization, Dense, Dropout, GlobalAveragePooling2D, Activation, \
    Flatten
from keras.optimizers import Adam
from keras.regularizers import l2
from nets_custom import net_factory

from utils.helpers import l1_loss

class Mobnet_Custom(net_factory.NetworkFactory):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes = kwargs['classes']
        self.w = kwargs['weights']
        self.dropout_global = kwargs['dropout_global']
        self.dropout = kwargs['dropout']
        self.eta = kwargs['eta']
        self.dim = kwargs['dim']
        self.regulizer = kwargs['reg']


    def create_model(self):
        inputs = Input(shape=self.dim, name='input')
        model_mobilenet = MobileNet(input_shape=self.dim, alpha=1, depth_multiplier=1, dropout=self.dropout_global,
                                    include_top=False, weights=self.w, input_tensor=None)

        x = model_mobilenet(inputs)
        x = SeparableConv2D(filters=128, kernel_size=(7, 7), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dense(1024, activation='relu', kernel_regularizer=l2(self.regulizer))(x)
        x = Dropout(self.dropout)(x)
        z = Dense(self.classes, activation='tanh')(x)
        model = Model(inputs=inputs, outputs=z)

        adam = Adam(lr=self.eta)
        model.compile(optimizer=adam, loss=l1_loss, metrics=['mse', 'mae'] )

        print(model.summary())

        return model
