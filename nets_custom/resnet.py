from keras import Input, Model
from keras.applications import ResNet50
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.regularizers import l2

from nets_custom import net_factory
from utils.helpers import l1_loss

class ResNet_Custom(net_factory.NetworkFactory):

    def __init__(self, **kwargs):
        super().__init__( **kwargs)
        self.classes = kwargs['classes']
        self.w = kwargs['weights']
        self.dropout = kwargs['dropout']
        self.eta = kwargs['eta']
        self.dim = kwargs['dim']
        self.regulizer =kwargs['reg']



    def create_model(self):
        inputs = Input(shape=self.dim, name='input')
        resnet = ResNet50(include_top=False, weights=self.w, input_tensor=None,
                          input_shape=self.dim)
        x = resnet(inputs)
        x = GlobalAveragePooling2D()(x)

        x = Dense(256, activation='relu', kernel_regularizer=l2(self.regulizer))(x)
        x = Dropout(self.dropout)(x)
        z = Dense(self.classes, activation='tanh')(x)

        model = Model(inputs=inputs, outputs=z)

        adam = Adam(lr=self.eta)
        model.compile(optimizer=adam, loss=l1_loss, metrics=['mse','mae'])

        print(model.summary())

        return model
