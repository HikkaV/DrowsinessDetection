import keras

from train import Train

train = Train()
train.run_minimize()

# train.predict_normal('models/model819~resnet~loss0.0715680956130936.h5','/home/hikkav/hack/data/landmarks/test_images/0000.jpg',
#                   'scaler.pickle', False, 'pred_epoch1_resnet.png')
# model = keras.models.load_model('models/model819~resnet~loss0.0715680956130936.h5')
# train.make_submission_normal(model=model)