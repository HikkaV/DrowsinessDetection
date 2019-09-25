import keras
from train import Train

train = Train()
# train.run_minimize()
train.predict('models/model409~resnet~loss153.52494588077272.h5','/home/hikkav/hack/data/eye_tracking/test_images/0w4wco3.jpg',
                      'scaler.pickle', True, 'pred_epoch1_resnet.png', thr=True)
# model = keras.models.load_model('models/model819~resnet~loss0.0715680956130936.h5')
# train.make_submission_normal(model=model)