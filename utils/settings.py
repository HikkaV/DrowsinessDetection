dim = (128, 128)
weights = 'imagenet'
weights_darknet = '/home/hikkav/PycharmProjects/makeProjectData/weights/yolov3-wider_16000.weights'
cfg_darnket = '/home/hikkav/PycharmProjects/makeProjectData/cfg/yolov3-face.cfg'
train_batch = 64
valid_batch = 16
path_to_valid = '/home/hikkav/hack/data/landmarks/test.csv'
path_to_train = '/home/hikkav/DrowsinessDetection/data_analysis/new_data.csv'
random_state = 19
abs_path = 'new_data/'
path_to_hist = 'history.csv'
classes = 34
ncalls = 10
res_net = True
space_params = [
    (1e-5, 1e-1),  # eta
    (1e-2, 0.9),  # dropout
    (1e-3, 0.9),  # dropout_global
    (0.001, 1),  # l2
    (60,80) #epochs
]
