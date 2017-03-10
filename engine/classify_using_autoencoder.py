"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Use autoencoder representation as feature vector for image classifier
Usage: Run the command `python -m engine.classify_using_autoencoder`
       to view classification score
"""

import tensorflow as tf
from model.build_autoencoder import autoencoder
from utils.config import get, is_file_prefix
from data_scripts.fer2013_dataset import read_data_sets

from sklearn.linear_model import LogisticRegression
import engine.clustering

if __name__ == '__main__':
    '''
    assert is_file_prefix(
        'TRAIN.AUTOENCODER.CHECKPOINT'), 'training checkpoint not found!'
    print('building model...')
    sess = tf.InteractiveSession()  # start talking to tensorflow backend
    original_image, compressed, reconstruction = autoencoder()  # fetch model layers
    sess.run(tf.global_variables_initializer())  # initialize some globals
    saver = tf.train.Saver()  # prepare to restore model
    saver.restore(sess, get('TRAIN.AUTOENCODER.CHECKPOINT'))
    print('Yay! I restored weights from a saved model!')

    print('loading data...')
    faces = read_data_sets(one_hot=False)

    print('training classifier...')
    compressed_trainset = compressed.eval(
        feed_dict={original_image: faces.train.images})
    clf = LogisticRegression()  # TODO (optional): adjust hyperparameters
    clf.fit(compressed_trainset, faces.train.labels)

    print('testing classfier...')
    compressed_testset = compressed.eval(
        feed_dict={original_image: faces.test.images})
    accuracy = clf.score(compressed_testset, faces.test.labels)
    print('Autoencoder-based classifier achieves accuracy \n%f' % accuracy)

    '''
    print('testing cluster');
    data = FER2013();

    train_images, train_labels = data.preprocessed_data('train',
                                                        one_hot=one_hot,
                                                        balance_classes=balance_classes);
    val_images, val_labels = data.preprocessed_data('val',
                                                    one_hot=one_hot,
                                                    balance_classes=balance_classes);
    test_images, test_labels = data.preprocessed_data('test',
                                                      one_hot=one_hot,
                                                      balance_classes=balance_classes);
    points = build_face_image_points(training_images, y);
    points_test = build_face_image_points(test_images, y);
    pred_lb = [];
    cluster = k_means(points, 7, "cheat");
    label_list = [];
    for i in range(7):
        bucket = np.array([0,0,0,0,0,0,0]);
        temp_c = cluster.members[i];
        for j in range(len(temp_c)):
            bucket = np.add(bucket, temp_c.points[j].label);
        label_list += [bucket.argmax];
    center_list = cluster.get_centroids();

    for i in range(len(points_test)):
        dist = float("inf");
        c_id = 0; 
        for j in range(len(temp_c)):
            if(points_test[i].distance(center_list[j]) < dist):
                dist = points[i].distance(center_list[j]);
                c_id = j;
        pred_lb += [label_list[j]];

    c = 0;
    for i in range(len(points_test)):
        if(points_test[i].label[pred_lb[i]]):
            c += 1;
    print(c);






















