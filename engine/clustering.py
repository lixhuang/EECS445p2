"""
EECS 445 - Introduction to Machine Learning
Winter 2017 - Project 2
Clustering - Skeleton
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import model.clustering_classes as ccs
from utils import clustering_utils
#import model.clustering_classes as ccs
from data_scripts.fer2013 import FER2013


def build_face_image_points(X, y):
    """
    Input:
        X : (n,d) feature matrix, in which each row represents an image
        y: (n,1) array, vector, containing labels corresponding to X
    Returns:
        List of Points
    """
    (n, d) = X.shape
    images = {}
    points = []
    for i in range(0, n):
        if y[i] not in images.keys():
            images[y[i]] = []
        images[y[i]].append(X[i, :])
    for face in images.keys():
        count = 0
        for im in images[face]:
            points.append(ccs.Point(str(face) + '_' + str(count), face, im))
            count = count + 1

    return points


def random_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO
    k_list = [];
    d = len(points);
    rd = np.random.choice(d, k, False);
    for i in range(k):
        k_list += [points[rd[i]]];
    return np.array(k_list);


def calculate_min_dist(pt, chosen_points):
    min_dist = pt.distance(chosen_points[0])
    min_j = 0
    
    for j in range(len(chosen_points)):
      if pt.distance(chosen_points[j]) < min_dist:
        min_dist = pt.distance(chosen_points[j])
        min_j = j
    return min_dist, min_j

def k_means_pp_init(points, k):
    """
    Input:
        points: a list of point objects
        k: Number of initial centroids/medoids
    Returns:
        List of k unique points randomly selected from points
    """
    # TODO
    '''
    chosen_points = []
    chosen_points += [np.random.choice(points)]
    
    for i in range(k - 1):
      dist2 = []
      for j in points:
        min_dist, rubbish = calculate_min_dist(j, chosen_points)
        dist2 += [min_dist ** 2]
      dist2 = dist2 / sum(dist2)
      chosen_points += [np.random.choice(points, None, False, dist2)]
      
    print(len(chosen_points))
    return chosen_points
    '''
    
    d = len(points);
    rd = np.random.randint(d);    

    init_set = [];
    init_pt = points[rd];
    init_set += [init_pt];

    for i in range(k-1):
        dis_vec = [];
        for i in range(d):
            dis_vec += [points[i].distance(init_pt)];
        dis_vec = np.array(dis_vec);
        d_set = np.divide(dis_vec,np.sum(dis_vec));
        #print(k);
        #print(d_set);
        rd = np.random.choice(d, 1, False, p=d_set);
        init_pt = points[rd[0]];
        init_set += [init_pt];
    return np.array(init_set);
    

def build_cluster(center_list, points):
    k = len(center_list);
    #print("bilud: ", k);
    cluster_set = [];
    #print(center_list[0]);
    #print(center_list[1]);
    for i in range(k):
        #temp_set = ccs.Cluster([]);
        cluster_set += [[]];

    for i in range(len(points)):
        dist = float("inf");
        c_id = 0;
        for j in range(k):
            '''
            if(points[i].distance(center_list[j]) == 0):
                print("coorepond ", j)
            '''
            if(points[i].distance(center_list[j]) < dist):
                '''
                if(points[i].distance(center_list[j]) == 0):
                    print("j ", j);
                    print(i)
                    print("dist", dist);
                '''
                dist = points[i].distance(center_list[j]);
                c_id = j;
        cluster_set[c_id] += [points[i]];

    res = ccs.ClusterSet();
    for i in range(k):
        #print(len(cluster_set[i]));
        res.add(ccs.Cluster(np.array(cluster_set[i])));
    return res;

def k_means(points, k, init='random'):
    """
    Input:
        points: a list of Point objects
        k: the number of clusters we want to end up with
        init: The method of initialization, takes two valus 'cheat'
              and 'random'. If init='cheat', then use cheat_init to get
              initial clusters. If init='random', then use random_init
              to initialize clusters. Default value 'random'.

    Clusters points into k clusters using k_means clustering.

    Returns:
        Instance of ClusterSet corresponding to k clusters
    """
    # TODO
    print("K: ", k, "running")
    if(init == 'random'):
        init_set = random_init(points, k);
    else:
        init_set = k_means_pp_init(points, k);

    cluster_set = build_cluster(init_set, points);
    prev_clust = cluster_set;
    #print(len(cluster_set.members[0].points));
    #print(len(cluster_set.members[1].points));
    while 1:
        print("onece");
        center_set = cluster_set.get_centroids();
        cluster_set = build_cluster(center_set, points);
        if(prev_clust.equivalent(cluster_set)):
            break;
        prev_clust = cluster_set;

    return cluster_set;





def plot_performance(k_means_Scores, kpp_Scores, k_vals):
    """
    Input:
        KMeans_Scores: A list of len(k_vals) average purity scores from running the
                       KMeans algorithm with Random Init
        KPP_Scores: A list of len(k_vals) average purity scores from running the
                    KMeans algorithm with KMeans++ Init
        K_Vals: A list of integer k values used to calculate the above scores

    Uses matplotlib to generate a graph of performance vs. k
    """
    # TODO
    km, = plt.plot(k_vals, k_means_Scores, marker = '.', linestyle = '--');
    kpp, = plt.plot(k_vals, kpp_Scores, marker = '.', linestyle = '--');
    plt.legend([km, kpp], ['KMeans', 'KMeans++']);
    plt.xlabel('Number of clusters, k');
    plt.ylabel('Purity');
    plt.axis([1, 10, 0.16, 0.27]);
    plt.show();



def main():


    print('testing cluster');
    train_images, train_labels = clustering_utils.get_data();
    test_images, test_labels = clustering_utils.get_testdata();
    '''
    data = FER2013();

    train_images, train_labels = data.preprocessed_data('train',
                                                        one_hot=1,
                                                        balance_classes=1);
    test_images, test_labels = data.preprocessed_data('test',
                                                      one_hot=1,
                                                      balance_classes=1);
    '''
    points = build_face_image_points(train_images, train_labels);
    points_test = build_face_image_points(test_images, test_labels);
    pred_lb = [];
    cluster = k_means(points, 7, "cheat");
    label_list = [];
    for i in range(7):
        bucket = np.array([0,0,0,0,0,0,0]);
        temp_c = cluster.members[i];
        for j in range(len(temp_c.points)):
            bucket[temp_c.points[j].label] += 1;
        label_list += [bucket.argmax];
    center_list = cluster.get_centroids();

    for i in range(len(points_test)):
        dist = float("inf");
        c_id = 0; 
        for j in range(len(center_list)):
            if(points_test[i].distance(center_list[j]) < dist):
                dist = points[i].distance(center_list[j]);
                c_id = j;
        pred_lb += [label_list[j]];

    c = 0;
    for i in range(len(points_test)):
        if(points_test[i].label == pred_lb[i]):
            c += 1;
    print(c);

    return;



    X, y = clustering_utils.get_data()
    points = build_face_image_points(X, y)
    #print("data_read");
    score_p = [];
    for i in range(10):
        temp_c = k_means(points, i+1, "cheat");
        score_p += [temp_c.get_score()];
    score_r = [];
    for i in range(10):
        print(i);
        temp_c = k_means(points, i+1);
        score_r += [temp_c.get_score()];
    
    plot_performance(score_r, score_p, range(1,11));





if __name__ == '__main__':
    main()
