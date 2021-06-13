import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import LinearSVC, SVC
from utils import load_image_gray
import cyvlfeat as vlfeat
import pickle
import math

def build_vocabulary(image_paths, vocab_size):
    """
      This function will sample SIFT descriptors from the training images,
      cluster them with kmeans, and then return the cluster centers.

      Useful functions:
      -   Use load_image_gray(path) to load grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
              -  frames is a N x 2 matrix of locations, which can be thrown away
              here
              -  descriptors is a N x 128 matrix of SIFT features
            Note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster to
            compute. Also, be sure not to use the default value of step size. It
            will be very slow and you'll see relatively little performance gain
            from extremely dense sampling. You are welcome to use your own SIFT
            feature code! It will probably be slower, though.
      -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
              http://www.vlfeat.org/matlab/vl_kmeans.html
                -  X is a N x d numpy array of sampled SIFT features, where N is
                   the number of features sampled. N should be pretty large!
                -  K is the number of clusters desired (vocab_size)
                   cluster_centers is a K x d matrix of cluster centers. This is
                   your vocabulary.

      Args:
      -   image_paths: list of image paths.
      -   vocab_size: size of vocabulary

      Returns:
      -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
          cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here. You can try taking 20 features
    # per image.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size,dim))
    total_SIFT_features = np.zeros((20*len(image_paths), dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    N = 20*len(image_paths)
    D = dim
    K = vocab_size
    #image_paths should be a list or something that has len
    #we can iterate through the image_paths one by one
    for i in range (len(image_paths)):
        
        #load images from image_paths
        sample = load_image_gray(image_paths[i])
        #sample descriptors with vl_dsift
        #to get 20 samples, we need a step size appropriate
        #print(sample)
        #sample is numpy ndarray
        #sample_x, sample_y = sample.shape
        #total_pixels = sample_x * sample_y
        step_size = int(sample.size/(20*128))
        #print("this is step_size")
        #print(step_size)
        #adjust parameters accordingly
        #def dsift(image, step=1, size=3, bounds=None, window_size=-1, norm=False,
                    #fast=False, float_descriptors=False, geometry=(4, 4, 8),
                      #verbose=False):
        frames, descriptors = vlfeat.sift.dsift(sample, step = step_size, window_size=dim,
                                                    norm=True, fast=True)
        #for robustness in case if step isze is too big
        if descriptors.shape[0] < 20:
            #pad with zeros by adding it to a zero array
            descriptors = np.zeros((20,dim)) + descriptors
        #or too small
        else:
            descriptors = descriptors[np.random.randint(0, high=descriptors.shape[0], size=20)]
        #pad descriptors if not more than 20 points
        #else get 20 random rows
        
        total_SIFT_features[i*20:(i+1)*20, :] = descriptors[:20,:dim]
    #we have gotten our samples, time to get cluster centers
    #perform kmeans clustering on total_SIFT_features
    #def kmeans(data, num_centers, distance="l2", initialization="RANDSEL",
           #algorithm="LLOYD", num_repetitions=1, num_trees=3,
           #max_num_comparisons=100, max_num_iterations=100,
           #min_energy_variation=0., verbose=False):
    vocab = vlfeat.kmeans.kmeans(total_SIFT_features, K)
    
    #return vocab
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab

def bags_of_sifts(image_paths, vocab_filename):
    """
      You will want to construct SIFT features here in the same way you
      did in build_vocabulary() (except for possibly changing the sampling
      rate) and then assign each local feature to its nearest cluster center
      and build a histogram indicating how many times each cluster was used.
      Don't forget to normalize the histogram, or else a larger image with more
      SIFT features will look very different from a smaller version of the same
      image.

      Useful functions:
      -   Use load_image(path) to load RGB images and load_image_gray(path) to load
              grayscale images
      -   frames, descriptors = vlfeat.sift.dsift(img)
              http://www.vlfeat.org/matlab/vl_dsift.html
            frames is a M x 2 matrix of locations, which can be thrown away here
            descriptors is a M x 128 matrix of SIFT features
              note: there are step, bin size, and smoothing parameters you can
              manipulate for dsift(). We recommend debugging with the 'fast'
              parameter. This approximate version of SIFT is about 20 times faster
              to compute. Also, be sure not to use the default value of step size.
              It will be very slow and you'll see relatively little performance
              gain from extremely dense sampling. You are welcome to use your own
              SIFT feature code! It will probably be slower, though.
      -   D = cdist(X, Y)
            computes the distance matrix D between all pairs of rows in X and Y.
              -  X is a N x d numpy array of d-dimensional features arranged along
              N rows
              -  Y is a M x d numpy array of d-dimensional features arranged along
              N rows
              -  D is a N x M numpy array where d(i, j) is the distance between row
              i of X and row j of Y

      Args:
      -   image_paths: paths to N images
      -   vocab_filename: Path to the precomputed vocabulary.
              This function assumes that vocab_filename exists and contains an
              vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
              or visual word. This ndarray is saved to disk rather than passed in
              as a parameter to avoid recomputing the vocabulary every run.

      Returns:
      -   image_feats: N x d matrix, where d is the dimensionality of the
              feature representation. In this case, d will equal the number of
              clusters or equivalently the number of entries in each image's
              histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    first = 1
    
    vocab_size, vocab_length = vocab.shape
    M = vocab_size
    d = vocab_length
    number_of_images = len(image_paths)
    N = number_of_images
    #lets call N(i) the number of descriptors on the image
    dim = 128
    #check d = dim
    if (d != dim):
        print("d is " + d + "and dim is " + dim)
    #we have loaded the vocab file name, now to match features of the images to the vocab
    for i in range (len(image_paths)):
        #load images from image_paths
        sample = load_image_gray(image_paths[i])
        step_size = 10
        #get descriptors
        frames, descriptors = vlfeat.sift.dsift(sample, step = step_size, window_size=dim,
                                                    norm=True, fast=True)
        #IOPub data rate exceeded.
        #The notebook server will temporarily stop sending output
        #to the client in order to avoid crashing it.
        #thus i might need to limit the number of descriptors of each to 2000
        if descriptors.shape[0] > 1000:
            descriptors = descriptors[np.random.randint(0, high=descriptors.shape[0], size=1000)]
        D = cdist(descriptors, vocab)
        #we got an N(i) x M array, to collapse into M array, add to closest cluster for each feature
        histogram = np.zeros((M))
        for ind1 in range (D.shape[0]):
            NN = np.amin(D[ind1])
            NN_ind = np.where(D[ind1] == NN)[0][0]
            histogram[NN_ind] = histogram[NN_ind] + 1
        #normalize
        histogram = histogram / np.linalg.norm(histogram)
        feats.append(histogram)
    #if feats is not recognized as ndarray
    #print("feats")
    #print(feats)
    feats = np.vstack(feats)
    #print("feats.shape")
    #print(feats.shape)
    
            
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats


def nearest_neighbor_classifier(train_image_feats, train_labels, test_image_feats,
    metric='euclidean'):
    """
      This function will predict the category for every test image by finding
      the training image with most similar features. Instead of 1 nearest
      neighbor, you can vote based on k nearest neighbors which will increase
      performance (although you need to pick a reasonable value for k).

      Useful functions:
      -   D = cdist(X, Y)
            computes the distance matrix D between all pairs of rows in X and Y.
              -  X is a N x d numpy array of d-dimensional features arranged along
              N rows
              -  Y is a M x d numpy array of d-dimensional features arranged along
              N rows
              -  D is a N x M numpy array where d(i, j) is the distance between row
              i of X and row j of Y

      Args:
      -   train_image_feats:  N x d numpy array, where d is the dimensionality of
              the feature representation
      -   train_labels: N element list, where each entry is a string indicating
              the ground truth category for each training image
      -   test_image_feats: M x d numpy array, where d is the dimensionality of the
              feature representation. You can assume N = M, unless you have changed
              the starter code
      -   metric: (optional) metric to be used for nearest neighbor.
              Can be used to select different distance functions. The default
              metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
              well for histograms

      Returns:
      -   test_labels: M element list, where each entry is a string indicating the
              predicted category for each testing image
    """
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    M = test_image_feats.shape[0]
    N = train_image_feats.shape[0]
    #K = 1
    #load train_image_feats
    #load train_labels
    #each train_labels should correspond to same number of train_image_feats
    #load test_image_feats
    #for each test_image_feats calculate distance to each train_image_feats
    D = cdist(test_image_feats,train_image_feats)
    #D is M x N
    #for i in range (K):
        #find nearest neighbour
    #get test_labels from nearest neighbours
    #for every row
    for ind1 in range (D.shape[0]):
        NN = np.amin(D[ind1])
        NN_ind = np.where(D[ind1] == NN)[0][0]
        NN_label = train_labels[NN_ind]
        #add to test_labels
        test_labels.append(NN_label)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def svm_classify(train_image_feats, train_labels, test_image_feats):
    """
    This function will train a one-versus-one support vector machine (SVM)
    and then use the learned classifiers to predict the category of every test image. 

    Useful functions:
    -   sklearn SVC: One-versus-One approach
          https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    -   svm.fit(X, y)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    categories = list(set(train_labels))
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    M = test_image_feats.shape[0]
    N = train_image_feats.shape[0]
    
    svm = SVC(gamma='scale')
    svm.fit(train_image_feats,train_labels)
    
    test_labels = svm.predict(test_image_feats)
    
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels


def bags_of_sifts_spm(image_paths, vocab_filename, depth):
    """
    Bags of sifts with spatial pyramid matching.

    :param image_paths: paths to N images
    :param vocab_filename: Path to the precomputed vocabulary.
          This function assumes that vocab_filename exists and contains an
          vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
          or visual word. This ndarray is saved to disk rather than passed in
          as a parameter to avoid recomputing the vocabulary every run.
    :param depth: Depth L of spatial pyramid. Divide images and compute (sum)
          bags-of-sifts for all image partitions for all pyramid levels.
          Refer to the explanation in the notebook, tutorial slide and the 
          original paper (Lazebnik et al. 2006.) for more details.

    :return image_feats: N x d matrix, where d is the dimensionality of the
          feature representation. In this case, d will equal the number of
          clusters (vocab_size) times the number of regions in all pyramid levels,
          which is 21 (1+4+16) in this specific case.
    """
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)
    
    vocab_size = vocab.shape[0]
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################
    first = 1
    M = vocab_size
    dim = M
    for i in range (len(image_paths)):
        sample = load_image_gray(image_paths[i])
        img_hist = []
        for layer in range (depth):
            x_step, y_step = np.divide(sample.shape, 2**layer)
            for ind1 in range (2**layer):
                for ind2 in range (2**layer):
                    step_size = 10
                    sample_slice = sample[int(ind1*x_step):int((ind1+1)*x_step),
                                          int(ind2*y_step):int((ind2+1)*y_step)]
                    frames, descriptors = vlfeat.sift.dsift(sample_slice, step = step_size, window_size=dim,
                                                    norm=True, fast=True)
                    D = cdist(descriptors, vocab)
                    #we got an N(i) x M array, to collapse into M array, add to closest cluster for each feature
                    histogram = np.zeros((M))
                    for ind3 in range (D.shape[0]):
                        NN = np.amin(D[ind3])
                        NN_ind = np.where(D[ind3] == NN)[0][0]
                        histogram[NN_ind] = histogram[NN_ind] + 1
                    #normalize
                    histogram = histogram / np.linalg.norm(histogram)
                    if layer > 0:
                        #L is depth -1, so according to formular it should be depth-layer
                        weight = 1/(2**(depth-layer))
                        histogram = weight*histogram            
                    img_hist.append(histogram)
                    
        img_hist = np.concatenate(img_hist).ravel()
        feats.append(img_hist)
    feats = np.vstack(feats)
    print("feats.shape")
    print(feats.shape)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats
