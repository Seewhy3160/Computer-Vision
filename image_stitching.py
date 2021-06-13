import numpy as np
from skimage import filters
from skimage.feature import corner_peaks
from scipy.spatial.distance import cdist
from scipy.ndimage.filters import convolve
import math

from utils import pad, get_output_space, warp_image

             
def harris_corners(img, window_size=3, k=0.04):
    """
    Compute Harris corner response map. Follow the math equation
    R=Det(M)-k(Trace(M)^2).

    Hint:
        You may use the function scipy.ndimage.filters.convolve, 
        which is already imported above
        
    Args:
        img: Grayscale image of shape (H, W)
        window_size: size of the window function
        k: sensitivity parameter

    Returns:
        response: Harris response image of shape (H, W)
    """

    H, W = img.shape
    window = np.ones((window_size, window_size))

    response = np.zeros((H, W))

    dx = filters.sobel_v(img)
    dy = filters.sobel_h(img)

    ### YOUR CODE HERE
    # Find M
    # sum[:x, :y](window[x.y]*[[dx*dx, dx*dy],[dx*dy, dy*dy]])
    # need to find dx*dx, dy*dy, dx*dy
    dxdx = dx*dx
    dydy = dy*dy
    dxdy = dx*dy
    # no idea how to do it like the statement above, convolve by term using window
    w_dxdx = convolve(dxdx, window)
    w_dxdy = convolve(dxdy, window)
    w_dydy = convolve(dydy, window)
    
    ### this part didn't work
    ##thus M is
    # M = [[w_dxdx, w_dxdy],
    #      [w_dxdy, w_dydy]]
    ##get response = R=Det(M)-k(Trace(M)^2)
    # response = np.linalg.det(M) - k*np.trace(np.square(M))
    # cause response is (512,512) and the RHS is (2,2)
    # may i ask what is the right way to go about it?
    # no choice, gotta use for loops
    
    for ind1 in range(H):
        for ind2 in range(W):
            M = [[w_dxdx[ind1, ind2], w_dxdy[ind1, ind2]],
                 [w_dxdy[ind1, ind2], w_dydy[ind1, ind2]]] 
            response[ind1, ind2] = np.linalg.det(M) - k*np.trace(np.square(M))
    ### END YOUR CODE
    return response


def simple_descriptor(patch):
    """
    Describe the patch by normalizing the image values into a standard 
    normal distribution (having mean of 0 and standard deviation of 1) 
    and then flattening into a 1D array. 
    
    The normalization will make the descriptor more robust to change 
    in lighting condition.
    
    Hint:
        If a denominator is zero, divide by 1 instead.
    
    Args:
        patch: grayscale image patch of shape (h, w)
    
    Returns:
        feature: 1D array of shape (h * w)
    """
    feature = []
    
    ### YOUR CODE HERE
    # to calculate for each value, we need existing standard deviation and mean
    # for each value, we can deduct current mean, 
    # for std of 1, we can divide the remainder by the current standard deviation
    patch_flat = patch.flatten()
    mean = np.mean(patch_flat)
    std = np.std(patch_flat)
    if std == 0:
        std = 1
    #amax = np.amax(patch)
    #amin = np.amax(patch)
    
    # apply to whole array
    #feature = np.divide(np.subtract(patch, mean), std) 
    feature = (patch_flat - mean)/std
    
    # check by printing feature mean and std
    #print("this is feature mean")
    #print(np.mean(feature))
    #print("this is feature std")
    #print(np.std(feature))
    
    ### END YOUR CODE
    
    return feature


def describe_keypoints(image, keypoints, desc_func, patch_size=16):
    """
    Args:
        image: grayscale image of shape (H, W)
        keypoints: 2D array containing a keypoint (y, x) in each row
        desc_func: function that takes in an image patch and outputs
            a 1D feature vector describing the patch
        patch_size: size of a square patch at each keypoint
                
    Returns:
        desc: array of features describing the keypoints
    """

    image.astype(np.float32)
    desc = []

    for i, kp in enumerate(keypoints):
        y, x = kp
        patch = image[y-(patch_size//2):y+((patch_size+1)//2),
                      x-(patch_size//2):x+((patch_size+1)//2)]
        desc.append(desc_func(patch))
    return np.array(desc)


def match_descriptors(desc1, desc2, threshold=0.5):
    """
    Match the feature descriptors by finding distances between them. A match is formed 
    when the distance to the closest vector is much smaller than the distance to the 
    second-closest, that is, the ratio of the distances should be smaller
    than the threshold. Return the matches as pairs of vector indices.
    
    Hint: 
        You may use `scipy.spatial.distance.cdist` to compute distance between desc1 
        and desc2, which is already imported above
    
    Args:
        desc1: an array of shape (M, P) holding descriptors of size P about M keypoints
        desc2: an array of shape (N, P) holding descriptors of size P about N keypoints
        
    Returns:
        matches: an array of shape (Q, 2) where each row holds the indices of one pair 
        of matching descriptors
    """
    matches = []
    
    M = desc1.shape[0]
    N = desc2.shape[0]

    ### YOUR CODE HERE
    # M is the length of desc1
    # N is the length of desc2
    # p is the size of each descriptor point
    # assuming their p are the same, no need to get P, can se cdist
    # realised need p for shape
    P = desc1.shape[1]
    # distance is like in knn, in all the feature space
    #
    # for each point in desc1, find the first closest and second closest
    # keep list of dist
    dist = cdist(desc1, desc2, 'euclidean')
    for ind1 in range (M):
            
        # now find indice of closest and 2nd closest and their values
        closest = np.amin(dist[ind1])
        closest_ind = np.where(dist[ind1] == np.amin(dist[ind1]))[0]
        
        # since we know what is closest and its indice, we can divide that entry by the threshold,
        dist[ind1][closest_ind] = (dist[ind1][closest_ind])/threshold
        
        # if we get back the same entry, it fits the requirement/ no 2nd closest in threshold
        nxt_closest = np.amin(dist[ind1])
        nxt_closest_ind = np.where(dist[ind1] == np.amin(dist[ind1]))[0]
        if closest_ind == nxt_closest_ind:
            matches.append([ind1, closest_ind])
    # if distance to first/distance to second < threshold, return the indices of point and first closest to matches
    # in the end matches would have Q pairs of keypoints INDICES
    ### END YOUR CODE
    #np.reshape(matches, (,2))
    matches = np.asarray(matches)
    #print("these are matches")
    #print("formats fk")
    #print(matches)
    
    return matches


def fit_affine_matrix(p1, p2):
    """ Fit affine matrix such that p2 * H = p1 
    
    Args:
        p1: an array of shape (M, P)
        p2: an array of shape (M, P)
        
    Return:
        H: a matrix of shape (P * P) that transform p2 to p1.
    """

    assert (p1.shape[0] == p2.shape[0]),\
        'Different number of points in p1 and p2'
    p1 = pad(p1)
    p2 = pad(p2)

    H = np.linalg.lstsq(p2, p1)[0]

    # Sometimes numerical issues cause least-squares to produce the last
    # column which is not exactly [0, 0, 1]
    H[:,2] = np.array([0, 0, 1])
    return H


def ransac(keypoints1, keypoints2, matches, n_iters=200, threshold=20):
    """
    Use RANSAC to find a robust affine transformation

        1. Select random set of matches
        2. Compute affine transformation matrix
        3. Compute inliers
        4. Keep the largest set of inliers
        5. Re-compute least-squares estimate on all of the inliers

    Args:
        keypoints1: M1 x 2 matrix, each row is a point
        keypoints2: M2 x 2 matrix, each row is a point
        matches: N x 2 matrix, each row represents a match
            [index of keypoint1, index of keypoint 2]
        n_iters: the number of iterations RANSAC will run
        threshold: the number of threshold to find inliers

    Returns:
        H: a robust estimation of affine transformation from keypoints2 to
        keypoints 1
    """
    N = matches.shape[0]
    n_samples = int(N * 0.2)

    matched1 = pad(keypoints1[matches[:,0]])
    matched2 = pad(keypoints2[matches[:,1]])

    max_inliers = np.zeros(N)
    n_inliers = 0

    # RANSAC iteration start
    ### YOUR CODE HERE
    # format check matched1 and 2
    #print("this is matched 1")
    #print(matched1)
    
    first = 1
    max_inlier_size = 0
    features = keypoints1.shape[1]
    
    # for n_interations
    for ind1 in range (n_iters):
        # select n random set of matches
        # lets call it chosen cause it has that cool feeling to it
        chosen = np.random.choice(N, n_samples)
        p1 = keypoints1[matches[chosen,0]]
        p2 = keypoints1[matches[chosen,1]]
        
        # sanity check print p1 and p2
        #if first:
            #print("this is p1")
            #print(p1)
            #print("this is p2")
            #print(p2)
            #first = 0
        
        # p1, p2 have M points with P features
        # for set
        # compute H, affine transformation of the two points
        # using fit_affine_matrix(p1, p2)
        H = fit_affine_matrix(p1, p2)
        
        # sanity check H
        #if first:
            #print("this is H")
            #print(H)
            #print("this is matched2.dot(H)")
            #print(matched2.dot(H))
            #print("this is matched1")
            #print(matched1)
            #print("this is matched2.dot(H) - matched1")
            #print(matched2.dot(H) - matched1)
            #first = 0
        
        # then check the Sum of Squared Difference (SSD)
        # of each point in matches
        # with H * M1 and M2 with np.linalg.norm
        # probably np.linalg.norm(matched2.dot(H), matched1)
        # SSD = np.linalg.norm(np.dot(matched2,H) - matched1, ord = 2, axis = 1)
        # this SSD is a piece of 
        #wasted 3 hours
        # instructions unclear, similar problems all over slack
        # sanity check SSD
        # my own SSD    
        SSD = np.sum((matched2.dot(H)-matched1)**2, axis = 1)
        #if first:
            #print("N")
            #print(N)
            #print("this is SSD")
            #print(SSD)
            #print("this is shape of SSD")
            #print(SSD.shape)
            #first = 0
        
        # find pairs with SSD less than threshold
        matches_passed_ind = np.where(np.asarray(SSD) < threshold)[0]
        #this is not working lets just for loops, for loops are friends
        #matches_passed_ind = []
        #SSD = np.zeros((N))
        #for ind3 in range (N):
            #SSD
            #print("this is SSD[ind3]")
            #print(SSD[ind3])
            #if (SSD[ind3] < threshold):
                #matches_passed_ind.push(ind3)
            
        # sanity check matches passed indices
        #if first:
            #print(matches_passed_ind)
            #print(matches_passed_ind.shape)
        #if matches_passed_ind.shape[0] > 0 and first:
            #print("this are passing indices")
            #print(matches_passed_ind)
            #print("this is matches_passed.shape")
            #print(matches_passed_ind.shape)
            #first = 0
        # assign to inliners
        #if first:
            #print("this is matches_passed_ind.shape[0]")
            #print(matches_passed_ind.shape[0])
            #first=0
        if (matches_passed_ind.size > max_inlier_size):
            #print("this is a hit")
            max_inlier_size = matches_passed_ind.shape[0]
            max_inliers = matches_passed_ind
            #print("max_inliers")
            #print(max_inliers)
                           
    
    # keep largest group of inliners
    # recalculate SSD get H first
    # sanity check max_inliers
    #print("max_inliers")
    #print(max_inliers)
    #print("matches[max_inliers]")
    #print(matches[max_inliers])
    #print("keypoints1[matches[max_inliers, 0]]")
    #print(keypoints1[matches[max_inliers, 0]])
    #print("keypoints2[matches[max_inliers, 1]")
    #print(keypoints2[matches[max_inliers, 1]])
    # lots of tears and hours lost, finally did it, dammit python and numpy
    H = fit_affine_matrix(keypoints1[matches[max_inliers, 0]], keypoints2[matches[max_inliers, 1]])
    
    ### END YOUR CODE
    
    return H, matches[max_inliers]


def sift_descriptor(patch):
    """
    Implement a simplifed version of Scale-Invariant Feature Transform (SIFT).
    Paper reference: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    
    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each length of 16/4=4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
        patch: grayscale image patch of shape (h, w)

    Returns:
        feature: 1D array of shape (128, )
    """
    
    dx = filters.sobel_v(patch)
    dy = filters.sobel_h(patch)
    histogram = np.zeros((4,4,8))
    
    ### YOUR CODE HERE
    first = 1
    
    # get magnitude and angle for each pixel in 16*16
    # store in patch_mag_ang
    #patch_mag_ang = np.zeros((16,16,2))
    #mag = 0
    #ang = 1
    # does not work, no way to access [:,:,mag] ValueError: invalid number of arguments
    # cannot use [...,mag] ValueError: invalid number of arguments
    # no choice but to break them up, please send help
    #patch_mag = np.zeros((16, 16))
    #patch_ang = np.zeros((16, 16))
    
    # from equations
    #patch_mag_ang[:,:,mag] = np.sqrt(dx*dx + dy*dy)
    #patch_mag_ang[:,:,ang] = np.arctan2(dx/dy)
    patch_mag = np.sqrt(dx*dx + dy*dy)
    patch_ang = np.arctan2(dx, dy)
    
    # split into 4 by 4
    for ind1 in range(4):
        for ind2 in range(4):
            # from their respective pixels
            for ind11 in range(ind1*4, ind1*4+4):
                for ind22 in range(ind2*4, ind2*4+4):
                    # split into 8 from (-pi, pi) by adding pidividing by 1/4*pi
                    histo_num = (patch_ang[ind11, ind22] + np.pi) / (1/4 * np.pi)
                    
                    # add into the correct histogram
                    # by ratio
                    histo_num1 = int(np.floor(histo_num)%8)
                    histo_num2 = int(np.ceil(histo_num)%8)
                    histo_num_ratio = histo_num - histo_num1
                    #histogram[ind1, ind2, histo_num] += patch_mag[ind11, ind22]# no ratio
                    # with ratio
                    histogram[ind1, ind2, histo_num1] += (1-histo_num_ratio) * patch_mag[ind11, ind22]
                    histogram[ind1, ind2, histo_num2] += histo_num_ratio * patch_mag[ind11, ind22]
    
    # gotten histograms in 8 directions
    # sanity check
    #print("histogram")
    #print(histogram)
    
    # normalize
    histo_norm = np.zeros((4,4,8))
    for ind3 in range (4):
        for ind4 in range (4):
            #normal for the cell is
            normal = np.linalg.norm(histogram[ind3, ind4])
            if normal == 0:
                histo_norm[ind3,ind4] = 0
                
            else:
                histo_norm[ind3,ind4] = histogram[ind3,ind4]/ normal
            feature = histo_norm.flatten()
    #print("histo_norm")
    #print(histo_norm)
    
    #print("feature")
    #print(feature)
        
    ### END YOUR CODE
    
    return feature
