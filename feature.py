import os
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import cv2
import numpy as np
from tqdm import tqdm

def histogram_intersection(h1, h2):
    # import ipdb; ipdb.set_trace()
    sm = np.concatenate((np.expand_dims(h1, 1), np.expand_dims(h2, 1)), axis = 1)
    sm = np.min(sm, axis = 1)
    # import ipdb; ipdb.set_trace()
    return sm

def l2_distance(h1, h2):
    return np.linalg.norm(h1-h2)

def naive(img):
    return img.flatten()

def lbp(img_gray, no_points = 16, radius = 4, bin = 512):
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    radius = 4
    no_points = 9
    lpb_array = local_binary_pattern(img_gray, no_points, radius)
    lpb_array = np.histogram(lpb_array, bins=bin)
    # lpb_array = itemfreq(lpb_array.ravel())
    # Normalize the histogram
    # lpb_array = lpb_array[:, 1]/sum(lpb_array[:, 1])
    # import ipdb; ipdb.set_trace()
    # gt_array.append([int(gt_name.split("_")[0]), lpb_array[0]])
    return lpb_array[0]

def BGR_hist(img):
    B, G, R = img[:,:,0], img[:,:,1],img[:,:,2]
    B_array = np.histogram(B, bins=256, range = (0,255))
    G_array = np.histogram(G, bins=256, range = (0,255))
    R_array = np.histogram(R, bins=256, range = (0,255))
    # import ipdb; ipdb.set_trace()
    return np.hstack((B_array[0],G_array[0],R_array[0]))

def HSV_hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H_array = np.histogram(H, bins=360, range = (0,360))
    S_array = np.histogram(S, bins=256, range = (0,255))
    V_array = np.histogram(V, bins=256, range = (0,255))
    
    return np.hstack((H_array[0],S_array[0],V_array[0]))

def HS_hist(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H_array = np.histogram(H, bins=360, range = (0,360))
    S_array = np.histogram(S, bins=256, range = (0,255))
    
    return np.hstack((H_array[0],S_array[0]))

def HSV_hist_2_0(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H_1_array = np.histogram(H[0:64,:], bins=360, range = (0,360))
    H_2_array = np.histogram(H[64:,:], bins=360, range = (0,360))
    S_1_array = np.histogram(S[0:64,:], bins=256, range = (0,255))
    S_2_array = np.histogram(S[64:,:], bins=256, range = (0,255))
    V_1_array = np.histogram(V[0:64,:], bins=256, range = (0,255))
    V_2_array = np.histogram(V[64:,:], bins=256, range = (0,255))
    
    return np.hstack((H_1_array[0],H_2_array[0],S_1_array[0], S_2_array[0], V_1_array[0], V_2_array[0]))

def HSV_hist_2_0_nor(img):
    img = np.asarray(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), dtype = np.float)
    # import ipdb; ipdb.set_trace()
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H = H/(H+S+V)
    S = S/(H+S+V)
    V = V/(H+S+V)
    H_1_array = np.histogram(H[0:64,:], bins=360, range=(0,1))
    H_2_array = np.histogram(H[64:,:], bins=360, range=(0,1))
    S_1_array = np.histogram(S[0:64,:], bins=256, range=(0,1))
    S_2_array = np.histogram(S[64:,:], bins=256, range=(0,1))
    V_1_array = np.histogram(V[0:64,:], bins=256, range=(0,1))
    V_2_array = np.histogram(V[64:,:], bins=256, range=(0,1))
    
    return np.hstack((H_1_array[0],H_2_array[0],S_1_array[0], S_2_array[0], V_1_array[0], V_2_array[0]))

def BGR_hist_2_0(img):
    B, G, R = img[:,:,0], img[:,:,1],img[:,:,2]
    B_1_array = np.histogram(B[0:64,:], bins=256, range = (0,255))
    B_2_array = np.histogram(B[64:,:], bins=256, range = (0,255))
    G_1_array = np.histogram(G[0:64,:], bins=256, range = (0,255))
    G_2_array = np.histogram(G[64:,:], bins=256, range = (0,255))
    R_1_array = np.histogram(R[0:64,:], bins=256, range = (0,255))
    R_2_array = np.histogram(R[64:,:], bins=256, range = (0,255))

    return np.hstack((B_1_array[0], B_2_array[0], G_1_array[0], G_2_array[0], R_1_array[0], R_2_array[0]))


def BGR_hist_2_2(img):
    B, G, R = img[:,:,0], img[:,:,1],img[:,:,2]
    B_1_array = np.histogram(B[0:64,0:32], bins=256, range = (0,255))
    B_2_array = np.histogram(B[64:,0:32], bins=256, range = (0,255))
    G_1_array = np.histogram(G[0:64,0:32], bins=256, range = (0,255))
    G_2_array = np.histogram(G[64:,0:32], bins=256, range = (0,255))
    R_1_array = np.histogram(R[0:64,0:32], bins=256, range = (0,255))
    R_2_array = np.histogram(R[64:,0:32], bins=256, range = (0,255))

    B_3_array = np.histogram(B[0:64,32:], bins=256, range = (0,255))
    B_4_array = np.histogram(B[64:,32:], bins=256, range = (0,255))
    G_3_array = np.histogram(G[0:64,32:], bins=256, range = (0,255))
    G_4_array = np.histogram(G[64:,32:], bins=256, range = (0,255))
    R_3_array = np.histogram(R[0:64,32:], bins=256, range = (0,255))
    R_4_array = np.histogram(R[64:,32:], bins=256, range = (0,255))

    return np.hstack((B_1_array[0],B_2_array[0],B_3_array[0],B_4_array[0],G_1_array[0],G_2_array[0],G_3_array[0],G_4_array[0],R_1_array[0],R_2_array[0],R_3_array[0],R_4_array[0]))

def HSV_hist_2_2(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H_1_array = np.histogram(H[0:64,0:32], bins=360, range = (0,360))
    H_2_array = np.histogram(H[64:,0:32], bins=360, range = (0,360))
    S_1_array = np.histogram(S[0:64,0:32], bins=256, range = (0,255))
    S_2_array = np.histogram(S[64:,0:32], bins=256, range = (0,255))
    V_1_array = np.histogram(V[0:64,0:32], bins=256, range = (0,255))
    V_2_array = np.histogram(V[64:,0:32], bins=256, range = (0,255))

    H_3_array = np.histogram(H[0:64,32:], bins=360, range = (0,360))
    H_4_array = np.histogram(H[64:,32:], bins=360, range = (0,360))
    S_3_array = np.histogram(S[0:64,32:], bins=256, range = (0,255))
    S_4_array = np.histogram(S[64:,32:], bins=256, range = (0,255))
    V_3_array = np.histogram(V[0:64,32:], bins=256, range = (0,255))
    V_4_array = np.histogram(V[64:,32:], bins=256, range = (0,255))

    return np.hstack((H_1_array[0],H_2_array[0],H_3_array[0],H_4_array[0],S_1_array[0],S_2_array[0],S_3_array[0],S_4_array[0],V_1_array[0],V_2_array[0],V_3_array[0],V_4_array[0]))

def BGR_hist_4_0(img):
    B, G, R = img[:,:,0], img[:,:,1],img[:,:,2]
    B_1_array = np.histogram(B[0:32,:], bins=256, range = (0,255))
    B_2_array = np.histogram(B[32:64,:], bins=256, range = (0,255))
    B_3_array = np.histogram(B[64:96,:], bins=256, range = (0,255))
    B_4_array = np.histogram(B[96:,:], bins=256, range = (0,255))
    G_1_array = np.histogram(G[0:32,:], bins=256, range = (0,255))
    G_2_array = np.histogram(G[32:64,:], bins=256, range = (0,255))
    G_3_array = np.histogram(G[64:96,:], bins=256, range = (0,255))
    G_4_array = np.histogram(G[96:,:], bins=256, range = (0,255))
    R_1_array = np.histogram(R[0:32,:], bins=256, range = (0,255))
    R_2_array = np.histogram(R[32:64,:], bins=256, range = (0,255))
    R_3_array = np.histogram(R[64:96,:], bins=256, range = (0,255))
    R_4_array = np.histogram(R[96:,:], bins=256, range = (0,255))

    return np.hstack((B_1_array[0],
        B_2_array[0],
        B_3_array[0],
        B_4_array[0],
        G_1_array[0],
        G_2_array[0],
        G_3_array[0],
        G_4_array[0],
        R_1_array[0],
        R_2_array[0],
        R_3_array[0],
        R_4_array[0]))

def HSV_hist_4_0(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = img[:,:,0], img[:,:,1],img[:,:,2]
    H_1_array = np.histogram(H[0:32,:], bins=360, range = (0,360))
    H_2_array = np.histogram(H[32:64,:], bins=360, range = (0,360))
    H_3_array = np.histogram(H[64:96,:], bins=360, range = (0,360))
    H_4_array = np.histogram(H[96:,:], bins=360, range = (0,360))
    S_1_array = np.histogram(S[0:32,:], bins=256, range = (0,255))
    S_2_array = np.histogram(S[32:64,:], bins=256, range = (0,255))
    S_3_array = np.histogram(S[64:96,:], bins=256, range = (0,255))
    S_4_array = np.histogram(S[96:,:], bins=256, range = (0,255))
    V_1_array = np.histogram(V[0:32,:], bins=256, range = (0,255))
    V_2_array = np.histogram(V[32:64,:], bins=256, range = (0,255))
    V_3_array = np.histogram(V[64:96,:], bins=256, range = (0,255))
    V_4_array = np.histogram(V[96:,:], bins=256, range = (0,255))

    return np.hstack((H_1_array[0],
        H_2_array[0],
        H_3_array[0],
        H_4_array[0],
        S_1_array[0],
        S_2_array[0],
        S_3_array[0],
        S_4_array[0],
        V_1_array[0],
        V_2_array[0],
        V_3_array[0],
        V_4_array[0]))