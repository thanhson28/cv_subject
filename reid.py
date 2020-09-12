import os
from skimage.feature import local_binary_pattern
from scipy.stats import itemfreq
from sklearn.preprocessing import normalize
import feature
import cv2
import numpy as np
from time import time
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-b','--benchmark', action='store_true',
                    help='run benchmark')
parser.add_argument('-p','--path', type=str,
                    help='path to image')
parser.add_argument('-t','--type', type=str, default='HSV_hist_4_0',
                    help='feature type')     
parser.add_argument('-m','--metric', type=str, default='histogram_intersection',
                    help='distance metric')
parser.add_argument('-r','--rank', type=int, default=1,
                    help='rank')                                                

args = parser.parse_args()

def benchmark():
    feature_function = getattr(feature, args.type)
    metric_function = getattr(feature, args.metric)
    query_path = "./Market-1501-v15.09.15/query"
    gt_path = "./Market-1501-v15.09.15/gt_bbox"

    list_query_name = os.listdir(query_path)
    list_gt_name  = os.listdir(gt_path)

    gt_array = []
    cam_views = []
    extracted = False

    if not extracted:
        print("Extracting feature from gallery")
        for gt_name in tqdm(list_gt_name):
            if gt_name in list_query_name or '.jpg' not in gt_name:
                continue
            try:
                img = cv2.imread(os.path.join(gt_path, gt_name))
                img = cv2.resize(img, (64, 128))
            except Exception as e:
                print(e)
            # lpb_array = my_lbp(img_gray)
            feature_array = feature_function(img)
            # lpb_array = img.flatten()
            gt_array.append([int(gt_name.split("_")[0]), feature_array, gt_name])
            cam_views.append(gt_name.split("_")[1])
        
        np.save(args.type + '.npy', gt_array)
        np.save('cam_views.npy', gt_array)

    data = np.load(args.type + '.npy', allow_pickle=True)
    cam_views = np.load('cam_views.npy', allow_pickle=True)

    top_1 = 0
    top_5 = 0
    top_10 = 0
    top_15 = 0
    top_20 = 0

    accepted_number = len(list_query_name)

    feature_time = 0
    time_process = 0
    for query_name in tqdm(list_query_name):
        if '.jpg' not in query_name:
            continue
        try:
            img = cv2.imread(os.path.join(gt_path, query_name))
            img = cv2.resize(img, (64, 128))
        except Exception as e:
            print(e)
        
        start = time()
        feature_array = feature_function(img)
        feature_time += time() - start

        dist_array = []
        for obj in data:
            dist = metric_function(feature_array, obj[1])
            if args.metric == 'histogram_intersection':
                dist = 1 - np.sum(dist)/(256*3*128*64)
            # dist = np.linalg.norm(lpb_array-obj[1])
            dist_array.append([obj[0], dist])
        # import ipdb; ipdb.set_trace()
        sorted_by_second = sorted(dist_array, key=lambda tup: tup[1])
        sorted_by_second = [ele[0] for ele in sorted_by_second]
        time_process += time()- start
        # import ipdb; ipdb.set_trace()
        print(sorted_by_second[0])

        query_id = int(query_name.split("_")[0])
        print(query_id)
        if query_id in sorted_by_second[:1]:
            top_1 +=1
        if query_id in sorted_by_second[:5]:
            top_5 +=1
        if query_id in sorted_by_second[:10]:
            top_10 +=1
        if query_id in sorted_by_second[:15]:
            top_15 +=1
        if query_id in sorted_by_second[:20]:
            top_20 +=1

        print("top_1 :", top_1/accepted_number)
        print("top_5 :", top_5/accepted_number)
        print("top_10:", top_10/accepted_number)
        print("top_15:", top_15/accepted_number)
        print("top_20:", top_20/accepted_number)
    fl = open(args.type+"_"+args.metric+".txt", "w")
    fl.write(str(top_1/accepted_number)+" "+str(top_5/accepted_number)+" "+str(top_10/accepted_number)+" "+str(top_15/accepted_number)+" "+str(top_20/accepted_number)+"\n")
    fl.write(str(feature_time/accepted_number)+"\n")
    fl.write(str(time_process/accepted_number)+ "\n")
    fl.close()

def test(path, f_type, metric, rank):
    data_base = "./Market-1501-v15.09.15/gt_bbox"
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 128))
    feature_function = getattr(feature, f_type)
    metric_function = getattr(feature, metric)
    probe_feature = feature_function(img)
    list_img = os.listdir(data_base)
    results = []
    id_array = []
    for img_name in tqdm(list_img):
        if '.jpg' not in img_name:
            continue
        gallery_img = cv2.imread(os.path.join(data_base, img_name))
        gallery_img = cv2.resize(gallery_img, (64, 128))
        gallery_feature = feature_function(gallery_img)
        dist = metric_function(probe_feature, gallery_feature)
        if metric == 'histogram_intersection':
            dist = 1 - np.sum(dist)/(256*3*128*64)
        # import ipdb; ipdb.set_trace()
        idx = int(img_name.split("_")[0])
        results.append([dist, img_name, idx])
    # import ipdb; ipdb.set_trace()
    results = sorted(results, key=lambda tup: tup[0])
    results = results[:rank]
    rs_array = [cv2.imread(os.path.join(data_base, rs[1])) for rs in results]
    rs_array = [cv2.putText(rs_array[i], str(results[i][2]),(10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA) for i in range(len(rs_array))]
    space = np.asarray(np.ones(img.shape)*255, dtype = np.uint8)
    rs_array = np.hstack([img, space,  np.hstack(rs_array)])
    # cv2.imshow("result",rs_array )
    # cv2.waitKey(0)
    cv2.imwrite("result.png", rs_array)

        

if __name__== "__main__":
    if args.benchmark:
        print("feature type: ", args.type)
        print("distance metric: ", args.metric)
        benchmark()
    else:
        if args.path is None:
            print("Missing image path")
        else:
            test(args.path, args.type, args.metric, args.rank)