import os
import random
import argparse
import torch
from pprint import pprint
from torchvision.transforms import *
from utils import check_dir
from models.pretraining_backbone import ResNet18Backbone

from data.pretraining import DataReaderPlainImg


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_init', default='C:\\Users\\ayush\\projects\\DLLab22_CV\\ssl_dino\\results\\lr0.0005_bs128__local\\models\\ckpt_epoch9.pth',
                        type=str)
    parser.add_argument("--size", type=int, default=256, help="size of the images to feed the network")
    parser.add_argument('--output-root', type=str, default='results')
    args = parser.parse_args()

    args.output_folder = check_dir(
        os.path.join(args.output_root, "nearest_neighbors"))
    args.logs_folder = check_dir(os.path.join(args.output_folder, "logs"))

    return args


def main(args):
    # model

    model = ResNet18Backbone(args.weights_init)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('data/pretrain_weights_init.pth'))
    else:
        model.load_state_dict(torch.load('data/pretrain_weights_init.pth',
                                         map_location=torch.device('cpu'))['model'])
    #raise NotImplementedError("TODO: build model and load weights snapshot")

    # dataset
    #chose these indices to see how model works if image is blurry or occluded
    query_indices = [8,13]
    val_transform = Compose([Resize(args.size), CenterCrop((args.size, args.size)), ToTensor()])
    val_data = ImageFolder('crops/images/256/val', transform=val_transform)
    val_loader = DataLoader(dataset=val_data, shuffle=False)

    #raise NotImplementedError("Load the validation dataset (crops), use the transform above.")

    # choose/sample which images you want to compute the NNs of.
    # You can try different ones and pick the most interesting ones.

    nns = []
    for idx, img in enumerate(val_loader):
        if idx not in query_indices:
            continue
        print("Computing NNs for sample {}".format(idx))
        closest_idx, closest_dist = find_nn(model, img, val_loader, 5)
        nns.append((idx, closest_idx, closest_dist))
        raise NotImplementedError("TODO: retrieve the original NN images, save them and log the results.")


def find_nn(model, query_img, loader, k):
    """
    Find the k nearest neighbors (NNs) of a query image, in the feature space of the specified mode.
    Args:
         model: the model for computing the features
         query_img: the image of which to find the NNs
         loader: the loader for the dataset in which to look for the NNs
         k: the number of NNs to retrieve
    Returns:
        closest_idx: the indices of the NNs in the dataset, for retrieving the images
        closest_dist: the L2 distance of each NN to the features of the query image
    """
    import heapq
    query_tensor, query_label = query_img
    query_img_feature = model(query_tensor)

    q = []
    for idx, (img_tensor, label) in enumerate(loader):
        feature_tensor = model(img_tensor)
        feature_distance = ((feature_tensor - query_img_feature) ** 2).sum()

        #Minimize heapq, Therefore using negative distance
        heapq.heappush(q, (- feature_distance, idx))
        if len(q) > k:
            heapq.heappop(q)

    closest_idx, closest_dist = [], []
    while len(q) > 0:
        idx, dist = heapq.heappop(q)
        closest_dist.append(- dist)
        closest_idx.append(idx)

    return closest_idx, closest_dist

if __name__ == '__main__':
    args = parse_arguments()
    pprint(vars(args))
    print()
    main(args) 
