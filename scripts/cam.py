import argparse
import cv2
import sys
import os
import numpy as np
import torch
from torchvision import models
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
sys.path.append(os.path.split(rootPath)[0])

from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, LayerCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from models.vgg import vgg16
from utils.LoadData import valid_data_loader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument("--img_dir", type=str, default='', help='Directory of validation images')
    parser.add_argument("--test_list", type=str, default='dataset/valid.txt')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument('--aug_smooth', action='store_true', help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam', 'ablationcam', 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument("--checkpoint", type=str, default="result/checkpoints/best.pth")
    parser.add_argument("--model_name", type=str, default="vgg16")
    parser.add_argument("--layer_num", type=str, default="30")
    parser.add_argument("--save_cam_dir", default="result/localization_maps/")

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


if __name__ == '__main__':
    args = get_args()
    epoch_num = os.path.basename(args.checkpoint).split('.')[0]
    stage = os.path.basename(args.test_list).split('.')[0]
    stage = stage.split("_")[0]
    args.stage = stage
    pred_label_csv_name = os.path.join(args.save_cam_dir, str(args.layer_num) + "_pred_label_slice_" + stage + ".csv")
    
    output_dir_cam = os.path.join(args.save_cam_dir, "cam")
    output_dir_graycam = os.path.join(args.save_cam_dir, "cam_graycam")
    if not os.path.exists(output_dir_cam):
        os.makedirs(output_dir_cam)
    if not os.path.exists(output_dir_graycam):
        os.makedirs(output_dir_graycam)    

    """load data"""
    data_loader = valid_data_loader(args)

    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    """load pretrained model"""
    if args.model_name == "vgg16":
        model = vgg16(pretrained=False)
    model = model.cuda()
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location='cuda:0')
    model.load_state_dict(ckpt['model'], strict=True)

    # Choose the target layer you want to compute the visualization for.
    if args.model_name == "vgg16":
        if int(args.layer_num) > 30 or int(args.layer_num) < 0:
            print("Please check the target layer number")
        else:
            target_layers = [model.features[int(args.layer_num)]]
    else:
        print("Please check the model name")

    pred_label_csv = []
    pred_cls_all, label_cls_all = [], []
    for idx, data in enumerate(data_loader):
        print("[%04d/%04d]" % (idx, len(data_loader)), end="\r")
        img, label, gt_map, img_name, raw_shape = data
        org_slice = img[0,0,:,:].unsqueeze(dim=0).float()
        gt_map = gt_map.float()
        img = img.cuda()
        img.requires_grad_()

        """generate CAM"""
        input_tensor = img
        target_category = None
        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1
            grayscale_cam, pred_cls = cam(input_tensor=input_tensor,
                                                 aug_smooth=args.aug_smooth,
                                                 eigen_smooth=args.eigen_smooth)  #  target_category

            H, W, _ = raw_shape
            H, W = np.array(H[0]), np.array(W[0])
            rgb_img = F.interpolate(img, [H, W])
            rgb_img = rgb_img.squeeze().detach().cpu().numpy()
            rgb_img = np.transpose(rgb_img, (1,2,0))
            rgb_img = np.float32(rgb_img)

            # grayscale_cam
            grayscale_cam = grayscale_cam[0, :]
            grayscale_cam = cv2.resize(grayscale_cam, [int(W), int(H)])
            
            # cam on image
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)                # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
        
        """save for classification results computation"""
        cls_label = 1 if label[0][1] else 0
        pred_cls_all.append(pred_cls)
        label_cls_all.append(cls_label)

        """save cam and predicted label"""
        save_name = os.path.basename(img_name[0]).split('.')[0]
        cv2.imwrite(f'{output_dir_graycam}/{save_name}.jpg', np.uint8(255 * grayscale_cam))
        cv2.imwrite(f'{output_dir_cam}/{save_name}.jpg', cam_image)        
        pred_label_csv.append({"filename": save_name, "label": cls_label, "pred_label": target_category})        

    dataframe = pd.DataFrame(pred_label_csv, columns=["filename", "label", "pred_label"])
    dataframe.to_csv(pred_label_csv_name, index=False)
    
    """compute clssification results"""
    val_cls_acc = accuracy_score(label_cls_all, pred_cls_all)
    val_cls_prec = precision_score(label_cls_all, pred_cls_all, zero_division=0)
    val_cls_recall = recall_score(label_cls_all, pred_cls_all, zero_division=0)
    val_cls_f1 = f1_score(label_cls_all, pred_cls_all, zero_division=0)
    print('classification accuracy: %.4f, precision: %.4f, recall: %.4f, f1: %.4f' % (val_cls_acc, val_cls_prec, val_cls_recall, val_cls_f1))

