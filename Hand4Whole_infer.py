import os
import cv2
import sys
import torch
import pickle
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
from ultralytics import YOLO
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

sys.path.insert(0, osp.join('..', 'main'))
sys.path.insert(0, osp.join('..', 'data'))
sys.path.insert(0, osp.join('..', 'common'))
from config import cfg
from model import get_model
from utils.preprocessing import load_img, process_bbox, generate_patch_image
from utils.human_models import smpl_x
from utils.vis import render_mesh, save_obj

def parse_args():
    # arg.img_path: .../examples/raw_mmWave_human/
    # args.output_folder: out
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--img_path', type=str, default='input.png')
    parser.add_argument('--output_folder', type=str, default='output_hand4')
    args = parser.parse_args()
    # test gpus
    if not args.gpu_ids:
        assert 0, print("Please set proper gpu ids")

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    return args

args = parse_args()
cfg.set_args(args.gpu_ids)
cudnn.benchmark = True

# snapshot load
model_path = './snapshot_6.pth.tar'
assert osp.exists(model_path), 'Cannot find model at ' + model_path
print('Load checkpoint from {}'.format(model_path))
model = get_model('test')
model = DataParallel(model).cuda()
ckpt = torch.load(model_path)
model.load_state_dict(ckpt['network'], strict=False)
model.eval()
modelYolo = YOLO("yolov8n.pt")

# prepare input image
transform = transforms.ToTensor()

for person in tqdm(os.listdir(args.img_path)):
    person_path = os.path.join(args.img_path, person)
    output_path = osp.join(person_path, args.output_folder)
    os.makedirs(output_path, exist_ok=True)
    already_flag = osp.join(output_path, '44_render_original_img.jpg')
    if os.path.exists(already_flag):
        continue
    for i in range(45):
        img_name = f'\\rgb\\{i}.jpg'
        img_path = person_path + img_name
        print(img_path)
        if not os.path.exists(img_path):
            continue
        original_img = load_img(img_path)
        original_img_height, original_img_width = original_img.shape[:2]
        print(original_img_height, original_img_width)
        # prepare bbox
        result = modelYolo.predict(source=original_img, show=False, classes=[0])
        bbox = result[0].boxes.xyxy
        bbox = bbox.cpu().numpy().tolist()
        if len(bbox) == 0:
            bbox = [214, 90, 642, 540] # xmin, ymin, width, height
        else:
            bbox = bbox[0]
            bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]
        print(bbox)
        bbox = process_bbox(bbox, original_img_width, original_img_height)
        img, img2bb_trans, bb2img_trans = generate_patch_image(original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape)
        img = transform(img.astype(np.float32))/255
        img = img.cuda()[None,:,:,:]

        # forward
        inputs = {'img': img}
        targets = {}
        meta_info = {}
        with torch.no_grad():
            out = model(inputs, targets, meta_info, 'test')
        mesh_cam = out['smplx_mesh_cam'].detach().cpu().numpy()[0]
        mesh_nn = out['smplx_mesh_nn'].detach().cpu().numpy()[0]

        # save mesh
        save_obj(mesh_nn, smpl_x.face, os.path.join(output_path, f'frame{i}.obj'))

        vis_img = original_img.copy()
        focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
        princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
        rendered_img = render_mesh(vis_img, mesh_cam, smpl_x.face, {'focal': focal, 'princpt': princpt})
        cv2.imwrite(os.path.join(output_path, f'{i}_render_original_img.jpg'), rendered_img)

        # save SMPL-X parameters
        para = {
            'smplx_root_pose': out['smplx_root_pose'].detach().cpu().numpy(),
            'smplx_body_pose': out['smplx_body_pose'].detach().cpu().numpy(),
            'smplx_lhand_pose': out['smplx_lhand_pose'].detach().cpu().numpy(),
            'smplx_rhand_pose': out['smplx_rhand_pose'].detach().cpu().numpy(),
            'smplx_shape': out['smplx_shape'].detach().cpu().numpy(),
            'smplx_joint_cam': out['smplx_joint_cam'].detach().cpu().numpy(),
            'body_joint_hm': out['body_joint_hm'].detach().cpu().numpy(),
            'hand_feat': out['hand_feat'].detach().cpu().numpy()
        }
        with open(os.path.join(output_path, f'{i}_person_0.pkl'), 'wb') as file:
            pickle.dump(para, file)
