import torch
import os
import cv2
import numpy as np
import argparse
from pytorch.model.detector import DetectorMulInputOneTensor


def test(pthfile, imgpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DetectorMulInputOneTensor(3, 2, True, mode="test").to(device)
    model.load_state_dict(torch.load(pthfile, map_location=device))

    # sets the module in eval node
    model.eval()

    for imagename in os.listdir(imgpath):
        if 'jpg' in imagename:
            org_img = cv2.imread(os.path.join(imgpath, imagename))
            img = torch.from_numpy(org_img).reshape(1, 64, 64, 3)
            img = img.permute(0, 3, 1, 2)
            img = img.to(device).float() / 255.0
            label_path = os.path.join(imgpath, imagename).replace('jpg', 'txt')
            org_position = torch.tensor([-100000000, -100000000])
            label = []
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    l = line.strip().split(" ")
                    label.append([0, l[0], l[1], l[2], l[3], l[4]])
            target = np.array(label, dtype=np.float32)

            input = torch.zeros(12296)
            input[0:12288] = img.reshape(-1)
            input[12288:12290] = org_position
            input[12290:] = torch.from_numpy(target[0])
            out = model(input)
            print(out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pthfile', type=str, required=True, help='the pytorch model')
    parser.add_argument('--imgpath', type=str, required=True, help='test image path')
    args = parser.parse_args()
    test(args.pthfile, args.imgpath)
