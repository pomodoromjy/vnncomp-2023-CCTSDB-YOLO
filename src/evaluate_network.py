import os
import torch
import torch.onnx
import onnxruntime
import cv2
import numpy as np

onnxpath = '../net/onnx/patch-1.onnx'
imgpath = '../dataset/patch-1/images/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_onnx():

  for imagename in os.listdir(imgpath):
    if 'jpg' in imagename:

      # original image and ground truth
      org_img = cv2.imread(os.path.join(imgpath, imagename))
      img = torch.from_numpy(org_img).reshape(1, 64, 64, 3)
      img = img.permute(0, 3, 1, 2)
      img = np.array(img.to(device).float() / 255.0)
      label_path = os.path.join(imgpath.replace('images', 'gt_labels'), imagename).replace('jpg', 'txt')
      org_position = [-100000000, -100000000]
      label = []
      with open(label_path, 'r') as f:
        for line in f.readlines():
          l = line.strip().split(" ")
          label.append([0, l[0], l[1], l[2], l[3], l[4]])
      target = np.array(label, dtype=np.float32)

      input = np.zeros(12296)
      input[0:12288] = img.reshape(-1)
      input[12288:12290] = org_position
      input[12290:] = target[0]


      ort_session = onnxruntime.InferenceSession(onnxpath)
      # ONNX RUNTIME
      inname = [input.name for input in ort_session.get_inputs()]
      outname = [output.name for output in ort_session.get_outputs()]
      org_ort_inputs = {inname[0]: input.astype('float32')}

      org_ort_outs = ort_session.run(outname, org_ort_inputs)
      print("org_ort_outs:", org_ort_outs)



if __name__ == '__main__':
  test_onnx()
