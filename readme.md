**A  set of simplified Yolo-FastestV2 Benchmarks on CCTSDB 2021 for Neural Network Verification**

We propose a new set of benchmarks of simplified Yolo-FastestV2 on CCTSDB 2021 for neural network verification in this repository.

**Motivation**

Last year, we proposed a simple semantic segmentation benchmark to enable neural network verification tools to be applied to autonomous driving scenarios. This year, we will introduce a more challenging benchmark, which is object detection in autonomous driving scenes. Considering the time constraints of the competition and the feasibility of verification tools, we modify Yolo-FastestV2, a classic end-to-end object detection architecture, which includes backbone, neck, and head components. In order to further reduce the computational burden, we simplified the backbone and neck, and for the head, to achieve the purpose of single object detection and to avoid performing nms operations in the model, we abandoned the box regression method and adopted landmark regression for coordinate detection instead.

In addition, previous benchmarks aimed to verify the model's robustness in the digital world. In order to make neural network verification tools more practical, we propose verifying the model's robustness in the physical world this year. Specifically, we will provide an image and its corresponding label, as well as a fixed-size patch (either 1x1 or 3x3), and we hope that participants can verify whether the model is robust after attaching the patch to any position of the image within the specified time 60s.

**Model details**

The ONNX format networks are available in the *[onnx] (net/onnx/)* folder. And the inference script(`evaluate_network.py`) of onnx models can be found in the *[src] (src/)* folder.

**Data Format**

The input images and target coordinates need to be normalized to a range of 0-1. There are three categories of targets, represented by 0 (mandatory), 1 (prohibitory), and 2 (warning).

**Data Selection**

We used the training set from CCTSDB 2021, which includes a total of 16,356 images (26,838 instances). We further divided all instances in a 9:1 ratio, resulting in a training set with 23,856 instances and a test set with 2,982 instances. We selected images with iou>0.5 and correct category classification from the test set, and ultimately, we will randomly select 16 images for verification.

**More details**

- The model input is an array with a length of 12296, containing imgs (array[0:12288]), position (array[12288:12290]), and targets (array[12290:12296]). "imgs" represents the input original image with size [1,3,64,64], where 1 is the batch size, 3 is the number of channels, and the width and height of the image are both 64. "position" represents the range for attaching the patch, with the first value representing the horizontal coordinate range, the second value representing the vertical coordinate range, and the starting point being the top-left corner of the entire image; the patch size is fixed, with 1x1 and 3x3 dimensions available. "targets" represents the target labels (predicted by the onnx model instead of the ground truth of the instance), with the first value representing the target index (since all images only contain one target, this index value is always 0), the second value representing the target category, and the third to sixth values representing the coordinates of the top-left and bottom-right corners of the target.
- The ONNX model has only one output, which is the product of iou between the predicted box and the ground-truth box, and whether the predicted category is consistent with the real category, represented as output=iou(pred_bbox,gt_bbox)*equal(pred_cls,gt_cls). If the final output for the input with the added patch is less than 0.5, the model is considered non-robust for that patch; conversely, if the final output is greater than 0.5, the model is considered robust for that patch.
