import numpy as np
import os
import PIL.Image as Image
import sys
import numpy.random as random

def get_random_images(path,random,length,seed):
    list = []
    for filename in os.listdir(path):
        list.append(filename.split('.')[0] + '.jpg')
    random_sel_list = []
    if random:
        np.random.seed(seed)
        random_sel_list = np.random.choice(list,length,replace=False)
    else:
        for index in range(len(list)):
            while len(random_sel_list) < length:
                random_sel_list.append(list[index])
    return random_sel_list

def write_vnn_spec(img_pre, list, dir_path, prefix="spec", n_class=1,
                   mean=0.0, std=1.0, csv='',network_path='',vnnlib_path=''):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    for network in os.listdir(network_path):
        network = network.split('.')[0]
        selected_list = list[network]
        for index in range(len(selected_list)):
            imagename = selected_list[index]
            pred_file = os.path.join(img_pre[network].replace('images', 'pred_labels'), imagename.replace('jpg', 'txt'))
            spec_name = f"{prefix}_onnx_{network}_idx_{imagename.split('.')[0]}.vnnlib"
            spec_path = os.path.join(dir_path, spec_name)
            x = Image.open(os.path.join(img_pre[network], imagename))
            x = np.array(x)[:,:,(2,1,0)]
            x = x.transpose(2, 0, 1)
            x = np.array(x) / 255
            x = ((x - mean) / std).reshape(-1)

            input_len = 12296

            with open(pred_file, 'r') as f:
                for line in f.readlines():
                    l = [float(anno) for anno in line.strip().split(" ")]


            with open(spec_path, "w") as f:
                f.write(f"; Spec for sample id {imagename}\n")

                f.write(f"\n; Definition of input variables(image, position and pred res)\n")
                for i in range(input_len):
                    f.write(f"(declare-const X_{i} Real)\n")

                f.write(f"\n; Definition of output variables\n")
                for i in range(n_class):
                    f.write(f"(declare-const Y_{i} Real)\n")

                f.write(f"\n; Definition of input constraints(image, position and pred res)\n")
                for i in range(input_len):
                    if i < 64*64*3:
                        f.write(f"(assert (>= X_{i} {x[i]:.8f}))\n")
                        f.write(f"(assert (<= X_{i} {x[i]:.8f}))\n")
                    elif i == 12288 or i == 12289:
                        f.write(f"(assert (>= X_{i} {0:.8f}))\n")
                        f.write(f"(assert (<= X_{i} {62:.8f}))\n")
                    elif i > 12289:
                        f.write(f"(assert (>= X_{i} {l[i-12290]:.8f}))\n")
                        f.write(f"(assert (<= X_{i} {l[i-12290]:.8f}))\n")

                f.write(f"\n; Definition of output constraints\n")
                for i in range(n_class):
                    f.write(f"(assert (<= Y_{i} 0.5))\n")
    #make csv file
    if not os.path.exists(csv):
        os.system(r"touch {}".format(csv))
    csvFile = open(csv, "w")
    timeout = 350
    for vnnLibFile in os.listdir(vnnlib_path):
        net1 = "patch-1"
        net2 = "patch-3"
        if "patch-1" in vnnLibFile:
            net1 = net1 + '.onnx'
            print(f"{net1},{vnnLibFile},{timeout}", file=csvFile)
        else:
            net2 = net2 + '.onnx'
            print(f"{net2},{vnnLibFile},{timeout}", file=csvFile)
    csvFile.close()


def main():
    seed = int(sys.argv[1])
    # seed = 1
    mean = 0.0
    std = 1.0
    csv = "./instances.csv"

    img_file_pre = {'patch-1': './dataset/patch-1/images',
                    'patch-3': './dataset/patch-3/images'}
    patch_1_list = get_random_images(img_file_pre['patch-1'],random=True,length=16,seed=seed)
    patch_3_list = get_random_images(img_file_pre['patch-3'],random=True,length=16,seed=seed)
    list = {'patch-1':patch_1_list,'patch-3':patch_3_list}
    network_path = './onnx/'
    vnnlib_path = './vnnlib/'

    mean = np.array(mean).reshape((1, -1, 1, 1)).astype(np.float32)
    std = np.array(std).reshape((1, -1, 1, 1)).astype(np.float32)
    write_vnn_spec(img_file_pre, list, dir_path='./vnnlib', prefix='spec', n_class=1, mean=mean, std=std,csv=csv,
                       network_path=network_path, vnnlib_path=vnnlib_path)


if __name__ == "__main__":
    main()