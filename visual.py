import torch
from net import *
import numpy as np
import os
import cv2

def normalize(image,image_mean,image_std):
    image = image.astype(np.float32)/255
    image = (image - image_mean)/image_std
    return image
    
def toTensor(image):
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image,axis=0)
    image = torch.from_numpy(image)
    image = image.cuda().half()
    return image
    
def zeroPaddding(tensor,psize=32):
    h, w = tensor.size()[-2:]
    ph, pw = (psize - h % psize), (psize - w % psize)
    # print(ph,pw)
    (pl, pr) = (pw // 2, pw - pw // 2) if pw != psize else (0, 0)
    (pt, pb) = (ph // 2, ph - ph // 2) if ph != psize else (0, 0)
    if (ph != psize) or (pw != psize):
        tmp_pad = [pl, pr, pt, pb]
        # print(tmp_pad)
        tensor = F.pad(tensor, tmp_pad)
    return tensor

def load_checkpoint(net,ckpt_pth,start_epoch):
    ckpt = torch.load(ckpt_pth)
    net.load_state_dict(ckpt['state_dict'],strict = False)
    if 'epoch' in ckpt:
        start_epoch = ckpt['epoch']
    if 'optimizer' in ckpt:
        optimizer = ckpt['optimizer']
    if 'train_loss' in ckpt:
        net.train_loss = ckpt['train_loss']
    if 'val_loss' in ckpt:
        net.val_loss = ckpt['val_loss']
    
    print("====> load checkpoint '{}' (epoch{})".format(ckpt_pth,start_epoch))
    if 'measure' in ckpt:
        net.measure = ckpt['measure']
    print('accuracy',net.measure['accuracy'][start_epoch-1])
    print('precision',net.measure['precision'][start_epoch-1])
    print('recall',net.measure['recall'][start_epoch-1])
    print('f1-score',net.measure['f1-score'][start_epoch-1])
    print('iou',net.measure['iou'][start_epoch-1])

    return net

def get_output(net,sample):
    net.eval()
    with torch.no_grad():
        image = sample['image'].cuda().half()
        output = net(image)
        output = output.squeeze().cpu().detach().numpy()
        output[output >=0.5] = 1
        output[output <0.5] = 0
        return output
    
if __name__ == '__main__':
    net = MobileNetV4Segmentation()
    net = nn.DataParallel(net)
    net = net.cuda().half()
    snapshot = "./results/models/mobilenetv4_small/"
    save_dir = snapshot + 'visual'
    os.makedirs(save_dir,exist_ok=True)

    image_mean = np.array([0.5308,0.5797,0.5452]).reshape((1,1,3))
    image_std = np.array([0.2175,0.2113,0.2199]).reshape((1,1,3))

    print(">>>>> load best checkpoint <<<<<")
    load_checkpoint(net,snapshot + 'model_best_r1-score_patch.pth',200)

    img_dir = "./inference_imgs/"
    img_name = "20250709_200730"
    img_path = img_dir + img_name + ".png"
    img = cv2.imread(img_path)
    print(img.shape)
    img_norm = normalize(img,image_mean,image_std)
    img_tensor = toTensor(img_norm)
    img_zeroPadding = zeroPaddding(img_tensor)
    output = get_output(net,{'image':img_zeroPadding})
    print(np.sum(output))
    pred_mas_vis = (output*255).astype(np.uint8)

    pred_colored = cv2.applyColorMap(pred_mas_vis,cv2.COLORMAP_JET)
    pred_colored = cv2.resize(pred_colored,(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST)
    overlay_pred = cv2.addWeighted(img,0.6,pred_colored,0.4,0)
    concat_vis = np.concatenate((img,overlay_pred),axis=1)

    out_path = os.path.join(save_dir,img_name +".png")
    cv2.imwrite(out_path,pred_mas_vis)
    # cv2.imwrite(out_path,concat_vis[:,:,::-1])
    print("save visualization")