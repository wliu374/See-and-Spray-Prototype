import torch
from net import *
import numpy as np
import rospy
class ModelManager:
    def __init__(self,device = "cuda",model_name = "mobilenetv4"):
        self.models = {
            "mobilenetv4": MobileNetV4Segmentation(),
        }
        self.ckpt_paths = {
            "mobilenetv4": "/home/wenxin/catkin_ws/src/Onboard-SDK-ROS/dji_sdk/results/models/mobilenetv4_small/model_best_r1-score_patch.pth",
        }
        self.model_name = model_name
        self.device = device
        self.image_mean = [0.5308,0.5797,0.5452]
        self.image_std = [0.2175,0.2113,0.2199]
        self.image_mean = np.array(self.image_mean).reshape((1,1,3))
        self.image_std = np.array(self.image_std).reshape((1,1,3))
        self.model = self.load_model()

    def load_model(self):
        model = self.models[self.model_name]
        ckpt_path = self.ckpt_paths[self.model_name]
        ckpt = torch.load(ckpt_path,map_location=self.device)
        model.load_state_dict(ckpt['state_dict'],strict = False)
        model.to(self.device).eval().half()
        print(f"✅ Model loaded!")
        return model
    
    def normalize(self,image):
        image = image.astype(np.float16)/255
        image = (image - self.image_mean)/self.image_std
        return image
    
    def toTensor(self,image):
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image,axis=0)
        image = torch.from_numpy(image)
        image = image.cuda().half()
        return image
    
    def zeroPaddding(self,tensor,psize=32):
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
    
    def run_dummy(self,image):
        image = self.normalize(image)
        image = self.toTensor(image)
        image = self.zeroPaddding(image)
        dummy_input = torch.randn_like(image).to(self.device)
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_input)
        print("Dummy input inference complete. Model warmed up!")
    
    def run_inference(self,image):
        image = self.normalize(image)
        image = self.toTensor(image)
        image = self.zeroPaddding(image)
        with torch.no_grad():
            output = self.model(image)
            output = output.squeeze().float().cpu().numpy()
            # print("output shape",output.shape)
            # Binarize output

            # output = np.clip(output,0,None)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            rospy.loginfo(f"✅ Inference completed!")
            # print("sum:",np.sum(output))
        return output

if __name__ == "__main__":
    # You can optionally parse args if needed here
    manager = ModelManager(device="cuda", model_name="mobilenetv4")
    manager.run_dummy()


