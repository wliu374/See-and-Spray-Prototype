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
            "mobilenetv4": "/home/wenxin/catkin_ws/src/Onboard-SDK-ROS/dji_sdk/results/models/mobilenetv4_small/model_best_r1-score.pth",
        }
        self.model_name = model_name
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        model = self.models[self.model_name]
        ckpt_path = self.ckpt_paths[self.model_name]
        ckpt = torch.load(ckpt_path,map_location=self.device)
        model.load_state_dict(ckpt['state_dict'],strict = False)
        model.to(self.device).eval().half()
        print(f"✅ Model loaded!")
        return model
    
    def toTensor(self,image):
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image,axis=0)
        image = torch.from_numpy(image)
        image = image.cuda().half()
        return image
    
    def run_dummy(self,image):
        image = self.toTensor(image)
        dummy_input = torch.randn_like(image).to(self.device)
        for _ in range(5):
            with torch.no_grad():
                _ = self.model(dummy_input)
        print("Dummy input inference complete. Model warmed up!")
    
    def run_inference(self,image):
        image = self.toTensor(image)
        # print(image.size())
        with torch.no_grad():
            output = self.model(image)
            output = output.squeeze().cpu().numpy()

            # Binarize output
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            rospy.loginfo(f"✅ Inference completed!")
        return output

if __name__ == "__main__":
    # You can optionally parse args if needed here
    manager = ModelManager(device="cuda", model_name="mobilenetv4")
    manager.run_dummy()


