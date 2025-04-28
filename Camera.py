import cv2
import time
from Focuser import Focuser

class Camera:
    def __init__(self, i2c_bus=10, sensor_id = 0,
                 capture_width=960, capture_height=540, 
                 display_width=960,display_height=540,
                 flip_method=0, framerate=60):
        self.sensor_id = sensor_id
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.flip_method = flip_method
        self.framerate = framerate
        self.best_focus = 10
        self.finish_focus = False

        self.cap = cv2.VideoCapture(self.gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Failed to open camera.")

        self.focuser = Focuser(i2c_bus)
        self.focusing(self.best_focus)

    def gstreamer_pipeline(self):
        return (
            "nvarguscamerasrc sensor-id=%d ! "
            "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                self.sensor_id,
                self.capture_width,
                self.capture_height,
                self.framerate,
                self.flip_method,
                self.display_width,
                self.display_height,
            )
        )

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("⚠️ Failed to read frame.")
        return frame

    def focusing(self, val):
        self.focuser.set(Focuser.OPT_FOCUS, val)

    def laplacian(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_16U)
        return cv2.mean(lap)[0]

    def autofocus(self):
        max_value = 0
        best_focus = 10
        last_value = 0
        dec_count = 0
        skip_frame = 6
        focal_distance = 10

        print("🔧 Autofocus: sweeping...")
        cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            frame = self.read()
            cv2.imshow("CSI Camera", frame)

            if skip_frame == 0:
                skip_frame = 6
                if dec_count < 6 and focal_distance < 1000:
                    self.focusing(focal_distance)
                    val = self.laplacian(frame)
                    if val > max_value:
                        max_value = val
                        best_focus = focal_distance
                    if val < last_value:
                        dec_count += 1
                    else:
                        dec_count = 0
                    if dec_count < 6:
                        last_value = val
                        focal_distance += 10
                else:
                    break
            else:
                skip_frame -= 1

            cv2.waitKey(1)

        self.focusing(best_focus)
        self.finish_focus = True
        self.best_focus = best_focus
        print(f"✅ Autofocus complete. Best focus: {best_focus}")

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        cam = Camera(i2c_bus=10)
        cam.autofocus()

        cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            frame = cam.read()
            cv2.imshow("CSI Camera", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to quit
                break
        cam.release()
    except Exception as e:
        print(f"❌ Node crashed: {e}")