import tensorrt as trt
import os
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# INT8 calibration reference. Supply your own ONNX model and calibration images.
class YOLOEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, img_dir, batch_size, input_shape=(640, 640)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.img_dir = img_dir
        self.img_list = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.batch_size * 3 * input_shape[0] * input_shape[1] * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.img_list):
            return None
        
        batch_imgs = []
        for i in range(self.batch_size):
            img_path = self.img_list[self.current_index + i]
            # Calibration input is RGB, CHW, normalized to [0, 1].
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.input_shape)
            img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            batch_imgs.append(img)
            
        self.current_index += self.batch_size
        data = np.ascontiguousarray(np.stack(batch_imgs))
        cuda.memcpy_htod(self.device_input, data)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        return None  # Recalibrate on each run.

    def write_calibration_cache(self, cache):
        with open("yolo_calib.cache", "wb") as f:
            f.write(cache)

def build_int8_engine(onnx_file_path, engine_file_path, calib_img_dir):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_file_path, 'rb') as model:
        parser.parse(model.read())
        
    config = builder.create_config()
    # Limit the TensorRT workspace to 1 GiB.
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) 
    
    config.set_flag(trt.BuilderFlag.INT8)
    
    calibrator = YOLOEntropyCalibrator(calib_img_dir, batch_size=4)
    config.int8_calibrator = calibrator
    
    print("Calibrating and building the INT8 engine. This may take a few minutes...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print(f"Saved INT8 engine to {engine_file_path}")

if __name__ == "__main__":
    build_int8_engine("best.onnx", "yolo_int8.engine", "calib_images/")
