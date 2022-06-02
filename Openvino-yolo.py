import sys
import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt
import logging
from openvino.inference_engine import IENetwork, IECore
import matplotlib.pyplot as plt
from time import time
import argparse
from PIL import Image

logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
log         = logging.getLogger()

# FOR YOLOV5 ----
def letterbox(img, size=(640, 640)):
    w, h = size
    prepimg     = img[:, :, ::-1].copy()
    prepimgr    = cv2.resize(prepimg, (w, h))
    meta        = {'original_shape': prepimg.shape,
                'resized_shape': prepimgr.shape}
    prepimg     = Image.fromarray(prepimgr)
    prepimg     = prepimg.resize((w, h), Image.ANTIALIAS)
    img     = np.asarray(prepimgr)
    return img/255    

def parse_yolo_region(input_image, outputs):

    class_ids = []
    confidences = []
    boxes = []
    outputs = [outputs]

    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]

    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)

            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)                
                box = np.array([left, top, width, height])
                boxes.append(box)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    box = None
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        log.info("\nDetected boxes for batch {}:".format(1))
        log.info(" Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ")
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        log.info("{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4}".format(classes[class_ids[i]], confidences[i], left, top, width, height))        
    
        text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
        dim, baseline = text_size[0], text_size[1]
        cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED)
        cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

    boxes = np.array(boxes)
    ibox = list(boxes[indices])
    return input_image, ibox


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=False, help="Path of the yolo model.",default='yolov5/yolov5n_openvino_model/yolov5n.xml')
    parser.add_argument("--labels", required=False, help="Path of the yolo labels.",default='yolo_80classes.txt')
    parser.add_argument("--input", help="Required. Path to an image/video file. (webCam by default)", required=False, type=str, default=int(0))
    parser.add_argument('--device', type=str, default='CPU', help='Specify the target device to infer on; CPU, GPU, FPGA or MYRIAD is acceptable. \
                                                                   Sample will look for a suitable plugin for device specified (CPU by default)')
    parser.add_argument('--output', action='store', type=str, default='output.mp4')
    args                = parser.parse_args()
    

    #VINO config
    ie                  = IECore()
    net                 = ie.read_network(model=args.model)
    input_blob          = next(iter(net.input_info))
    dims                = net.input_info[input_blob].input_data.shape
    device              = "CPU"
    exec_net            = ie.load_network(network=net, num_requests=2, device_name=device)
    classesFile         = args.labels
    success             = True
    fnum                = 0
    n, c, h, w          = dims
    net.batch_size      = n
    is_async_mode       = True
    #vid config
    vcap                = cv2.VideoCapture(args.input)
    frame_width         = int(vcap.get(3))
    frame_height        = int(vcap.get(4))
    fps                 = int(vcap.get(5))
    #output config
    out                 = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    # Constants.
    INPUT_WIDTH         = 640
    INPUT_HEIGHT        = 640
    SCORE_THRESHOLD     = 0.5
    NMS_THRESHOLD       = 0.45
    CONFIDENCE_THRESHOLD= 0.45
    # Text parameters.
    FONT_FACE           = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE          = 0.7
    THICKNESS           = 1
    fontSize            = 0.8
    # Colors
    BLACK               = (0,0,0)
    BLUE                = (255,178,50)
    YELLOW              = (0,255,255)
    RED                 = (0,0,255)



    number_input_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Total no of Frames : ',number_input_frames)
    number_input_frames = 1 if number_input_frames != -1 and number_input_frames < 0 else number_input_frames

    with open(classesFile, 'rt') as f:
        classes         = f.read().rstrip('\n').split('\n')
        
    print(' processing ... please wait...')
    while success:
        success, imcv   = vcap.read()
        fnum += 1
        imcv            = cv2.cvtColor(imcv, cv2.COLOR_BGR2RGB)
        in_frame        = letterbox(imcv.copy(), (w, h))
        in_frame        = in_frame.transpose((2, 0, 1))  
        in_frame        = in_frame.reshape((n, c, h, w))
        start_time      = time()

        if is_async_mode == True:
            request_id = 1
            exec_net.start_async(request_id=request_id, inputs={input_blob: in_frame}) 
        else:
            request_id = 0
            exec_net.infer(inputs={input_blob: in_frame})
        
        det_time = time() - start_time
        if exec_net.requests[request_id].wait(-1) == 0: 
            output  = exec_net.requests[request_id].output_blobs 
            for layer_name, out_blob in output.items():
                imcv1, ibox = parse_yolo_region(imcv.copy(), out_blob.buffer)
            parsing_time = time() - start_time
                      
        if ibox is not None:
            label = "Frame = %d, Inference time = %.2f ms" % (fnum, parsing_time)
            cv2.putText(imcv1, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (0, 0, 255), 2)
        else:
            label = "Frame = %d, NO DETECTION" % fnum
            print('NO DETECTION!', end='')
            cv2.putText(imcv1, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, fontSize, (255, 0, 255), 2)

        imcv1_rz = cv2.resize(imcv1, (frame_width, frame_height))
        final = cv2.cvtColor(imcv1_rz, cv2.COLOR_BGR2RGB)
        out.write(final)
        cv2.imwrite('current_frame.jpg',final)

        # Display
        cv2.imshow('demo',final)
        if cv2.waitKey(1)&0xFF == ord('q'):
            break

    if not success:
        print("Done processing !!!")
        print("Output file is stored as ", args.output)

    vcap.release()
    out.release()
