from cv2 import waitKey
import torch
import cv2
import numpy as np
import time


# Model
model_path = r"anpr_best.pt"  #custom model path
video_path = r"TEST.mp4\pexels-hudson-coelho-5579754.mp4"  #input video path
cpu_or_cuda = "cpu"  #choose device; "cpu" or "cuda"(if cuda is available)
device = torch.device(cpu_or_cuda)
model =  torch.hub.load('yolov5','custom', source ='local', path='anpr_best.pt',force_reload=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, force_reload=True)
model = model.to(device)
frame = cv2.VideoCapture(video_path)

frame_width = int(frame.get(3))
frame_height = int(frame.get(4))
size = (frame_width, frame_height)
writer = cv2.VideoWriter('',-1,8,size)

text_font = cv2.FONT_HERSHEY_PLAIN
color= (255,247,2)
text_font_scale = 1.25
prev_frame_time = 0
new_frame_time = 0

# Inference Loop
while True:
    ret, image = frame.read()
    if ret:
        x_shape, y_shape = image.shape[1], image.shape[0]

        output = model(image)
        result = np.array(output.pandas().xyxy[0])
        for i in result:
            if i[4]>0.6:
                p1 = (int(i[0]),int(i[1]))
                p2 = (int(i[2]),int(i[3]))
                text_origin = (int(i[0]),int(i[1])-5)
            #print(p1,p2)
                
                cv2.rectangle(image,p1,p2,color=color,thickness=2)  #drawing bounding boxes
                cv2.putText(image,text=f"{i[-1]} {i[-3]:.2f}",org=text_origin,
                            fontFace=text_font,fontScale=text_font_scale,
                            color=color,thickness=2)  #class and confidence text
            #x1, y1, x2, y2 = int(i[0]*x_shape), int(i[1]*y_shape), int(i[2]*x_shape), int(i[3]*y_shape) 
              #  cv2.imshow("plate",image[int(i[0]):int(i[2]),int(i[1]):int(i[3])])
                #cv2.imshow()

        new_frame_time = time.time()

        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(image, fps, (7, 70), text_font, 3, (100, 255, 0), 3, cv2.LINE_AA)
        writer.write(image)
        print(i[4])
        
        cv2.imshow("image",image)

    else:
        break

    if waitKey(1) & 0xFF == ord('q'):
        break

frame.release()
cv2.destroyAllWindows()