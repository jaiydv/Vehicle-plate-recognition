from matplotlib.pyplot import text
import torch
import cv2

from PIL import Image
from pytesseract import pytesseract

#Define path to tessaract.exe
path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Define path to image
path_to_image = 'test\Screenshot (171).png'

            
def detectx (frame, model):
    frame = [frame]
    print(f"[INFO] Detecting. . . ")
    results = model(frame)
    
    labels, cordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    return labels, cordinates
def plot_boxes(results, frame,classes):

   
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections. . . ")
    print(f"[INFO] Looping through all detections. . . ")


    ### looping through the detections
    for i in range(n):
        row = cord[i]
        text=""
        if row[4] >= 0.55: ### threshold value for detection. We are discarding everything below this value
            print(f"[INFO] Extracting BBox coordinates. . . ")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape) ## BBOx coordniates
            text_d = classes[int(labels[i])]
            
            if text_d == 'license_plate':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) ## BBox
                cv2.rectangle(frame, (x1, y1-20), (x2, y1), (0, 255,0), -1) ## for text label background

                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

                cv2.putText(frame, text_d + f" {round(float(row[4]),2)}", (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255), 2)

                #code for ocr
                plate = frame[y1:y2, x1:x2]
                
                pytesseract.tesseract_cmd = path_to_tesseract
                predicted_result = pytesseract.image_to_string(plate, lang ='eng', config ='--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
      
                text = "".join(predicted_result.split()).replace(":", "").replace("-", "")
                print(text)
                print(predicted_result)
                plot_img = frame.copy()
                frame = cv2.putText(plot_img, f"{text}", (x1-20, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 2)
                # frame=plate
    return frame

def main(img_path=None, vid_path=None,vid_out = None):

    print(f"[INFO] Loading model... ")
    ## loading the custom trained model
    model =  torch.hub.load('yolov5','custom', source ='local', path='anpr_best.pt',force_reload=True)
    classes = ["license_plate"] ### class names in string format


    if img_path != None:
        print(f"[INFO] Working with image: {img_path}")
        frame = cv2.imread(img_path)
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        results = detectx(frame, model = model) ### DETECTION HAPPENING HERE    

        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        frame = plot_boxes(results, frame,classes = classes)

        cv2.namedWindow("img_only", cv2.WINDOW_NORMAL) ## creating a free windown to show the result

        while True:
            # frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            cv2.imshow("img_only", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print(f"[INFO] Exiting. . . ")
               # cv2.imwrite("final_output.jpg",frame) ## if you want to save he output result.

                break

    elif vid_path !=None:
        print(f"[INFO] Working with video: {vid_path}")

        ## reading the video
        cap = cv2.VideoCapture(vid_path)


        if vid_out: ### creating the video writer if video output path is given

            # by default VideoCapture returns float instead of int
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*'mp4v') ##(*'XVID')
            out = cv2.VideoWriter(vid_out, codec, fps, (width, height))

        # assert cap.isOpened()
        frame_no = 1

        cv2.namedWindow("vid_out", cv2.WINDOW_NORMAL)
        while True:
            # start_time = time.time()
            ret, frame = cap.read()
            if ret :
                print(f"[INFO] Working with frame {frame_no} ")

                frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results = detectx(frame, model = model)
                frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                frame = plot_boxes(results, frame,classes = classes)
                
                cv2.imshow("vid_out", frame)
                if vid_out:
                    print(f"[INFO] Saving output video. . . ")
                    out.write(frame)

                if cv2.waitKey(20) & 0xFF == ord('q'):
                    break
                frame_no += 1
        
        print(f"[INFO] Clening up. . . ")
        ### releaseing the writer
        out.release()
        
        ## closing all windows
        cv2.destroyAllWindows()



### -------------------  calling the main function-------------------------------


print("hi")
main(vid_path="test\pexels-hudson-coelho-5579754.mp4")