import cv2
import numpy as np
import os
import time
import sounddevice as sd
import queue
from vosk import Model, KaldiRecognizer
import threading

# object detection/yolo detection(yolo is just object dtection)
config_path = os.path.join(os.path.dirname(__file__), 'yolov3-tiny.cfg')
weights_path = os.path.join(os.path.dirname(__file__), 'yolov3-tiny.weights')
names_path = os.path.join(os.path.dirname(__file__), 'coco.names')

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
# use nvidia gpu or another gpu if available
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    cv2.ocl.setUseOpenCL(True)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)

classes = [line.strip() for line in open(names_path)]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# speech setup using vosk(works offline and is accurate)
vosk_model_path = os.path.join(os.path.dirname(__file__), "vosk_model/vosk-model-small-en-us-0.15")
model = Model(vosk_model_path)
recognizer = KaldiRecognizer(model, 16000)
q = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    q.put(bytes(indata))

speech_text = ""
listening = False  # flag to control speech loop

def speech_loop():
    global speech_text, listening
    with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                           channels=1, callback=audio_callback):
        print("listening as long as u hold s")
        while listening:
            if not q.empty():
                data = q.get()
                if recognizer.AcceptWaveform(data):
                    import json
                    speech_text = json.loads(recognizer.Result()).get("text", "")
                else:
                    import json
                    speech_text = json.loads(recognizer.PartialResult()).get("partial", "")
        # final result after releasing s key
        final_res = recognizer.FinalResult()
        import json
        speech_text = json.loads(final_res).get("text", "")
        print("you said:", speech_text)

# video capture for yolo
cap = cv2.VideoCapture(0)
confThreshold, nmsThreshold = 0.5, 0.4
frame_count, start_time = 0, time.time()
output_layers = net.getUnconnectedOutLayersNames()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_height, frame_width = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF

    # start/stop speech
    if key == ord('s') and not listening:
        # start listening in separate thread
        listening = True
        speech_thread = threading.Thread(target=speech_loop)
        speech_thread.start()
        # pause object detection
        continue
    elif key != ord('s') and listening:
        # s released = stop listening
        listening = False
        speech_thread.join()
        continue

    # yolo detection
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (255, 255), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids, confidences, boxes = [], [], []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confThreshold:
                cx, cy = int(detection[0]*frame_width), int(detection[1]*frame_height)
                w, h = int(detection[2]*frame_width), int(detection[3]*frame_height)
                x, y = int(cx - w/2), int(cy - h/2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices.flatten():
        x, y, w, h = boxes[i]
        color = colors[class_ids[i]].astype(int)
        label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color.tolist(), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    # display speech test
    if speech_text:
        cv2.putText(frame, speech_text, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("yolo and speech", frame)

    if key == ord('q'):
        if listening:   
            listening = False
            speech_thread.join()
        break

cap.release()
cv2.destroyAllWindows()
