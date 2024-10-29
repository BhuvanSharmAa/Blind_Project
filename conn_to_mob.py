cap = cv2.VideoCapture('http://192.168.205.181:8080/video')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  

if not cap.isOpened():
    print("Error")
    exit()

frame_skip = 3
frame_count = 0
detected_objects = []


warning_distance_threshold = 1.0

def process_frame(frame):
    global detected_objects

    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:  
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

   
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detected_objects = []
    if len(indexes) > 0: 
        for i in indexes.flatten():  
            detected_object = str(classes[class_ids[i]])
            x, y, w, h = boxes[i]
            distance = estimate_distance(h)  

            
            position = "center"
            if x + w / 2 < width / 3:
                position = "left"
            elif x + w / 2 > (2 * width) / 3:
                position = "right"

            detected_objects.append((detected_object, position))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{detected_object} - {distance}m - {position}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

            
            if detected_object in ["person", "car", "chair", "table"] and distance < warning_distance_threshold:
                speak_async(
                    f"Warning! A {detected_object} is detected on your {position}. Distance is approximately {distance} meters.")



def speech_recognition_thread():
    while True:
        get_speech_input(lambda command: (
            print(f"Command received: {command}"),
            speak_async(find_answer(command, detected_objects)) if command else None
        ))



threading.Thread(target=speech_recognition_thread, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        print("Received an empty frame.")
        continue

    frame_count += 1
    if frame_count % frame_skip == 0:
        process_frame(frame)  
    cv2.imshow('Live Stream', frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q' || 'r' || 's' || 'd'):
        break

cap.release()
cv2.destroyAllWindows()
