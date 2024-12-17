import cv2
import numpy as np

# Функция для детекции объектов YOLO
def yolo_detection(image_path, net, output_layers, confidence_threshold=0.5):
    image = cv2.imread(image_path)
    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    return image, boxes, confidences

# Функция для анализа HSV и отображения масок
def hsv_detection(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Определение диапазонов цветов
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([70, 255, 255])

    # Создание масок
    mask_red = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1), cv2.inRange(hsv_image, lower_red2, upper_red2))
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Морфологическая обработка
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_CLOSE, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel)

    # Поиск контуров
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return image, contours_red, contours_yellow, contours_green

# Функция фильтрации контуров по положению светофора (YOLO-боксы)
def filter_contours_by_yolo_boxes(contours, yolo_boxes, min_area=100):
    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area < min_area:
            continue  # Удаляем мелкие контуры
        for box in yolo_boxes:
            bx, by, bw, bh = box
            if bx <= x <= bx + bw and by <= y <= by + bh:  # Проверка нахождения внутри YOLO-бокса
                filtered_contours.append(contour)
                break
    return filtered_contours

# Функция удаления вложенных прямоугольников
def remove_nested_boxes(contours):
    boxes = [cv2.boundingRect(c) for c in contours]
    final_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        is_nested = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j and x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                is_nested = True
                break
        if not is_nested:
            final_boxes.append((x1, y1, w1, h1))
    return final_boxes

# Функция выбора основного цвета светофора
def select_main_color(filtered_red, filtered_yellow, filtered_green):
    red_area = sum(cv2.contourArea(c) for c in filtered_red)
    yellow_area = sum(cv2.contourArea(c) for c in filtered_yellow)
    green_area = sum(cv2.contourArea(c) for c in filtered_green)

    if red_area > yellow_area and red_area > green_area:
        return "red", filtered_red
    elif yellow_area > red_area and yellow_area > green_area:
        return "yellow", filtered_yellow
    else:
        return "green", filtered_green

# -------------------------------
image_path = "жеее.jpg"

# Загрузка YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# YOLO
image_yolo, boxes, confidences = yolo_detection(image_path, net, output_layers)

# HSV
image_hsv, contours_red, contours_yellow, contours_green = hsv_detection(image_path)

# Фильтрация контуров по YOLO-боксам
filtered_red = filter_contours_by_yolo_boxes(contours_red, boxes)
filtered_yellow = filter_contours_by_yolo_boxes(contours_yellow, boxes)
filtered_green = filter_contours_by_yolo_boxes(contours_green, boxes)

# Удаление вложенных квадратов
main_contours = []
for color_contours in [filtered_red, filtered_yellow, filtered_green]:
    main_contours.extend(remove_nested_boxes(color_contours))

# Выбор основного цвета
main_color, filtered_contours = select_main_color(filtered_red, filtered_yellow, filtered_green)
final_boxes = remove_nested_boxes(filtered_contours)

# Отрисовка только основного цвета
for x, y, w, h in final_boxes:
    color = (0, 0, 255) if main_color == "red" else (0, 255, 255) if main_color == "yellow" else (0, 255, 0)
    cv2.rectangle(image_yolo, (x, y), (x + w, y + h), color, 2)


cv2.imshow("Filtered Traffic Light Detection", image_yolo)
cv2.waitKey(0)
cv2.destroyAllWindows()
