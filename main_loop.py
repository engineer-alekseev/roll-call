# Импортирование библиотек 
import cv2 # # opencv-python для работы с видеопотоком и изображениями
import sys # sys для взаимодействия с ОС

# Импортирование функций из utils.py
from utils import (check_known_faces, 
                   preprocess_known_faces, 
                   detect_faces, 
                   recognize_faces, 
                   form_report)

# Вызов функции получения входных данных из директории known_faces
known_faces = check_known_faces()

# Завершение основного скрипта, если входные данные отсутствуют
if not known_faces:
    sys.exit()

# Вывод списка имен для загруженных векторных представлений
known_face_names, known_face_encodings = known_faces
for name in known_face_names:
    print(name)
# Инициализация словаря с результатами распознавания
known_faces_dict = dict.fromkeys(known_face_names, False)
# Получение видеопотока
video_capture = cv2.VideoCapture(0)

frame_counter = 0
n_frames = 1
# С помощью переменной n_frames
# можно настроить колличество кадров, отправляемых на распознавание
# n_frames = 1 - каждый кадр, n_frames = 2 - каждый второй и т.д.
# Это позволяет улучшить производительность но снижает качество распознавание
# Для увеличения частоты кадров следует сначала воспользоваться
# Параметром resize в вызове функции detect_faces и если это не помагает,
# то уже тогда пропускать кадры
while video_capture.isOpened():
# Чтение из видеопоттока отдельных кадров    
    status, frame = video_capture.read()
    frame_counter += 1

# Завершение скрипта, если нет видеопотока
    if not status:
        break

    if frame_counter % n_frames == 0:
# Получение координат найденных лиц и их векторных представлений
        face_locations, face_encodings = detect_faces(frame, 0.25)
# Cверка полученных лиц с известными из входных данных
        face_names = recognize_faces(face_encodings, known_faces)
# Дорисовка в видеопоток, транслируемый в интерфейс прямоугольников вокруг найденных лиц
# и добавление подписей для распознанных лиц
        for face_location, face_name in zip(face_locations, face_names):
            
            top, right, bottom, left = face_location
            
            if face_name == 'Unknown':
                cv2.putText(frame, face_name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 6, 121), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (4, 6, 121), 4)

            else:
                known_faces_dict[face_name] = True
                cv2.putText(frame, face_name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 128, 0), 1)
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 128, 0), 2)
            
        x, y = 10, 20

        for name in known_face_names:
            if known_faces_dict[name]:
                cv2.putText(frame, name + ' is present', (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 128, 0), 1)
            else:
                cv2.putText(frame, name + ' is missng', (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (4, 6, 121), 1)
            y += 15
# Отрисовка интерфейса
    cv2.imshow('stream', frame)

# Выход из цикла распознавания если все лица, содержащиеся в входных данных найдены в видеопотоке 
    if not (False in set(known_faces_dict.values())):
            print("All known faces are detected")
# перед завершением работы формируется отчет
            form_report(known_faces_dict)
            break


# выход из цикла распознавания по нажатию клавиши "q", если присутствуют не все лица, полученные из входных данных   
    if cv2.waitKey(1) & 0xFF == ord('q'):
# перед завершением работы формируется отчет
        form_report(known_faces_dict)
        break

video_capture.release()
cv2.destroyAllWindows()
