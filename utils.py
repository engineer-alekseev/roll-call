# Импорты библиотек
import os # os для взаимодействия с операционной системой
import cv2 # opencv-python для работы с видеопотоком и изображениями
import face_recognition # face-recognition для детекции и распознавания лиц
import numpy as np # numpy для манипуляций с векторными представлениями изображений
import pickle as pkl # pickle для сериализации и хранения векторных представлений
import pandas as pd # pandas для работы с табличными данными


def check_known_faces():
    """Проверяет наличие подготовленных векторных представлений лиц,
    в зависимости от их наличия позволяет применить уже готовые или
    закодировать новые векторные представления
    """   
    files = os.listdir()
# Проверка на наличие директории с исходными данными,
# Если она отсутствует, выводится соответствующее сообщение
# Работа программы завершается
    if 'known_faces' not in files:
        print('known faces directory does not exist')
        print('No faces to compare with')

        return False

    os.chdir('known_faces')
    files = os.listdir()

# Выбор существующих векторных представлений или кодирование новых
    if 'face_encodings.pkl' in files:
        answer = input('find existing face_encodings, use it? y/n: ')
# Загрузка ранее сохраненных векторных представлений лиц
        if answer == 'y':
            with open('face_encodings.pkl', 'rb') as file:
               known_faces = pkl.load(file)
               print('face encodings loaded')
               os.chdir('..')
               return known_faces
        
        else:
            print('face_encodings.pkl will be rewritten')

# Кодирование новых векторных представлений из исходных фото
    os.chdir('..')
    known_faces = preprocess_known_faces()
    
    return known_faces


def preprocess_known_faces():
    """Подготавливает и сохраняет векторные представления по фото"""

# Подготовка списка лиц для распознавания по содержимому дериктории known_faces
    os.chdir('known_faces')
    folders = [folder for folder in os.listdir() if os.path.isdir(folder)]
    
    known_face_names = []
    known_face_encodings = []

    for folder in folders:        
        face_name = folder
        known_face_names.append(face_name)
# Кодирование векторных представлений лиц
        os.chdir(folder)
        file = os.listdir()[0]
        image = cv2.imread(file)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_encoding = face_recognition.face_encodings(rgb_image)[0]
        known_face_encodings.append(image_encoding)
        os.chdir('..')    

# Сохранение полученных результатов в дериктории known_faces  
    known_faces = [known_face_names, known_face_encodings]
    with open('face_encodings.pkl', 'wb') as file:
        pkl.dump(known_faces, file)  

    os.chdir('..')
# Передача векторных представлений лиц основному скрипту
    return known_faces


def detect_faces(frame, resize = None):
    """Обнаружение изображений лиц в видеопотоке, принимает:

    frame - векторное представление изображения кадра,
    resize - необязательный параметр уменьшения размера кадра,
    позволяет значительно увеличить производительность,
    применять при проблемах с производительностью (низкая частота кадров),
    рекомендуемые значения 0.25 - 0.5

    возвращает:
    face_locations - координаты обнаруженных лиц в кадре
    face_encodings - векторные представления найденных лиц"""


# Уменьшение размера кадра, если был передан параметр resize
    if resize:
        frame = cv2.resize(frame, (0, 0), fx = resize, fy = resize)
    
# Конвертация кадра из формата BGR (opencv-python) в RGB (face-recognition)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
# Обнаружение кординат лиц в кадре
    face_locations = face_recognition.face_locations(frame)
# Подготовка векторных представлений лиц
    face_encodings = face_recognition.face_encodings(frame, face_locations)

# Пересчет координат лиц в кадре перед возвратом основному скрипту,
# если кадр сжимался перед детекцией лиц
    if resize:
        face_locations = np.array(face_locations)
        face_locations = face_locations / resize

# возврат координат и векторных представлений лиц, обнаруженных в кадре основному скрипту
    return face_locations.astype(int), face_encodings


def recognize_faces(face_encodings, known_faces):
    """Выполняет сверку лиц, обнаруженнных в видеопотоке с лицами из входных данных.

    принимает:
    face_enkodings - векторные представления лиц, найденных в кадре,
    known_faces - векторные представления лиц из входных данных
    
    возвращает метки(имена) распознанных лиц
    """
    
    known_face_names, known_face_encodings = known_faces
    face_names = []
    

# Сверка векторных представлений лиц,
# если лицо не распознано, ему присваивается метка "Unknown"
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'Unknown'

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

# возврат меток(имен) найденных в кадре лиц
    return face_names


def form_report(known_faces_dict):
    """Формирует отчет о результатах переклички,
    сохраняет его в файл report.xlsx"""
    known_faces_dict = {name:('present' if status else 'missing') for (name, status) in known_faces_dict.items()}
    df = pd.DataFrame.from_dict(known_faces_dict, orient = 'index', columns = ['Status'])
    df.to_excel('report.xlsx')
    
