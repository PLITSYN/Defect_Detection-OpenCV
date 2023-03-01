import os
import cv2
import numpy as np

IMAGE_SIZE = (800, 800)

# -- Пороговое преобразование --
# THRESHOLD_VALUE необходимо увеличить если контуров не будет обнаружено,
# *уменьшить, если число контуров будет некорректно
THRESHOLD_VALUE = 200
MAX_VALUE = 255

# -- Invert Threshold --
INV_THRESHOLD_VALUE = 50
INV_MAX_VALUE = 255

# -- Canny --
THRESHOLD1 = 100
THRESHOLD2 = 70

# -- Параметры контура --
CON_COLOR = (0, 0, 255)
CON_THICKNESS = 1

# -- Свойства изображений --
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
STACK_IMG_SIZE = (300, 300)

################################

while True:

    files = os.listdir('images')
    print("===================================================")
    print("=         Доступные файлы для обработки           =")
    print("===================================================")
    for i in files:
        print('-> {}\t '.format(i), end='')
        if files.index(i) % 3 == 0 and files.index(i) != 0:
            print('\n')
    print("\n===================================================")

    # выбор изображения из директории с расширением (прим.: img1.jpeg)
    file = input("Выберите файл из доступных(q - выход): ").strip()
    # выход из программы
    if file == 'q' or file == 'Q':
        break

    PATH = 'images/' + file
    # Путь к файлу
    imageOri = cv2.imread(PATH)

    try:
        # Преобразование в оттенки серого
        image = cv2.cvtColor(imageOri, cv2.COLOR_BGR2GRAY)
    except:
        print("Неверный ввод! Введите коректное название(прим.: 'imgSample.jpg')")
    else:
        # Изменение размера изображения
        image = cv2.resize(image, IMAGE_SIZE)
        imageOri = cv2.resize(imageOri, IMAGE_SIZE)
        # -- DEBUG --
        # cv2.imshow('Grayscale', image)
        # cv2.imshow('Original', imageOri)
        # cv2.waitKey(0)

        # Пороговое преобразование
        ret, thresh_basic = cv2.threshold(image, THRESHOLD_VALUE, MAX_VALUE, cv2.THRESH_BINARY)
        # -- DEBUG --
        # cv2.imshow("Thresh basic", thresh_basic)

        # Матрица 5 на 5 в качестве ядра
        kernel = np.ones((5, 5), np.uint8)

        # Морф. Операция(размытие)
        img_erosion = cv2.erode(thresh_basic, kernel, iterations=1)
        # -- DEBUG --
        # cv2.imshow("Eroged", img_erosion)
        # cv2.waitKey(0)

        # Находим углы Canny
        edged = cv2.Canny(img_erosion, THRESHOLD1, THRESHOLD2)
        # -- DEBUG --
        # cv2.imshow('Canny', edged)
        # cv2.waitKey(0)

        # Находим Контуры
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # +++++++++++++++++++++++++++
        # -- Итоговые изображения  --
        # +++++++++++++++++++++++++++
        font = cv2.FONT_HERSHEY_SIMPLEX

        imageRz = cv2.resize(image, STACK_IMG_SIZE)
        thresh_basicRz = cv2.resize(thresh_basic, STACK_IMG_SIZE)
        img_erosionRz = cv2.resize(img_erosion, STACK_IMG_SIZE)
        edgedRz = cv2.resize(edged, STACK_IMG_SIZE)

        imageRz = cv2.putText(imageRz, 'GrayScale', (5, 15), font, 0.5, RED, 1, cv2.LINE_AA)
        thresh_basicRz = cv2.putText(thresh_basicRz, 'ThresholdBinary', (5, 15), font, 0.5, RED, 1, cv2.LINE_AA)
        img_erosionRz = cv2.putText(img_erosionRz, 'Morphology-Erosion', (5, 15), font, 0.5, RED, 1, cv2.LINE_AA)
        edgedRz = cv2.putText(edgedRz, 'Canny Edges', (5, 15), font, 0.5, RED, 1, cv2.LINE_AA)

        # Исходное изображение
        cv2.imshow('Original Image', imageOri)
        # Количество контуров
        num_of_con = str(len(contours) - 1)
        print("Количество обнаруженных дефектов = " + num_of_con)
        if len(contours) > 1:
            print('=========================================')
            print('=       Обнаруженные дефекты            =')
            print('=========================================\n\n')

        # Отрисовка контуров поверх исходного
        if int(num_of_con) != 0:
            for i in range(int(num_of_con)):
                highlighted_img = cv2.drawContours(imageOri, contours, i, CON_COLOR, CON_THICKNESS)

            highlighted_img = cv2.putText(highlighted_img, '{} defect(s) detected'.
                                          format(num_of_con), (5, 15),
                                          font, 0.5, GREEN, 1, cv2.LINE_AA)
        else:
            highlighted_img = cv2.putText(imageOri, 'Unable to detect defects!',
                                          (5, 15), font, 0.5, RED, 2, cv2.LINE_AA)


        # Отображение выделенных дефектов
        cv2.imshow('Highlighted Defect', highlighted_img)
        # Сохранение изображений с дефектами
        cv2.imwrite('Output Images/{}_DEFECTS.jpg'.format(file.split('.')[0]), highlighted_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
