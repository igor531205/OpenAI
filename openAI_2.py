import cv2
import numpy


def get_frame(cap, scaling_factor):  # Определение функции, получающей текущий кадр из веб-камеры
    # Чтение текущего кадра из объекта захвата видео
    _, frame = cap.read()

    # Изменение размера изображения
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame


if __name__ == '__main__':
    # Определение объекта захвата видео
    cap = cv2.VideoCapture(0)

    # Определение масштабного множителя для изображений
    scaling_factor = 0.5

    # Чтение кадров из веб-камеры до тех пор, пока
    # пользователь не нажмет клавишу <Esc>
    while True:

        # Захват текущего кадра
        frame = get_frame(cap, scaling_factor)

        # Преобразуем изображение в цветовое пространство HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Определение диапазона цветов кожи в HSV
        lower = numpy.array([0, 70, 60])
        upper = numpy.array([50, 150, 255])

        # Ограничение НSV-изображения для получения
        # только цветов кожи
        mask = cv2.inRange(hsv, lower, upper)

        # Вьmолнение операции побитового И для маски
        # и исходного изображения
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

        # Вьmолнение медианного размытия
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)

        # Отображение входного и выходного кадров
        cv2.imshow('Bxoднoe изображение', frame)
        cv2.imshow('Bыxoднoe изображение', img_median_blurred)

        # Проверка того, не нажал ли пользователь клавишу <Esc>
        key = cv2.waitKey(5)
        if key == 27:
            break

    # Закрытие всех окон
    cv2.destroyAllWindows()
