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

    # Определение объекта вычитания фона
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()

    # Определим количество предьщущих кадров, которые следует
    # использовать для обучения. Этот фактор управляет скоростью
    # обучения алгоритма. Под скоростью обучения подразумевается
    # скорость, с которой ваша модель будет учиться распознавать
    # фон. Чем выше значение параметра 'history', тем ниже
    # скорость обучения.
    history = 100

    # Определение скорости обучения
    learning_rate = 1.0 / history

    # Определение масштабного множителя для изображений
    scaling_factor = 0.5

    # Чтение кадров из веб-камеры до тех пор, пока
    # пользователь не нажмет клавишу <Esc>
    while True:

        # Захват текущего кадра
        frame = get_frame(cap, scaling_factor)

        # Вычисление маски
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # Преобразование изображения из градаций серого
        # в пространство RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Вывод изображений
        cv2.imshow('Bыxoднoe изображение', frame)
        cv2.imshow('Bыxoднoe изображение', mask & frame)

        # Проверка того, не нажал ли пользователь клавишу <Esc>
        key = cv2.waitKey(5)
        if key == 27:
            break

    # Сброс объекта захвата видео
    cap.release()

    # Закрытие всех окон
    cv2.destroyAllWindows()
