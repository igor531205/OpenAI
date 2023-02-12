import cv2


def frame_diff(prev_frame, cur_frame, next_frame):  # Вычисление разности между кадрами
    # Разность между текущим и следующим кадрами
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)
    # Разность между текущим и предьщущим кадрами
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    return cv2.bitwise_and(diff_frames_1, diff_frames_2)


def get_frame(cap, scaling_factor):  # Определение функции, получающей текущий кадр из веб-камеры
    # Чтение текущего кадра из объекта захвата видео
    _, frame = cap.read()

    # Изменение размера изображения
    frame = cv2.resize(frame, None, fx=scaling_factor,
                       fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Преобразование в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray


if __name__ == '__main__':
    # Определение объекта захвата видео
    cap = cv2.VideoCapture(0)

    # Определение масштабного множителя для изображений
    scaling_factor = 0.5

    # Захват текущего кадра
    prev_frame = get_frame(cap, scaling_factor)

    # Захват следующего кадра
    cur_frame = get_frame(cap, scaling_factor)

    # Захват последующего кадра
    next_frame = get_frame(cap, scaling_factor)

    # Чтение кадров из веб-камеры до тех пор, пока
    # пользователь не нажмет клавишу <Esc>
    while True:
        # Отображение разности между кадрами
        cv2.imshow('Пepeмeщeниe объекта', frame_diff(
            prev_frame, cur_frame, next_frame))

        # Обновление переменных
        prev_frame = cur_frame
        cur_frame = next_frame

        # Захват следующего кадра
        next_frame = get_frame(cap, scaling_factor)

        # Проверка того, не нажал ли пользователь клавишу <Esc>
        key = cv2.waitKey(10)
        if key == 27:
            break

    # Закрытие всех окон
    cv2.destroyAllWindows()
