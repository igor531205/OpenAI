import cv2
import numpy


# Определение класса, содержащего всю функциональность,
# необходимую для отслеживания объектов
class ObjectTracker(object):
    def __init__(self, scaling_factor=0.5):
        # Инициализация объекта захвата видео
        self.cap = cv2.VideoCapture(0)

        # Чтение текущего кадра из объекта захвата видео
        _, self.frame = self.cap.read()

        # Масштабный множитель для захваченного изображения
        self.scaling_factor = scaling_factor

        # Изменение размера изображения
        self.frame = cv2.resize(self.frame, None,
                                fx=self.scaling_factor,
                                fy=self.scaling_factor,
                                interpolation=cv2.INTER_AREA)

        # Создание окна для отображения кадра
        cv2.namedWindow('Object Tracker')

        # Установка функции обратного вызова, отслеживающей
        # события мыши
        cv2.setMouseCallback('Object Tracker', self.mouse_event)

        # Инициализируем переменные для отслеживания прямоугольной
        # рамки выбора.
        self.selection = None

        # Инициализация переменной, связанной с начальной
        # позицией
        self.drag_start = None

        # Инициализация переменной, связанной с состоянием
        # отслеживания
        self.tracking_state = 0

    # Определение метода для отслеживания событий мыши
    def mouse_event(self, event, x, y, flags, param):
        # Преобразование координат Х и У в 16-битовые
        # целые числа NumPy
        x, y = numpy.int16([x, y])

        # Проверка нажатия кнопки мыши
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0

        # Проверка того, не начал ли пользователь выделять область
        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                # Извлечение размеров кадра
                h, w = self.frame.shape[:2]

                # Получение начальной позиции
                xi, yi = self.drag_start

                # Получение максимальной и минимальной координаты
                x0, y0 = numpy.maximum(0, numpy.minimum([xi, yi], [x, y]))
                x1, y1 = numpy.minimum([w, h], numpy.maximum([xi, yi],
                                                             [x, y]))

                # Сброс переменной selection
                self.selection = None

                # Завершение выделения прямоугольной области
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.selection = (x0, y0, x1, y1)

            else:
                # Если выделение завершено, начать отслеживание
                self.drag_start = None
                if self.selection is not None:
                    self.tracking_state = 1

    # Метод, начинающий отслеживание объекта
    def start_tracking(self):
        # Итерируем до тех пор, пока пользователь не нажмет
        # клавишу <Esc>
        while True:
            # Захват кадра из веб-камеры
            _, self.frame = self.cap.read()

            # Изменение размера входного кадра
            self.frame = cv2.resize(self.frame, None,
                                    fx=self.scaling_factor,
                                    fy=self.scaling_factor,
                                    interpolation=cv2.INTER_AREA)

            # Создание копии кадра
            vis = self.frame.copy()

            # Преобразуем изображение в цветовое пространство HSV
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

            # Создание маски на основании предварительно
            # установленных пороговых значений
            mask = cv2.inRange(hsv, numpy.array((0., 60., 32.)),
                               numpy.array((180., 255., 255.)))

            # Проверка выделения пользователем области
            if self.selection:

                # Извлечение координат выделенного прямоугольника
                x0, y0, xl, yl = self.selection

                # Извлечем окно отслеживания
                self.track_window = (x0, y0, xl - x0, yl - y0)

                # Извлечение интересующей нас области
                hsv_roi = hsv[y0:yl, x0:xl]
                mask_roi = mask[y0:yl, x0:xl]

                # Вычисление гистограммы интересующей нас области
                # НSV-изображения с использованием маски
                hist = cv2.calcHist([hsv_roi], [0], mask_roi,
                                    [16], [0, 180])

                # Нормализация и переформирование гистограммы
                cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
                self.hist = hist.reshape(-1)

                # Извлечение интересующей нас области из кадра
                vis_roi = vis[y0:yl, x0:xl]

                # Вычисление негативного изображения
                cv2.bitwise_not(vis_roi, vis_roi)
                vis[mask == 0] = 0

            # Проверка того, находится ли система
            # в состоянии "отслеживание"
            if self.tracking_state == 1:
                # Сброс переменной selection variaЫe
                self.selection = None

                # Вычисление проекции гистограммы на просвет
                hsv_backproj = cv2.calcBackProject([hsv],
                                                   [0], self.hist,
                                                   [0, 180], 1)

                # Вычисление результата применения операции
                # побитового И к проекции гистограммы на
                # просвет и маске
                hsv_backproj &= mask

                # Определение критерия для прекращения
                # работы трекера
                term_crit = (cv2.TERM_CRITERIA_EPS |
                             cv2.TERM_CRITERIA_COUNT, 10, 1)

                # Применение алгоритма CAМShift к 'hsv_backproj'
                track_box, self.track_window = cv2.CamShift(
                    hsv_backproj, self.track_window, term_crit)

                # Вычерчивание эллипса вокруг объекта
                cv2.ellipse(vis, track_box, (0, 255, 0), 2)

            # Отображение живого видео
            cv2.imshow('Object Tracker', vis)

            # Проверка того, не нажал ли пользователь клавишу <Esc>
            key = cv2.waitKey(5)
            if key == 27:
                break
        # Закрытие всех окон
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # Запуск трекера
    ObjectTracker().start_tracking()
