from PIL import Image, ImageFilter, ImageOps, ImageEnhance
import numpy as np
import cv2

def rotate_image(image, angle):
    return image.rotate(-angle, expand=True)

def scale_image(image, scale):
    width, height = image.size
    return image.resize((int(width * scale), int(height * scale)))

def crop_center(image):
    width, height = image.size
    new_width, new_height = width // 2, height // 2
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    return image.crop((left, top, right, bottom))

def shift_image(image, x_shift, y_shift):
    np_image = np.array(image)
    shifted = np.roll(np_image, shift=(y_shift, x_shift), axis=(0, 1))
    return Image.fromarray(shifted)

def apply_sepia(image):
    np_image = np.array(image)
    sepia_filter = np.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
    sepia_image = np.dot(np_image[..., :3], sepia_filter.T)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return Image.fromarray(sepia_image)

def adjust_saturation(image, factor):
    converter = ImageEnhance.Color(image)
    return converter.enhance(factor)

def invert_colors(image):
    return ImageOps.invert(image)

def process_image(image_path):
    image = Image.open(image_path)

    # Поворот на 90 градусов по часовой стрелке
    rotated_90_cw = image.rotate(-90, expand=True)
    rotated_90_cw.show(title="Поворот на 90 градусов по часовой стрелке")

    # Поворот на 90 градусов против часовой стрелки
    rotated_90_ccw = image.rotate(90, expand=True)
    rotated_90_ccw.show(title="Поворот на 90 градусов против часовой стрелки")

    # Зеркальное отражение по горизонтали
    flipped_horizontal = ImageOps.mirror(image)
    flipped_horizontal.show(title="Зеркальное отражение по горизонтали")

    # Зеркальное отражение по вертикали
    flipped_vertical = ImageOps.flip(image)
    flipped_vertical.show(title="Зеркальное отражение по вертикали")

    # Масштабирование в 2 раза
    scaled_up = scale_image(image, 2)
    scaled_up.show(title="Масштабирование в 2 раза")

    # Масштабирование в 0.5 раза
    scaled_down = scale_image(image, 0.5)
    scaled_down.show(title="Масштабирование в 0.5 раза")

    # Обрезка центральной части
    cropped_center = crop_center(image)
    cropped_center.show(title="Обрезка центральной части")

    # Поворот на произвольный угол (например, 45 градусов)
    rotated_arbitrary = rotate_image(image, 45)
    rotated_arbitrary.show(title="Поворот на 45 градусов")

    # Сдвиг изображения (например, на 50 пикселей по X и Y)
    shifted = shift_image(image, 50, 50)
    shifted.show(title="Сдвиг изображения")

    # Применение размытия
    blurred = image.filter(ImageFilter.GaussianBlur(5))
    blurred.show(title="Применение размытия")

    # Преобразование в черно-белое
    bw_image = image.convert("1")
    bw_image.show(title="Черно-белое изображение")

    # Преобразование в оттенки серого
    grayscale = image.convert("L")
    grayscale.show(title="Оттенки серого")

    # Преобразование в оттенки красного
    red_channel = image.copy()
    red_channel_np = np.array(red_channel)
    red_channel_np[..., 1:] = 0
    red_image = Image.fromarray(red_channel_np)
    red_image.show(title="Оттенки красного")

    # Преобразование в оттенки зеленого
    green_channel = image.copy()
    green_channel_np = np.array(green_channel)
    green_channel_np[..., ::2] = 0
    green_image = Image.fromarray(green_channel_np)
    green_image.show(title="Оттенки зеленого")

    # Преобразование в оттенки синего
    blue_channel = image.copy()
    blue_channel_np = np.array(blue_channel)
    blue_channel_np[..., :2] = 0
    blue_image = Image.fromarray(blue_channel_np)
    blue_image.show(title="Оттенки синего")

    # Применение эффекта сепии
    sepia = apply_sepia(image)
    sepia.show(title="Эффект сепии")

    # Увеличение насыщенности
    saturated_up = adjust_saturation(image, 1.5)
    saturated_up.show(title="Увеличение насыщенности")

    # Уменьшение насыщенности
    saturated_down = adjust_saturation(image, 0.5)
    saturated_down.show(title="Уменьшение насыщенности")

    # Инверсия цветов
    inverted = invert_colors(image)
    inverted.show(title="Инверсия цветов")

if __name__ == "__main__":
    input_image_path = "3.jpg"
    process_image(input_image_path)
