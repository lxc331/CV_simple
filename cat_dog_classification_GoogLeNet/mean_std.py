import os
from PIL import Image
import numpy as np


def calculate_mean_std(folder_path):
    """
    计算图像数据集的均值和标准差

    Args:
        folder_path (str): 图像数据集所在的文件夹路径

    Returns:
        tuple: (mean, std) 均值和标准差，均为3通道(RGB)的数组
    """
    print("开始计算图像数据集的均值和标准差...")

    # 存储所有图像的像素值
    pixel_values = []

    # 遍历文件夹中的图片文件
    total_images = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, filename)

                try:
                    image = Image.open(image_path)

                    # 如果图像是灰度图，转换为RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    image_array = np.array(image)

                    # 归一化像素值到0-1之间
                    normalized_image_array = image_array / 255.0

                    # 将图像数据重塑为(height*width, 3)的形状
                    reshaped_image = normalized_image_array.reshape(-1, 3)

                    pixel_values.append(reshaped_image)
                    total_images += 1

                    if total_images % 100 == 0:
                        print(f"已处理 {total_images} 张图片...")

                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {e}")
                    continue

    if len(pixel_values) == 0:
        raise ValueError("未找到任何有效的图像文件")

    # 合并所有图像的像素值
    all_pixels = np.concatenate(pixel_values, axis=0)

    # 计算每个通道的均值和标准差
    mean = np.mean(all_pixels, axis=0)
    std = np.std(all_pixels, axis=0)

    print(f"总共处理了 {total_images} 张图片")
    print(f"数据集包含 {all_pixels.shape[0]} 个像素点")

    return mean, std


def calculate_mean_std_single_pass(folder_path):
    """
    使用单次遍历计算图像数据集的均值和标准差（内存友好版）

    Args:
        folder_path (str): 图像数据集所在的文件夹路径

    Returns:
        tuple: (mean, std) 均值和标准差，均为3通道(RGB)的数组
    """
    print("开始计算图像数据集的均值和标准差（单次遍历）...")

    # 初始化累积变量
    total_pixels = 0
    sum_channels = np.zeros(3)  # 累积每个通道的像素值总和
    sum_squared_channels = np.zeros(3)  # 累积每个通道的像素值平方和

    total_images = 0

    # 单次遍历计算均值和方差
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
                image_path = os.path.join(root, filename)

                try:
                    image = Image.open(image_path)

                    # 如果图像是灰度图，转换为RGB
                    if image.mode != 'RGB':
                        image = image.convert('RGB')

                    image_array = np.array(image)

                    # 归一化像素值到0-1之间
                    normalized_image_array = image_array / 255.0

                    # 计算像素数量（高度 * 宽度）
                    h, w, c = normalized_image_array.shape
                    pixels_in_image = h * w

                    # 累积每个通道的值和平方值
                    sum_channels += np.sum(normalized_image_array, axis=(0, 1))
                    sum_squared_channels += np.sum(normalized_image_array ** 2, axis=(0, 1))

                    total_pixels += pixels_in_image
                    total_images += 1

                    if total_images % 100 == 0:
                        print(f"已处理 {total_images} 张图片...")

                except Exception as e:
                    print(f"处理图片 {image_path} 时出错: {e}")
                    continue

    if total_pixels == 0:
        raise ValueError("未找到任何有效的图像文件")

    # 计算均值
    mean = sum_channels / total_pixels

    # 计算方差 Var(X) = E[X^2] - (E[X])^2
    var = (sum_squared_channels / total_pixels) - (mean ** 2)

    # 计算标准差
    std = np.sqrt(var)

    print(f"总共处理了 {total_images} 张图片")
    print(f"数据集包含 {total_pixels} 个像素点")

    return mean, std


if __name__ == "__main__":
    # 定义数据集路径
    folder_path = './data'

    print("=" * 50)
    print("图像数据集统计信息计算")
    print("=" * 50)

    try:
        # 使用单次遍历方法计算（更节省内存）
        mean, std = calculate_mean_std_single_pass(folder_path)

        print("\n" + "=" * 50)
        print("计算结果:")
        print("=" * 50)
        print(f"均值 (Mean): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"标准差 (Std): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        print("=" * 50)
        print("\n在PyTorch数据预处理中可以这样使用:")
        print(f"transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")

    except Exception as e:
        print(f"计算过程中出现错误: {e}")