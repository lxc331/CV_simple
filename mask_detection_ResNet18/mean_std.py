import os
from PIL import Image
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Union


class ImageStatsCalculator:
    """
    图像数据集统计信息计算器
    用于计算图像数据集的均值和标准差，适用于深度学习预处理
    """

    def __init__(self, supported_formats: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')):
        """
        初始化计算器

        Args:
            supported_formats: 支持的图像格式
        """
        self.supported_formats = supported_formats
        self.logger = self._setup_logger()

    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def calculate_stats(self, folder_path: Union[str, Path], method: str = 'single_pass') -> Tuple[
        np.ndarray, np.ndarray]:
        """
        计算图像数据集的均值和标准差

        Args:
            folder_path: 图像数据集所在的文件夹路径
            method: 计算方法 ('single_pass' 或 'memory_intensive')

        Returns:
            tuple: (mean, std) 均值和标准差，均为3通道(RGB)的数组

        Raises:
            ValueError: 当找不到有效图像文件时
            FileNotFoundError: 当路径不存在时
        """
        folder_path = Path(folder_path)

        if not folder_path.exists():
            raise FileNotFoundError(f"路径不存在: {folder_path}")

        # 打印当前工作目录和目标路径信息
        current_dir = Path.cwd()
        self.logger.info(f"当前工作目录: {current_dir}")
        self.logger.info(f"目标数据目录: {folder_path.absolute()}")

        # 检查目标目录是否存在
        if not folder_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {folder_path.absolute()}")

        if method == 'single_pass':
            return self._calculate_single_pass(folder_path)
        elif method == 'memory_intensive':
            return self._calculate_memory_intensive(folder_path)
        else:
            raise ValueError(f"不支持的方法: {method}. 请选择 'single_pass' 或 'memory_intensive'")

    def _calculate_memory_intensive(self, folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用内存密集型方法计算均值和标准差（适用于小数据集）

        Args:
            folder_path: 图像数据集所在的文件夹路径

        Returns:
            tuple: (mean, std) 均值和标准差
        """
        self.logger.info("开始计算图像数据集的均值和标准差 (内存密集型方法)...")

        pixel_values = []
        total_images = 0
        processed_images = 0

        # 遍历文件夹中的图片文件
        image_paths = self._get_image_paths(folder_path)
        self.logger.info(f"找到 {len(image_paths)} 个图像文件")

        for image_path in image_paths:
            try:
                image = Image.open(image_path)

                # 转换为RGB格式
                image = self._convert_to_rgb(image)

                image_array = np.array(image)

                # 归一化像素值到0-1之间
                normalized_image_array = image_array.astype(np.float32) / 255.0

                # 将图像数据重塑为(height*width, 3)的形状
                reshaped_image = normalized_image_array.reshape(-1, 3)

                pixel_values.append(reshaped_image)
                total_images += 1
                processed_images += 1

                # 每处理100张图片输出一次进度
                if total_images % 100 == 0:
                    self.logger.info(f"已处理 {total_images} 张图片...")

            except Exception as e:
                self.logger.warning(f"处理图片 {image_path} 时出错: {e}")
                continue

        if processed_images == 0:
            raise ValueError("未找到任何有效的图像文件")

        # 合并所有图像的像素值
        all_pixels = np.concatenate(pixel_values, axis=0)

        # 计算每个通道的均值和标准差
        mean = np.mean(all_pixels, axis=0)
        std = np.std(all_pixels, axis=0)

        self.logger.info(f"总共处理了 {processed_images} 张图片")
        self.logger.info(f"数据集包含 {all_pixels.shape[0]} 个像素点")

        return mean, std

    def _calculate_single_pass(self, folder_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用单次遍历方法计算均值和标准差（内存友好版，适用于大数据集）

        Args:
            folder_path: 图像数据集所在的文件夹路径

        Returns:
            tuple: (mean, std) 均值和标准差
        """
        self.logger.info("开始计算图像数据集的均值和标准差（单次遍历方法）...")

        # 初始化累积变量
        total_pixels = 0
        sum_channels = np.zeros(3, dtype=np.float64)  # 累积每个通道的像素值总和
        sum_squared_channels = np.zeros(3, dtype=np.float64)  # 累积每个通道的像素值平方和

        total_images = 0

        # 获取所有图像路径
        image_paths = self._get_image_paths(folder_path)
        self.logger.info(f"找到 {len(image_paths)} 个图像文件")

        # 单次遍历计算均值和方差
        for image_path in image_paths:
            try:
                image = Image.open(image_path)

                # 躍换为RGB格式
                image = self._convert_to_rgb(image)

                image_array = np.array(image, dtype=np.float32)

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

                # 每处理100张图片输出一次进度
                if total_images % 100 == 0:
                    self.logger.info(f"已处理 {total_images} 张图片...")

            except Exception as e:
                self.logger.warning(f"处理图片 {image_path} 时出错: {e}")
                continue

        if total_pixels == 0:
            raise ValueError("未找到任何有效的图像文件")

        # 计算均值
        mean = sum_channels / total_pixels

        # 计算方差 Var(X) = E[X^2] - (E[X])^2
        var = (sum_squared_channels / total_pixels) - (mean ** 2)

        # 计算标准差
        std = np.sqrt(var)

        self.logger.info(f"总共处理了 {total_images} 张图片")
        self.logger.info(f"数据集包含 {total_pixels} 个像素点")

        return mean, std

    def _get_image_paths(self, folder_path: Path) -> list:
        """
        获取文件夹中所有图像文件的路径

        Args:
            folder_path: 文件夹路径

        Returns:
            list: 图像文件路径列表
        """
        image_paths = []

        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.lower().endswith(self.supported_formats):
                    image_paths.append(Path(root) / filename)

        return image_paths

    def _convert_to_rgb(self, image: Image.Image) -> Image.Image:
        """
        将图像转换为RGB格式

        Args:
            image: PIL图像对象

        Returns:
            Image.Image: RGB格式的图像
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image

    def print_pytorch_usage(self, mean: np.ndarray, std: np.ndarray):
        """
        打印PyTorch使用方式

        Args:
            mean: 计算出的均值
            std: 计算出的标准差
        """
        print("\n" + "=" * 50)
        print("计算结果:")
        print("=" * 50)
        print(f"均值 (Mean): [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"标准差 (Std): [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
        print("=" * 50)
        print("\n在PyTorch数据预处理中可以这样使用:")
        print(f"transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")


def main():
    """主函数"""
    # 定义数据集路径 - 修改为绝对路径或确认当前目录下的路径
    folder_path = './data'

    # 检查是否存在 ./data 目录
    data_path = Path(folder_path)
    if not data_path.exists():
        print(f"警告: {folder_path} 目录不存在")
        # 尝试其他可能的路径
        possible_paths = [
            '../data',  # 上级目录
            '../../data',  # 上上级目录
            Path(__file__).parent / 'data'  # 脚本所在目录的data子目录
        ]

        for path in possible_paths:
            if Path(path).exists():
                folder_path = path
                print(f"使用路径: {folder_path}")
                break
    else:
        print(f"使用路径: {folder_path}")

    print("=" * 60)
    print("图像数据集统计信息计算工具")
    print("=" * 60)

    calculator = ImageStatsCalculator()

    try:
        # 使用单次遍历方法计算（更节省内存）
        mean, std = calculator.calculate_stats(folder_path, method='single_pass')

        # 打印结果
        calculator.print_pytorch_usage(mean, std)

    except Exception as e:
        print(f"计算过程中出现错误: {e}")


if __name__ == "__main__":
    main()
