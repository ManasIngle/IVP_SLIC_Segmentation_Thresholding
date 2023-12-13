import math
from skimage import io, color
import numpy as np
from tqdm import trange
from PIL import Image

class SuperpixelCluster(object):
    cluster_index = 1

    def _init_(self, row, col, l=0, a=0, b=0):
        self.update(row, col, l, a, b)
        self.pixels = []
        self.cluster_number = self.cluster_index
        SuperpixelCluster.cluster_index += 1

    def update(self, row, col, l, a, b):
        self.row = row
        self.col = col
        self.l = l
        self.a = a
        self.b = b

    def _str_(self):
        return "{},{}:{} {} {} ".format(self.row, self.col, self.l, self.a, self.b)

    def _repr_(self):
        return self._str_()

class SuperpixelProcessor(object):
    @staticmethod
    def open_image(file_path):
        """
        Return:
            3D array, row col [LAB]
        """
        rgb_image = io.imread(file_path)
        lab_array = color.rgb2lab(rgb_image)
        return lab_array

    @staticmethod
    def save_lab_image(file_path, lab_array):
        """
        Convert the array to RGB, then save the image
        :param file_path:
        :param lab_array:
        :return:
        """
        rgb_array = color.lab2rgb(lab_array)
        rgb_array = (rgb_array * 255).astype(np.uint8)  # Convert to uint8
        rgb_image = Image.fromarray(rgb_array)
        rgb_image.save(file_path)

    def make_cluster(self, row, col):
        row = int(row)
        col = int(col)
        return SuperpixelCluster(row, col,
                                 self.image_data[row][col][0],
                                 self.image_data[row][col][1],
                                 self.image_data[row][col][2])

    def _init_(self, image_path, num_clusters, compactness):
        self.num_clusters = num_clusters
        self.compactness = compactness

        self.image_data = self.open_image(image_path)
        self.image_height = self.image_data.shape[0]
        self.image_width = self.image_data.shape[1]
        self.num_pixels = self.image_height * self.image_width
        self.step_size = int(math.sqrt(self.num_pixels / self.num_clusters))

        self.clusters = []
        self.pixel_label = {}
        self.distance = np.full((self.image_height, self.image_width), np.inf)

    def initialize_clusters(self):
        row = self.step_size / 2
        col = self.step_size / 2
        while row < self.image_height:
            while col < self.image_width:
                self.clusters.append(self.make_cluster(row, col))
                col += self.step_size
            col = self.step_size / 2
            row += self.step_size

    def get_gradient(self, row, col):
        if col + 1 >= self.image_width:
            col = self.image_width - 2
        if row + 1 >= self.image_height:
            row = self.image_height - 2

        gradient = self.image_data[row + 1][col + 1][0] - self.image_data[row][col][0] + \
                   self.image_data[row + 1][col + 1][1] - self.image_data[row][col][1] + \
                   self.image_data[row + 1][col + 1][2] - self.image_data[row][col][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.row, cluster.col)
            for d_row in range(-1, 2):
                for d_col in range(-1, 2):
                    new_row = cluster.row + d_row
                    new_col = cluster.col + d_col
                    new_gradient = self.get_gradient(new_row, new_col)
                    if new_gradient < cluster_gradient:
                        cluster.update(new_row, new_col, self.image_data[new_row][new_col][0],
                                       self.image_data[new_row][new_col][1], self.image_data[new_row][new_col][2])
                        cluster_gradient = new_gradient

    def assign_pixels(self):
        for cluster in self.clusters:
            for row in range(cluster.row - 2 * self.step_size, cluster.row + 2 * self.step_size):
                if row < 0 or row >= self.image_height:
                    continue
                for col in range(cluster.col - 2 * self.step_size, cluster.col + 2 * self.step_size):
                    if col < 0 or col >= self.image_width:
                        continue
                    L, A, B = self.image_data[row][col]
                    Dc = math.sqrt(
                        math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = math.sqrt(
                        math.pow(row - cluster.row, 2) +
                        math.pow(col - cluster.col, 2))
                    D = math.sqrt(math.pow(Dc / self.compactness, 2) + math.pow(Ds / self.step_size, 2))
                    if D < self.distance[row][col]:
                        if (row, col) not in self.pixel_label:
                            self.pixel_label[(row, col)] = cluster
                            cluster.pixels.append((row, col))
                        else:
                            self.pixel_label[(row, col)].pixels.remove((row, col))
                            self.pixel_label[(row, col)] = cluster
                            cluster.pixels.append((row, col))
                        self.distance[row][col] = D

    def update_cluster_centers(self):
        for cluster in self.clusters:
            sum_row = sum_col = number = 1
            for pixel in cluster.pixels:
                sum_row += pixel[0]
                sum_col += pixel[1]
                number += 1
            new_row = int(sum_row / number)
            new_col = int(sum_col / number)
            cluster.update(new_row, new_col, self.image_data[new_row][new_col][0],
                           self.image_data[new_row][new_col][1], self.image_data[new_row][new_col][2])

    def save_current_image(self, image_name):
        image_array = np.copy(self.image_data)
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                image_array[pixel[0]][pixel[1]][0] = cluster.l
                image_array[pixel[0]][pixel[1]][1] = cluster.a
                image_array[pixel[0]][pixel[1]][2] = cluster.b
            image_array[cluster.row][cluster.col][0] = 0
            image_array[cluster.row][cluster.col][1] = 0
            image_array[cluster.row][cluster.col][2] = 0
        self.save_lab_image(image_name, image_array)

    def iterate_10_times(self):
        self.initialize_clusters()
        self.move_clusters()
        for i in trange(20):
            self.assign_pixels()
            self.update_cluster_centers()
            if i % 5 == 0:
                name = 'Image{compactness}_K{num_clusters}_loop{loop}.png'.format(loop=i, compactness=self.compactness,
                                                                                   num_clusters=self.num_clusters)
                self.save_current_image(name)

if __name__ == '__main__':
    processor_1 = SuperpixelProcessor('supimg.jpg', 200, 40)
    processor_1.iterate_10_times()

    processor_2 = SuperpixelProcessor('supimg.jpg', 300, 40)
    processor_2.iterate_10_times()

    processor_3 = SuperpixelProcessor('supimg.jpg', 500, 40)
    processor_3.iterate_10_times()

    processor_4 = SuperpixelProcessor('supimg.jpg', 1000, 40)
    processor_4.iterate_10_times()

    processor_5 = SuperpixelProcessor('supimg.jpg', 200, 5)
    processor_5.iterate_10_times()

    processor_6 = SuperpixelProcessor('supimg.jpg', 300, 5)
    processor_6.iterate_10_times()


    processor_7 = SuperpixelProcessor('supimg.jpg', 500, 5)
    processor_7.iterate_10_times()

    processor_8 = SuperpixelProcessor('supimg.jpg', 1000, 5)
    processor_8.iterate_10_times()

