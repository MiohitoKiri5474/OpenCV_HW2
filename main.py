import os
import sys
from collections import OrderedDict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchsummary
import torchvision
import torchvision.transforms as transforms
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image
from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from training.model import VGG19 as vgg19_bn

file_name = None
image = None
Block5_img = None
mouse_pos = None
app = QApplication(sys.argv)
window = QWidget()
Block1_label = QLabel("There are _ coins in the image. ")
Block4_blank = QLabel("")
Block4_blank.setAlignment(Qt.AlignCenter)
Block4_blank.setFixedSize(500, 300)
Block5_blank = QLabel("")
Block5_blank.setAlignment(Qt.AlignCenter)
Block5_blank.setFixedSize(500, 300)
blank_pixmap = QPixmap(Block4_blank.size())
Block4_predict = QLabel("")
Block5_predict = QLabel("")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg19_model = None
resnet50_model = None


def load_vgg19():
    global vgg19_model
    vgg19_model_path = "./model/model_VGG19_BN.pth"
    vgg19_model = vgg19_bn(in_channels=1, num_classes=10).to(device)
    state_dict = torch.load(vgg19_model_path, map_location=torch.device(device))
    vgg19_model.load_state_dict(state_dict)
    vgg19_model.to(device)
    vgg19_model.eval()


def load_resnet50():
    global resnet50_model
    resnet50_model_path = "./model/model_ResNet50.pth"
    # resnet50_model = resnet50 ( blocks = [3, 4, 6, 3], num_classes = 10 ).to ( device )
    resnet50_model = torchvision.models.resnet50().to(device)
    nr_filters = resnet50_model.fc.in_features
    resnet50_model.fc = nn.Linear(nr_filters, 1)
    state_dict = torch.load(resnet50_model_path, map_location=torch.device(device))
    resnet50_model.load_state_dict(state_dict)
    resnet50_model.to(device)
    resnet50_model.eval()


transform_VGG19_BN = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std=(0.5)),
    ]
)

transform_ResNet50 = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# General
def get_path():
    global file_name
    file_name = QFileDialog.getOpenFileName(None, "open file", ".")[0]


# Load images
def load_img_btn_clicked():
    global image
    get_path()
    image = cv2.imread(file_name)
    if image is None:
        print("[ERROR]: Image cannot load. ")
    else:
        print("Loaded Image ", file_name)


# For Block1
def circle_process():
    if image is None:
        print("[ERROR]: Please load image first. ")

    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,
        param2=10,
        minRadius=20,
        maxRadius=25,
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))

    return circles


def Block1_btn_1_1_clicked():
    print("1.1 Draw Contour clicked")
    if image is None:
        print("[ERROR]: Please load image first. ")
        return

    center = np.zeros_like(image)
    process = image.copy()
    circles = circle_process()

    if circles is not None:
        for i in circles[0, :]:
            cv2.circle(process, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(center, (i[0], i[1]), 2, (255, 255, 255), 3)

    cv2.imshow("img_src", image)
    cv2.imshow("img_process", process)
    cv2.imshow("Circle_center", center)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block1_btn_1_2_clicked():
    print("1.2 Count Coins clicked")
    if image is None:
        print("[ERROR]: Please load image first. ")
        return

    circles = circle_process()

    cnt = 0
    if circles is not None:
        for i in circles[0, :]:
            cnt = cnt + 1
            print(f"({i[0]}, {i[1]}), {i[2]}")

    Block1_label.setText("There are " + str(cnt) + " coins in the image. ")


# For Block2
def Block2_btn_clicked():
    print("2 Histogram clicked")
    if image is None:
        print("[ERROR]: Please Load image first. ")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)

    ori_hist, ori_bins = hist, bins = np.histogram(gray_image.flatten(), 256, [0, 256])
    pdf = hist / hist.sum()
    cdf = np.cumsum(pdf)
    lookup_table = np.round(cdf * 255).astype("uint8")
    equalized_image_manual = cv2.LUT(gray_image, lookup_table)

    plt.figure(figsize=(15, 5))

    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.imshow(gray_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Equalized with OpenCV")
    plt.imshow(equalized_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("Equalized Manually")
    plt.imshow(equalized_image_manual, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.title("Histogram of Equalized (Manually)")
    man_hist, man_bins = np.histogram(equalized_image_manual.flatten(), 256, [0, 256])
    plt.bar(range(256), man_hist, width=1, color="gray")

    plt.subplot(2, 3, 4)
    plt.title("Histogram of Original")
    plt.bar(range(256), ori_hist, width=1, color="gray")

    plt.subplot(2, 3, 5)
    plt.title("Histogram of Equalized (OpenCV)")
    equ_hist, equ_bins = np.histogram(equalized_image.flatten(), 256, [0, 256])
    plt.bar(range(256), equ_hist, width=1, color="gray")

    plt.show()


# For Block3
def erosion(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape
    result = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            roi = image[i : i + k_rows, j : j + k_cols]

            if roi.shape == kernel.shape:
                result[i, j] = np.min(roi)

    return result


def dilation(image, kernel):
    rows, cols = image.shape
    k_rows, k_cols = kernel.shape
    result = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(rows):
        for j in range(cols):
            roi = image[i : i + k_rows, j : j + k_cols]

            if roi.shape == kernel.shape:
                result[i, j] = np.max(roi)

    return result


def Block3_btn_3_1_clicked():
    print("3.1 Closing clicked")
    if image is None:
        print("[ERROR]: Please load image first. ")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    result = erosion(dilation(binary_image, kernel), kernel)

    cv2.imshow("Original Image", gray_image)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Block3_btn_3_2_clicked():
    print("3.2 Opening clicked")
    if image is None:
        print("[ERROR]: Please load image first. ")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    result = dilation(erosion(binary_image, kernel), kernel)

    cv2.imshow("Original Image", gray_image)
    cv2.imshow("Result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# For Block4
def clear_block4_blank():
    global mouse_pos
    blank_pixmap.fill(Qt.black)
    Block4_blank.setPixmap(blank_pixmap)
    mouse_pos = None


def mouse_move(self):
    global mouse_pos
    if self.buttons() == Qt.LeftButton and mouse_pos:
        painter = QPainter(blank_pixmap)
        pen = QPen(QColor(Qt.white))
        pen.setWidth(10)
        painter.setPen(pen)
        painter.drawLine(mouse_pos, self.pos())
        painter.end()

        Block4_blank.setPixmap(blank_pixmap)
        mouse_pos = self.pos()


def mouse_press(self):
    global mouse_pos
    if self.button() == 1:
        mouse_pos = self.pos()


def mouse_release(self):
    global mouse_pos
    if self.button() == 1:
        mouse_pos = None


def Block4_btn_4_1_clicked():
    print("4.1 Load Model and Show Model Structure clicked")
    model = vgg19_bn(in_channels=3, num_classes=10)

    torchsummary.summary(model, (3, 224, 224))


def Block4_btn_4_2_clicked():
    print("4.2 Show Accuracy and Loss Clicked")
    pixmap = QPixmap("./plot/VGG19_BN_plot.png")

    pixmap = pixmap.scaled(Block4_blank.size(), Qt.KeepAspectRatio)
    Block4_blank.setPixmap(pixmap)
    Block4_blank.setAlignment(Qt.AlignCenter)


def Block4_btn_4_3_clicked():
    print("4.3 Predict Clicked")

    # save the image
    blank_pixmap.save("predict.png")

    ori_img = cv2.imread("predict.png")
    gray_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
    image = transform_VGG19_BN(gray_img)
    image = image.unsqueeze(0).to(device)
    output = vgg19_model(image)
    label = np.array(output.detach()).argmax()

    Block4_predict.setText(str(label))
    Block4_predict.setAlignment(Qt.AlignCenter)

    plt.bar(
        range(10),
        torch.nn.functional.softmax(output.detach()[0], dim=0),
        align="center",
    )
    plt.xticks(range(10), [str(i) for i in range(10)])
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title(str(label))
    plt.show()

    # clean the tmp files
    os.remove("predict.png")


def Block4_btn_4_4_clicked():
    print("4.4 Reset clicked")
    clear_block4_blank()
    Block4_predict.setText("")
    Block4_predict.setAlignment(Qt.AlignCenter)


# For Block5
def Block5_load_img_clicked():
    global Block5_img
    get_path()
    Block5_img = cv2.imread(file_name)
    if Block5_img is None:
        print("[ERROR]: Image cannot load. ")
    else:
        print("Loaded Image ", file_name)

    Block5_predict.setText("")
    Block5_predict.setAlignment(Qt.AlignCenter)

    pixmap = QPixmap(file_name)
    pixmap = pixmap.scaled(Block5_blank.size(), Qt.KeepAspectRatio)
    Block5_blank.setPixmap(pixmap)
    Block5_blank.setAlignment(Qt.AlignCenter)


def Block5_btn_5_1_clicked():
    print("5.1 Show Images clicked")

    image1 = cv2.imread("./dataset/inference_dataset/Cat/190315.jpg")
    image2 = cv2.imread("./dataset/inference_dataset/Dog/12051.jpg")

    plt.subplot(1, 2, 1)
    plt.title("Cat")
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Dog")
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.show()


def Block5_btn_5_2_clicked():
    print("5.2 Show Model Structure clicked")

    torchsummary.summary(resnet50_model, (3, 224, 224))


def Block5_btn_5_3_clicked():
    print("5.3 Show Comprasion clicked")

    plot = cv2.imread("./plot/ResNet50_comparison.png")

    plt.title("Comprasion")
    plt.imshow(cv2.cvtColor(plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def Block5_btn_5_4_clicked():
    print("5.4 Inference clicked")

    if Block5_img is None:
        print("[ERROR]: Please load image first. ")
        return

    image = transform_ResNet50(Block5_img)
    image = image.unsqueeze(dim=0).to(device)
    output = "Cat" if torch.sigmoid(resnet50_model(image)) < 0.5 else "Dog"

    Block5_predict.setText(output)
    Block5_predict.setAlignment(Qt.AlignCenter)


def main():
    # --------------------------- #
    # Create Application and window
    window.setWindowTitle("HW2 GUI")
    window.resize(1000, 1000)

    # --------------------------- #
    # Create a main layout to contain the five blocks
    main_layout = QHBoxLayout()

    # --------------------------- #
    # Create three vertical layouts for organizing the blocks

    load_img = QVBoxLayout()
    first_layout = QVBoxLayout()
    second_layout = QVBoxLayout()

    # --------------------------- #
    # Create five blocks as QLabel widgets

    block1 = QWidget()
    block2 = QWidget()
    block3 = QWidget()
    block4 = QWidget()
    block5 = QWidget()

    # --------------------------- #
    # Create BTNs

    # For load images
    load_img_btn = QPushButton("Load Image")

    # For Block1
    Block1_btn_1_1 = QPushButton("1.1 Draw Contour")
    Block1_btn_1_2 = QPushButton("1.2 Count Coins")

    # For Block2
    Block2_btn = QPushButton("2. Histogram Equalization")

    # For Block3
    Block3_btn_3_1 = QPushButton("3.1 Closing")
    Block3_btn_3_2 = QPushButton("3.2 Opening")

    # For Block4
    Block4_btn_4_1 = QPushButton("4.1 Show Model Structure")
    Block4_btn_4_2 = QPushButton("4.2 Show Accuracy as Loss")
    Block4_btn_4_3 = QPushButton("4.3 Predict")
    Block4_btn_4_4 = QPushButton("4.4 Reset")

    # For Block5
    Block5_load_img = QPushButton("Load Image")
    Block5_btn_5_1 = QPushButton("5.1 Show Images")
    Block5_btn_5_2 = QPushButton("5.2 Show Model Structure")
    Block5_btn_5_3 = QPushButton("5.3 Show Comprasion")
    Block5_btn_5_4 = QPushButton("5.4 Inference")

    # --------------------------- #
    # Add BTNs into each label

    # For Block1
    Block1_layout = QVBoxLayout()
    Block1_layout.addWidget(QLabel("1. Hough Circle Transform"))
    Block1_layout.addWidget(Block1_btn_1_1)
    Block1_layout.addWidget(Block1_btn_1_2)
    Block1_layout.addWidget(Block1_label)
    block1.setLayout(Block1_layout)

    # For Block2
    Block2_layout = QVBoxLayout()
    Block2_layout.addWidget(QLabel("2. Histogram Equalization"))
    Block2_layout.addWidget(Block2_btn)
    block2.setLayout(Block2_layout)

    # For Block3
    Block3_layout = QVBoxLayout()
    Block3_layout.addWidget(QLabel("3. Morphology Operation"))
    Block3_layout.addWidget(Block3_btn_3_1)
    Block3_layout.addWidget(Block3_btn_3_2)
    block3.setLayout(Block3_layout)

    # For Block4
    Block4_layout_overall_with_title = QVBoxLayout()
    Block4_layout_overall = QHBoxLayout()

    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget(Block4_btn_4_1)
    Block4_layout.addWidget(Block4_btn_4_2)
    Block4_layout.addWidget(Block4_btn_4_3)
    Block4_layout.addWidget(Block4_btn_4_4)
    Block4_layout.addWidget(Block4_predict)

    Block4_image_layout = QVBoxLayout()
    Block4_image_layout.addWidget(Block4_blank)

    Block4_layout_overall_with_title.addWidget(
        QLabel("4. MNIST Classifier Using VGG19")
    )
    Block4_layout_overall.addLayout(Block4_layout)
    Block4_layout_overall.addLayout(Block4_image_layout)
    Block4_layout_overall_with_title.addLayout(Block4_layout_overall)
    block4.setLayout(Block4_layout_overall_with_title)

    # For Block3
    Block5_layout_overall = QHBoxLayout()
    Block5_layout = QVBoxLayout()
    Block5_image_layout = QVBoxLayout()
    Block5_layout.addWidget(QLabel("5. ResNet50"))
    Block5_layout.addWidget(Block5_load_img)
    Block5_layout.addWidget(Block5_btn_5_1)
    Block5_layout.addWidget(Block5_btn_5_2)
    Block5_layout.addWidget(Block5_btn_5_3)
    Block5_layout.addWidget(Block5_btn_5_4)

    Block5_image_layout.addWidget(Block5_blank)
    Block5_image_layout.addWidget(Block5_predict)

    Block5_layout_overall.addLayout(Block5_layout)
    Block5_layout_overall.addLayout(Block5_image_layout)
    block5.setLayout(Block5_layout_overall)

    # --------------------------- #
    # Connect functions and BNTs

    # Load images
    load_img_btn.clicked.connect(load_img_btn_clicked)

    # For Block1
    Block1_btn_1_1.clicked.connect(Block1_btn_1_1_clicked)
    Block1_btn_1_2.clicked.connect(Block1_btn_1_2_clicked)

    # For Block2
    Block2_btn.clicked.connect(Block2_btn_clicked)

    # For Block3
    Block3_btn_3_1.clicked.connect(Block3_btn_3_1_clicked)
    Block3_btn_3_2.clicked.connect(Block3_btn_3_2_clicked)

    # For Block4
    Block4_btn_4_1.clicked.connect(Block4_btn_4_1_clicked)
    Block4_btn_4_2.clicked.connect(Block4_btn_4_2_clicked)
    Block4_btn_4_3.clicked.connect(Block4_btn_4_3_clicked)
    Block4_btn_4_4.clicked.connect(Block4_btn_4_4_clicked)
    Block4_blank.setMouseTracking(True)
    Block4_blank.mouseMoveEvent = mouse_move
    Block4_blank.mousePressEvent = mouse_press
    Block4_blank.mouseReleaseEvent = mouse_release
    clear_block4_blank()

    # For Block5
    Block5_load_img.clicked.connect(Block5_load_img_clicked)
    Block5_btn_5_1.clicked.connect(Block5_btn_5_1_clicked)
    Block5_btn_5_2.clicked.connect(Block5_btn_5_2_clicked)
    Block5_btn_5_3.clicked.connect(Block5_btn_5_3_clicked)
    Block5_btn_5_4.clicked.connect(Block5_btn_5_4_clicked)

    # --------------------------- #
    # Add the blocks into the layouts

    load_img.addWidget(load_img_btn)
    first_layout.addWidget(block1)
    first_layout.addWidget(block2)
    first_layout.addWidget(block3)
    second_layout.addWidget(block4)
    second_layout.addWidget(block5)

    # --------------------------- #
    # Add the layouts into the main layout
    main_layout.addLayout(load_img)
    main_layout.addLayout(first_layout)
    main_layout.addLayout(second_layout)

    # --------------------------- #
    # Set the main layout for the window and show the window
    window.setLayout(main_layout)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    load_vgg19()
    load_resnet50()
    main()
