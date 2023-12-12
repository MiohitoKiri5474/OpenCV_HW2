import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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

file_name = None
image = None
app = QApplication(sys.argv)
window = QWidget()

Block1_label = QLabel("There are _ coins in the image. ")


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
    print("Draw Contour clicked")
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
    print("Count Coins clicked")
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

    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


# For Block3
def Block3_btn_3_1_clicked():
    print("TODO: 3.1")


def Block3_btn_3_2_clicked():
    print("TODO: 3.2")


# For Block4
def Block4_btn_4_1_clicked():
    print("TODO: 4.1")


def Block4_btn_4_2_clicked():
    print("TODO: 4.2")


def Block4_btn_4_3_clicked():
    print("TODO: 4.3")


def Block4_btn_4_4_clicked():
    print("TODO: 4.4")


# For Block5
def Block5_load_img_clicked():
    global Block5_img
    get_path()
    Block5_image = cv2.imread(file_name)
    if image is None:
        print("[ERROR]: Image cannot load. ")
    else:
        print("Loaded Image ", file_name)


def Block5_btn_5_1_clicked():
    print("TODO: 5.1")


def Block5_btn_5_2_clicked():
    print("TODO: 5.2")


def Block5_btn_5_3_clicked():
    print("TODO: 5.3")


def Block5_btn_5_4_clicked():
    print("TODO: 5.4")


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
    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget(QLabel("4. MNIST Classifier Using VGG19"))
    Block4_layout.addWidget(Block4_btn_4_1)
    Block4_layout.addWidget(Block4_btn_4_2)
    Block4_layout.addWidget(Block4_btn_4_3)
    Block4_layout.addWidget(Block4_btn_4_4)
    block4.setLayout(Block4_layout)

    # For Block3
    Block5_layout = QVBoxLayout()
    Block5_layout.addWidget(QLabel("5. ResNet50"))
    Block5_layout.addWidget(Block5_load_img)
    Block5_layout.addWidget(Block5_btn_5_1)
    Block5_layout.addWidget(Block5_btn_5_2)
    Block5_layout.addWidget(Block5_btn_5_3)
    Block5_layout.addWidget(Block5_btn_5_4)
    block5.setLayout(Block5_layout)

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
    main()
