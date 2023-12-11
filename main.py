import sys

import cv2
import numpy as np
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


def main():
    # --------------------------- #
    # Create Application and window
    app = QApplication ( sys.argv )
    window = QWidget()
    window.setWindowTitle ( "HW2 GUI" )
    window.resize ( 1000, 1000 )

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
    load_img_btn = QPushButton ( "Load Image" )

    # For Block1
    Block1_btn_1_1 = QPushButton ( "1.1 Draw Contour" )
    Block1_btn_1_2 = QPushButton ( "1.2 Count Coins" )

    # For Block2
    Block2_btn = QPushButton ( "2. Histogram Equalization" )

    # For Block3
    Block3_btn_3_1 = QPushButton ( "3.1 Closing" )
    Block3_btn_3_2 = QPushButton ( "3.2 Opening" )

    # For Block4
    Block4_btn_4_1 = QPushButton ( "4.1 Show Model Structure" )
    Block4_btn_4_2 = QPushButton ( "4.2 Show Accuracy as Loss" )
    Block4_btn_4_3 = QPushButton ( "4.3 Predict" )
    Block4_btn_4_4 = QPushButton ( "4.4 Reset")

    # For Block5
    Block5_load_img = QPushButton ( "Load Image" )
    Block5_btn_5_1 = QPushButton ( "5.1 Show Images" )
    Block5_btn_5_2 = QPushButton ( "5.2 Show Model Structure" )
    Block5_btn_5_3 = QPushButton ( "5.3 Show Comprasion" )
    Block5_btn_5_4 = QPushButton ( "5.4 Inference" )

    # --------------------------- #
    # Add BTNs into each label

    # For Block1
    Block1_layout = QVBoxLayout()
    Block1_layout.addWidget ( QLabel ( "1. Hough Circle Transform" ) )
    Block1_layout.addWidget ( Block1_btn_1_1 )
    Block1_layout.addWidget ( Block1_btn_1_2 )
    Block1_layout.addWidget ( QLabel ( "TODO: There are _ coins in the image. " ) )
    block1.setLayout ( Block1_layout )

    # For Block2
    Block2_layout = QVBoxLayout()
    Block2_layout.addWidget ( QLabel ( "2. Histogram Equalization" ) )
    Block2_layout.addWidget ( Block2_btn )
    block2.setLayout ( Block2_layout )
    
    # For Block3
    Block3_layout = QVBoxLayout()
    Block3_layout.addWidget ( QLabel ( "3. Morphology Operation" ) )
    Block3_layout.addWidget ( Block3_btn_3_1 )
    Block3_layout.addWidget ( Block3_btn_3_2 )
    block3.setLayout ( Block3_layout )

    # For Block4
    Block4_layout = QVBoxLayout()
    Block4_layout.addWidget ( QLabel ( "4. MNIST Classifier Using VGG19" ) )
    Block4_layout.addWidget ( Block4_btn_4_1 )
    Block4_layout.addWidget ( Block4_btn_4_2 )
    Block4_layout.addWidget ( Block4_btn_4_3 )
    Block4_layout.addWidget ( Block4_btn_4_4 )
    block4.setLayout ( Block4_layout )

    # For Block3
    Block5_layout = QVBoxLayout()
    Block5_layout.addWidget ( QLabel ( "5. ResNet50" ) )
    Block5_layout.addWidget ( Block5_load_img )
    Block5_layout.addWidget ( Block5_btn_5_1 )
    Block5_layout.addWidget ( Block5_btn_5_2 )
    Block5_layout.addWidget ( Block5_btn_5_3 )
    Block5_layout.addWidget ( Block5_btn_5_4 )
    block5.setLayout ( Block5_layout )

    # --------------------------- #
    # Add the blocks into the layouts

    load_img.addWidget ( load_img_btn )
    first_layout.addWidget ( block1 )
    first_layout.addWidget ( block2 )
    first_layout.addWidget ( block3 )
    second_layout.addWidget ( block4 )
    second_layout.addWidget ( block5 )

    # --------------------------- #
    # Add the layouts into the main layout
    main_layout.addLayout ( load_img )
    main_layout.addLayout ( first_layout )
    main_layout.addLayout ( second_layout )
    
    # --------------------------- #
    # Set the main layout for the window and show the window
    window.setLayout ( main_layout )
    window.show()

    sys.exit ( app.exec_() )



if __name__ == "__main__":
    main()
