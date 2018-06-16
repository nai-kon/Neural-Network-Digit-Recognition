#pragma once

#include <iostream>

#define MNIST_TRAINING_IMG_PATH     "train-images-idx3-ubyte"
#define MNIST_TRAINING_LABEL_PATH   "train-labels-idx1-ubyte"
#define MNIST_TEST_IMG_PATH			"t10k-images-idx3-ubyte"
#define MNIST_TEST_LABEL_PATH		"t10k-labels-idx1-ubyte"

#define MNIST_IMG_WIDTH			(28)
#define MNIST_IMG_HIGHT			(28)
#define MNIST_IMG_SIZE			(MNIST_IMG_WIDTH*MNIST_IMG_HIGHT)
#define MNIST_TRANING_IMG_CNT	(60000)
#define MNIST_TEST_IMG_CNT	    (10000)

typedef struct _mnist_data_{

	unsigned char ucImg[MNIST_IMG_SIZE]; 
	short  sLabel;                   // ���x��(�摜�f�[�^���������l)
} MNIST_DATA;

// MNIST�f�[�^�̃I�[�v��
int GetMnistDataFP(FILE **fppImg, FILE **fppLabel, LPCTSTR lpszImgPath, LPCTSTR lpszLabelPath);

// MNIST�̐����f�[�^���N���[�Y
void ReleaseMnistDataFP(FILE **fppImg, FILE **fppLabel);

// MNIST�̐����f�[�^�̉摜�E���x�����擾
int GetMnistImgLabel(FILE *fpImg, FILE *fpLabel, MNIST_DATA &mnistData, unsigned int uiPos);