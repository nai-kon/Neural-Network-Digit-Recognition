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
	short  sLabel;                   // ラベル(画像データが示す数値)
} MNIST_DATA;

// MNISTデータのオープン
int GetMnistDataFP(FILE **fppImg, FILE **fppLabel, LPCTSTR lpszImgPath, LPCTSTR lpszLabelPath);

// MNISTの数字データをクローズ
void ReleaseMnistDataFP(FILE **fppImg, FILE **fppLabel);

// MNISTの数字データの画像・ラベルを取得
int GetMnistImgLabel(FILE *fpImg, FILE *fpLabel, MNIST_DATA &mnistData, unsigned int uiPos);