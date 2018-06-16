
#include "stdafx.h"
#include "mnist.h"


// MNISTデータのファイルポインタを取得
int GetMnistDataFP(FILE **fppImg, FILE **fppLabel, LPCTSTR lpszImgPath, LPCTSTR lpszLabelPath)
{
	if(!fppImg || !fppLabel) return -1;

	// MNIST画像データ
	_tfopen_s(fppImg ,lpszImgPath, _T("rb"));
	if(*fppImg == NULL) return -1;

	// MNISTラベルデータ
	_tfopen_s(fppLabel, lpszLabelPath, _T("rb"));
	if(!*fppLabel) return -1;
	
	return 0;
}

// MNISTデータのファイルポインタを解放
void ReleaseMnistDataFP(FILE **fppImg, FILE **fppLabel)
{
	if(!fppImg || !fppLabel) return;

	if(*fppImg) fclose(*fppImg);
	*fppImg = NULL;

	if(*fppLabel) fclose(*fppLabel);
	*fppLabel = NULL;
}

// MNISTデータのイメージとラベルを取得
int GetMnistImgLabel(FILE *fpImg, FILE *fpLabel, MNIST_DATA &mnistData, unsigned int uiPos)
{
	if(!fpImg || !fpLabel) return -1;

	// 画像データ読み込み
	fseek(fpImg, 16 + uiPos * MNIST_IMG_SIZE, SEEK_SET);	// データ部は16byte目から
	fread(mnistData.ucImg, sizeof(char),  MNIST_IMG_SIZE, fpImg);

	// ラベルデータ読み込み
	fseek(fpLabel,	8 + uiPos, SEEK_SET);	// データ部は8byte目から
	char cLabel;
	fread(&cLabel, sizeof(char),  sizeof(char), fpLabel);
	mnistData.sLabel = (short)cLabel;

	return 0;
}
