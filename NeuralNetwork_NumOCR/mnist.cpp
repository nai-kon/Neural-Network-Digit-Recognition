
#include "stdafx.h"
#include "mnist.h"


// MNIST�f�[�^�̃t�@�C���|�C���^���擾
int GetMnistDataFP(FILE **fppImg, FILE **fppLabel, LPCTSTR lpszImgPath, LPCTSTR lpszLabelPath)
{
	if(!fppImg || !fppLabel) return -1;

	// MNIST�摜�f�[�^
	_tfopen_s(fppImg ,lpszImgPath, _T("rb"));
	if(*fppImg == NULL) return -1;

	// MNIST���x���f�[�^
	_tfopen_s(fppLabel, lpszLabelPath, _T("rb"));
	if(!*fppLabel) return -1;
	
	return 0;
}

// MNIST�f�[�^�̃t�@�C���|�C���^�����
void ReleaseMnistDataFP(FILE **fppImg, FILE **fppLabel)
{
	if(!fppImg || !fppLabel) return;

	if(*fppImg) fclose(*fppImg);
	*fppImg = NULL;

	if(*fppLabel) fclose(*fppLabel);
	*fppLabel = NULL;
}

// MNIST�f�[�^�̃C���[�W�ƃ��x�����擾
int GetMnistImgLabel(FILE *fpImg, FILE *fpLabel, MNIST_DATA &mnistData, unsigned int uiPos)
{
	if(!fpImg || !fpLabel) return -1;

	// �摜�f�[�^�ǂݍ���
	fseek(fpImg, 16 + uiPos * MNIST_IMG_SIZE, SEEK_SET);	// �f�[�^����16byte�ڂ���
	fread(mnistData.ucImg, sizeof(char),  MNIST_IMG_SIZE, fpImg);

	// ���x���f�[�^�ǂݍ���
	fseek(fpLabel,	8 + uiPos, SEEK_SET);	// �f�[�^����8byte�ڂ���
	char cLabel;
	fread(&cLabel, sizeof(char),  sizeof(char), fpLabel);
	mnistData.sLabel = (short)cLabel;

	return 0;
}
