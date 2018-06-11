
#include "stdafx.h"
#include "FFNN.h"
#include <math.h>
//#include "mnist.h"

FF_Neural::FF_Neural()
{
	m_dB1 = 0.5;			// �o�C�A�X
	m_il0Size = 784;	// ���͑w��784����(28x28)
	m_il1Size = 100;	// ���ԑw��100����
	m_il2Size = 10;		// �o�͑w��10�����Œ�
}


FF_Neural::~FF_Neural()
{
}

void printVec(const doubleVec &vec)
{
	int iSize = vec.size();
	for (int i = 0; i < iSize; i++) {
		printf("d[%d]:%f\n", i, vec[i]);
	}
}

// �V�O���C�h�֐�
double FF_Neural::Sigmoid(double dVal)
{
	return 1.0 / (1.0 + exp(-dVal));
}

// �V�O���C�h�֐��̔����l
double FF_Neural::SigmoidDash(double dVal)
{
	double dF = Sigmoid(dVal);
	return dF * (1.0 - dF);
}


// �\�t�g�}�b�N�X�֐�
void FF_Neural::VecSoftMax(doubleVec &vec)
{
	// �x�N�g���̍ő�l���擾(�I�[�o�[�t���[�΍�)
	double dMaxElm = *std::max_element(vec.begin(), vec.end());

	// ����̌v�Z
	double dExpAll = 0;
	int iVecCnt = vec.size();
	for (int iCnt = 0; iCnt < iVecCnt; iCnt++) {
		dExpAll += exp(vec.at(iCnt) - dMaxElm);
	}

	// �x�N�g���̍X�V
	for (int iCnt = 0; iCnt < iVecCnt; iCnt++) {
		vec[iCnt] = exp(vec.at(iCnt) - dMaxElm) / dExpAll;
	}
}


// �d�݂̏����l��ݒ�
void FF_Neural::InitWeight()
{
	// ���͑w-���ԑw�̏d�݂̏����l
	int iW1Size = m_il0Size * m_il1Size;
	m_vecW1.reserve(iW1Size);
	for (int iCnt = 0; iCnt < iW1Size; iCnt++) {
		m_vecW1.push_back(((2.0 * rand()) / RAND_MAX) - 1.0);	 // -1����+1�͈̔�
	}

	// ���ԑw-�o�͑w�̏d�݂̏����l
	int iW2Size = m_il1Size * m_il2Size;
	m_vecW2.reserve(iW2Size);
	for (int iCnt = 0; iCnt < iW2Size; iCnt++) {
		m_vecW2.push_back(((2.0 * rand()) / RAND_MAX) - 1.0);	 // -1����+1�͈̔�
	}
}

// �d�݂̓ǂݍ���
int	FF_Neural::LoadWeight()
{
	FILE *fp = NULL;
	fopen_s(&fp, FFNN_WEIGHT_FILE_PATH, "r");
	if (!fp) {
		printf("LoadWeight Fail");
		return -1;
	}

	// �o�C�A�X�̓ǂݍ���
	char buf[256];
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%lf\n", &m_dB1);

	// ���͑w�̎�����
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il0Size);

	// ���ԑw�̎�����
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il1Size);

	// �w�̎�����
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il2Size);

	// ���͑w-���ԑw�̏d��
	double dVal;
	fgets(buf, sizeof(buf), fp);
	int iW1Size = m_il0Size * m_il1Size;
	m_vecW1.reserve(iW1Size);
	for (int iCnt = 0; iCnt < iW1Size; iCnt++) {
		fscanf_s(fp, "%lf\n", &dVal);
		m_vecW1.push_back(dVal);
	}

	// ���ԑw-�o�͑w�̏d��
	fgets(buf, sizeof(buf), fp);
	int iW2Size = m_il1Size * m_il2Size;
	m_vecW2.reserve(iW2Size);
	for (int iCnt = 0; iCnt < iW2Size; iCnt++) {
		fscanf_s(fp, "%lf\n", &dVal);
		m_vecW2.push_back(dVal);
	}

	fclose(fp);

	return 0;
}

// �d�݂̓ǂݍ���
int	FF_Neural::SaveWeight()
{
	FILE *fp = NULL;
	fopen_s(&fp, FFNN_WEIGHT_FILE_PATH, "w");
	if (!fp) {
		printf("SaveWeight Fail");
		return -1;
	}

	fputs("#Bias\n", fp);
	fprintf(fp, "%f\n", m_dB1);

	fputs("#Neuron Num of l0Layer\n", fp);
	fprintf(fp, "%d\n", m_il0Size);

	fputs("#Neuron Num of l1Layer\n", fp);
	fprintf(fp, "%d\n", m_il1Size);

	fputs("#Neuron Num of l2Layer\n", fp);
	fprintf(fp, "%d\n", m_il2Size);

	fputs("#Weights of l0-l1\n", fp);
	int iSize = m_vecW1.size();
	for (int iCnt = 0; iCnt < iSize; iCnt++) {
		fprintf(fp, "%f\n", m_vecW1.at(iCnt));
	}

	fputs("#Weights of l1-l2\n", fp);
	iSize = m_vecW2.size();
	for (int iCnt = 0; iCnt < iSize; iCnt++) {
		fprintf(fp, "%f\n", m_vecW2.at(iCnt));
	}

	fclose(fp);

	return 0;
}

// �t�����`��
// ExpZ2 ... ���t���x��
double FF_Neural::BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2)
{
	// �������`��
	doubleVec vecZ1, vecZ2;
	ForwardProp(vecZ0, vecZ1, vecZ2);

	// �N���X�G���g���s�[�Ō덷�Z�o

	// �o�͑w�̌��z���Z�o
	double dLoss = 0;
	doubleVec vecDel2;	// �o�͑w�̃f���^E�̎Z�o
	vecDel2.reserve(m_il2Size);
	for (int iCnt = 0; iCnt < m_il2Size; iCnt++) {
		dLoss += ExpZ2[iCnt] * log(vecZ2[iCnt]);
		vecDel2.push_back(vecZ2[iCnt] - ExpZ2[iCnt]);
	}

	// ���ԑw�̌��z���Z�o
	doubleVec vecDel1;
	vecDel1.reserve(m_il1Size);
	for (int iCnt = 0; iCnt < m_il1Size; iCnt++) {

		double dDel1 = 0;
		double dSigDash = SigmoidDash(vecZ1[iCnt]);
		for (int il2 = 0; il2 < m_il2Size; il2++) {
			dDel1 += vecDel2[il2] * m_vecW2[iCnt + iCnt * il2] * dSigDash;
		}

		vecDel1.push_back(dDel1);
	}

	// �덷�t�`��
	double dEps = 0.015; //�w�K�W��

	for (int il2 = 0; il2 < m_il2Size; il2++) {
		// W2�̏d�ݍX�V
		int iW2Offset = il2 * m_il1Size;
		for (int iCnt = 0; iCnt < m_il1Size; iCnt++) {
			m_vecW2[iW2Offset + iCnt] -= dEps * vecDel2[il2] * vecZ1[iCnt];
		}
	}

	for (int il1 = 0; il1 < m_il1Size; il1++) {
		// W1�̏d�ݍX�V
		int iW1Offset = il1 * m_il0Size;
		for (int iCnt = 0; iCnt < m_il0Size; iCnt++) {
			m_vecW1[iW1Offset + iCnt] -= dEps * vecDel1[il1] * vecZ0[iCnt];
		}
	}

	return dLoss;
}

// �������`��
void FF_Neural::ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2)
{
	// ���ԑw�ւ̏������`��
	vecZ1.clear();
	vecZ1.reserve(m_il1Size);
	for (int il1 = 0; il1 < m_il1Size; il1++) {

		double dSum = 0;
		int iW1Offset = il1 * m_il0Size;
		for (int iCnt = 0; iCnt < m_il0Size; iCnt++) {
			dSum += vecZ0[iCnt] * m_vecW1[iW1Offset + iCnt];
		}
		// ������(�V�O���C�h�֐�)
		vecZ1.push_back(Sigmoid(dSum));
	}

	// �o�͑w�ւ̏������`��
	vecZ2.clear();
	vecZ2.reserve(m_il2Size);
	for (int il2 = 0; il2 < m_il2Size; il2++) {

		double dSum = m_dB1;	// �o�C�A�X�l��������
		int iW2Offset = il2 * m_il1Size;
		for (int iCnt = 0; iCnt < m_il1Size; iCnt++) {
			dSum += vecZ1[iCnt] * m_vecW2[iW2Offset + iCnt];
		}

		vecZ2.push_back(dSum);
	}

	// �\�t�g�}�b�N�X�֐��Ŋm���l�̌v�Z
	VecSoftMax(vecZ2);
}

/**
// �w�K
int	FF_Neural::Training(const doubleVec &vecIn, const doubleVec &ExpZ2)
{
FILE *fpImg = NULL;
FILE *fpLabel = NULL;

m_il0Size = MNIST_IMG_SIZE;

InitWeight();

if(GetFP_MnistTraningData(&fpImg, &fpLabel, false)!= 0){
printf("�P���f�[�^�I�[�v�����s\n");
return -1;
}

// ���t�f�[�^�̓ǂݍ���
double dErr = 0;
for(int iCnt = 0; iCnt < MNIST_TRANING_IMG_CNT; iCnt++){

MNIST_DATA sTraningData;
GetMnistImgAndLabel(fpImg, fpLabel, sTraningData, iCnt);

doubleVec vecImg, vecLabel;
ConvMNISTtoVec(sTraningData, vecImg, vecLabel);
CenteringInputImg(vecImg);

dErr = ffnn.BackProp(vecImg, vecLabel);
if(iCnt % 100 == 0) printf("iCnt = %d dMiss:%f\n", iCnt, dErr);
}


SaveWeight();


}
**/
