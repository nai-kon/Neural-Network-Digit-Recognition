#pragma once

#include <vector>
#include <algorithm>
#include "mnist.h"

// �d�݃f�[�^�̃t�@�C���p�X
#define FFNN_WEIGHT_FILE_PATH "WeightFile"

typedef std::vector<double> doubleVec;

void printVec(const doubleVec &vec); // �f�o�b�O�p


class FF_Neural
{
public:
	FF_Neural();
	~FF_Neural();

	// �w�K
	int	Training(CStatic &m_stResDisp);

	// ���_
	void Inference(const doubleVec& vecIn, CStatic &m_stResDisp);

	// �P���摜�p�X�̃Z�b�g
	void SetTraingFolderPath(LPCTSTR lpszImgPath) { m_pszTrainFolderPath = lpszImgPath; }
	void SetTestFolderPath(LPCTSTR lpszImgPath) { m_pszTestFolderPath = lpszImgPath; }

private:

	// �������`��
	void ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2);

	// �t�����`��
	double BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2);

	void InitWeight(); // �d�݂̃����_��������
	int	LoadWeight();  // �d�݂̓ǂݍ���
	int	SaveWeight();  // �d�݂̕ۑ�
	void VecSoftMax(doubleVec &vec);	// �\�t�g�}�b�N�X�֐�
	double Sigmoid(double dVal);		// �V�O���C�h�֐�
	double SigmoidDash(double dVal);	// �V�O���C�h�֐��̔����l
	BOOL CheckInference(const MNIST_DATA &Mn, doubleVec &vecOut);
	void ConvMNISTtoVec(const MNIST_DATA &Mn, doubleVec &vecImg, doubleVec &vecLabel);

	doubleVec m_vecW1;		// ���͑w-���ԑw�̏d��
	doubleVec m_vecW2;		// ���ԑw-�o�͑w�̏d��
	doubleVec m_vecB1;		// ���͑w-���ԑw�̃o�C�A�X
	doubleVec m_vecB2;		// ���ԑw-�o�͑w�̃o�C�A�X
	int	m_il0Size;			// ���͑w�̎�����
	int	m_il1Size;			// ���ԑw�̎�����
	int	m_il2Size;			// �o�͑w�̎�����
	LPCTSTR m_pszTrainFolderPath; // �g���[�j���OMNIST�t�H���_�p�X
	LPCTSTR m_pszTestFolderPath;  // �e�X�g�pMNIST�t�H���_�p�X

};