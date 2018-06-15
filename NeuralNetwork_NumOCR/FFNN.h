#pragma once

#include <vector>
#include <algorithm>

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
	int	Training(const CStatic &m_stResDisp);

	// ���_
	void Inference(const doubleVec& vecIn, const CStatic &m_stResDisp)

	// �P���摜�p�X�̃Z�b�g
	void SetTraingImgPath(LPCTSTR lpszImgPath) : m_pszTrainImgPath(lpszImgPath);
	void SetTestImgPath(LPCTSTR lpszImgPath) : m_pszTestImgPath(lpszImgPath);

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

	doubleVec m_vecW1;		// ���͑w-���ԑw�̏d��
	doubleVec m_vecW2;		// ���ԑw-�o�͑w�̏d��
	double m_dB1;			// �o�C�A�X
	int	m_il0Size;			// ���͑w�̎�����
	int	m_il1Size;			// ���ԑw�̎�����
	int	m_il2Size;			// �o�͑w�̎�����
	LPCTSTR m_pszTrainImgPath; // �g���[�j���OMNIST�摜�p�X
	LPCTSTR m_pszTestImgPath;  // �e�X�g�pMNIST�摜�p�X

};