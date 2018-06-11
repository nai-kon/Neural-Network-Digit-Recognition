#pragma once

#include <vector>
#include <algorithm>

// �d�݃f�[�^�̃t�@�C���p�X
#define FFNN_WEIGHT_FILE_PATH "WeightFile"

typedef std::vector<double> doubleVec;

void printVec(const doubleVec &vec); // �f�o�b�O�p

									 // ���`�d�^�j���[�����l�b�g���[�N
									 // ���͑w�A���ԑw(1�w)�A�o�͑w��3�w�\��
class FF_Neural
{
public:
	FF_Neural();
	~FF_Neural();

	// �������`��
	void ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2);

	// �t�����`��
	double BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2);

	// �d�݂̃����_��������
	void InitWeight();

	// �d�݃f�[�^�̓ǂݍ���
	int	LoadWeight();

	// �d�݃f�[�^�ǂݍ��ݍς݂� (���������ǂ߂Ă���ς݂Ƃ���)
	BOOL LoadComplete() { return m_vecW1.size() > 0; }

	// �d�݃f�[�^�̏�������
	int	SaveWeight();

private:

	void VecSoftMax(doubleVec &vec);	// �\�t�g�}�b�N�X�֐�
	double Sigmoid(double dVal);		// �V�O���C�h�֐�
	double SigmoidDash(double dVal);	// �V�O���C�h�֐��̔����l

	doubleVec m_vecW1;		// ���͑w-���ԑw�̏d��
	doubleVec m_vecW2;		// ���ԑw-�o�͑w�̏d��
	double m_dB1;			// �o�C�A�X
	int	m_il0Size;			// ���͑w�̎�����
	int	m_il1Size;			// ���ԑw�̎�����
	int	m_il2Size;			// �o�͑w�̎�����

};