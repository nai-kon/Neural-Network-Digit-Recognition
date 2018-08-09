#pragma once

#include <vector>
#include <algorithm>
#include "mnist.h"

// 重みデータのファイルパス
#define FFNN_WEIGHT_FILE_PATH "WeightFile"

typedef std::vector<double> doubleVec;

void printVec(const doubleVec &vec); // デバッグ用


class FF_Neural
{
public:
	FF_Neural();
	~FF_Neural();

	// 学習
	int	Training(CStatic &m_stResDisp);

	// 推論
	void Inference(const doubleVec& vecIn, CStatic &m_stResDisp);

	// 訓練画像パスのセット
	void SetTraingFolderPath(LPCTSTR lpszImgPath) { m_pszTrainFolderPath = lpszImgPath; }
	void SetTestFolderPath(LPCTSTR lpszImgPath) { m_pszTestFolderPath = lpszImgPath; }

private:

	// 順方向伝搬
	void ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2);

	// 逆方向伝搬
	double BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2);

	void InitWeight(); // 重みのランダム初期化
	int	LoadWeight();  // 重みの読み込み
	int	SaveWeight();  // 重みの保存
	void VecSoftMax(doubleVec &vec);	// ソフトマックス関数
	double Sigmoid(double dVal);		// シグモイド関数
	double SigmoidDash(double dVal);	// シグモイド関数の微分値
	BOOL CheckInference(const MNIST_DATA &Mn, doubleVec &vecOut);
	void ConvMNISTtoVec(const MNIST_DATA &Mn, doubleVec &vecImg, doubleVec &vecLabel);

	doubleVec m_vecW1;		// 入力層-中間層の重み
	doubleVec m_vecW2;		// 中間層-出力層の重み
	doubleVec m_vecB1;		// 入力層-中間層のバイアス
	doubleVec m_vecB2;		// 中間層-出力層のバイアス
	int	m_il0Size;			// 入力層の次元数
	int	m_il1Size;			// 中間層の次元数
	int	m_il2Size;			// 出力層の次元数
	LPCTSTR m_pszTrainFolderPath; // トレーニングMNISTフォルダパス
	LPCTSTR m_pszTestFolderPath;  // テスト用MNISTフォルダパス

};