#pragma once

#include <vector>
#include <algorithm>

// 重みデータのファイルパス
#define FFNN_WEIGHT_FILE_PATH "WeightFile"

typedef std::vector<double> doubleVec;

void printVec(const doubleVec &vec); // デバッグ用

									 // 順伝播型ニューラルネットワーク
									 // 入力層、中間層(1層)、出力層の3層構造
class FF_Neural
{
public:
	FF_Neural();
	~FF_Neural();

	// 順方向伝搬
	void ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2);

	// 逆方向伝搬
	double BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2);

	// 重みのランダム初期化
	void InitWeight();

	// 重みデータの読み込み
	int	LoadWeight();

	// 重みデータ読み込み済みか (次元数が読めてたら済みとする)
	BOOL LoadComplete() { return m_vecW1.size() > 0; }

	// 重みデータの書き込み
	int	SaveWeight();

private:

	void VecSoftMax(doubleVec &vec);	// ソフトマックス関数
	double Sigmoid(double dVal);		// シグモイド関数
	double SigmoidDash(double dVal);	// シグモイド関数の微分値

	doubleVec m_vecW1;		// 入力層-中間層の重み
	doubleVec m_vecW2;		// 中間層-出力層の重み
	double m_dB1;			// バイアス
	int	m_il0Size;			// 入力層の次元数
	int	m_il1Size;			// 中間層の次元数
	int	m_il2Size;			// 出力層の次元数

};