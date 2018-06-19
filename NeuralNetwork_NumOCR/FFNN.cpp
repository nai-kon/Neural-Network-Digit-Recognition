
#include "stdafx.h"
#include "FFNN.h"
#include <math.h>

FF_Neural::FF_Neural()
{
	m_dB1 = 0.5;				// バイアス
	m_il0Size = MNIST_IMG_SIZE;	// 入力層は784次元(28x28)
	m_il1Size = 100;			// 中間層は100次元
	m_il2Size = 10;				// 出力層は10次元固定
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

// シグモイド関数
double FF_Neural::Sigmoid(double dVal)
{
	return 1.0 / (1.0 + exp(-dVal));
}

// シグモイド関数の微分値
double FF_Neural::SigmoidDash(double dVal)
{
	double dF = Sigmoid(dVal);
	return dF * (1.0 - dF);
}


// ソフトマックス関数
void FF_Neural::VecSoftMax(doubleVec &vec)
{
	// ベクトルの最大値を取得(オーバーフロー対策)
	double dMaxElm = *std::max_element(vec.begin(), vec.end());

	// 分母の計算
	double dExpAll = 0;
	int iVecCnt = vec.size();
	for (int iCnt = 0; iCnt < iVecCnt; iCnt++) {
		dExpAll += exp(vec.at(iCnt) - dMaxElm);
	}

	// ベクトルの更新
	for (int iCnt = 0; iCnt < iVecCnt; iCnt++) {
		vec[iCnt] = exp(vec.at(iCnt) - dMaxElm) / dExpAll;
	}
}

void FF_Neural::ConvMNISTtoVec(const MNIST_DATA &Mn, doubleVec &vecImg, doubleVec &vecLabel)
{
	// 画像データ(0-1の間に正規化する)
	vecImg.clear();
	vecImg.reserve(MNIST_IMG_SIZE);
	for (int iCnt = 0; iCnt < MNIST_IMG_SIZE; iCnt++) {
		vecImg.push_back((double)Mn.ucImg[iCnt] / 255.0);
	}

	// ラベル
	vecLabel.clear();
	vecLabel.reserve(10);
	for (int iCnt = 0; iCnt < 10; iCnt++) {
		vecLabel.push_back(0);
	}
	vecLabel[Mn.sLabel] = 1;
}



// 重みの初期値を設定
void FF_Neural::InitWeight()
{
	// 入力層-中間層の重みの初期値
	int iW1Size = m_il0Size * m_il1Size;
	m_vecW1.clear();
	m_vecW1.reserve(iW1Size);
	for (int iCnt = 0; iCnt < iW1Size; iCnt++) {
		m_vecW1.push_back(((2.0 * rand()) / RAND_MAX) - 1.0);	 // -1から+1の範囲
	}

	// 中間層-出力層の重みの初期値
	int iW2Size = m_il1Size * m_il2Size;
	m_vecW2.clear();
	m_vecW2.reserve(iW2Size);
	for (int iCnt = 0; iCnt < iW2Size; iCnt++) {
		m_vecW2.push_back(((2.0 * rand()) / RAND_MAX) - 1.0);	 // -1から+1の範囲
	}
}

// 重みの読み込み
int	FF_Neural::LoadWeight()
{
	FILE *fp = NULL;
	fopen_s(&fp, FFNN_WEIGHT_FILE_PATH, "r");
	if (!fp) {
		AfxMessageBox(_T("Load:WeightFile Open Failed"));
		return -1;
	}

	// バイアスの読み込み
	char buf[256];
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%lf\n", &m_dB1);

	// 入力層の次元数
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il0Size);

	// 中間層の次元数
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il1Size);

	// 層の次元数
	fgets(buf, sizeof(buf), fp);
	fscanf_s(fp, "%d\n", &m_il2Size);

	// 入力層-中間層の重み
	double dVal;
	fgets(buf, sizeof(buf), fp);
	int iW1Size = m_il0Size * m_il1Size;
	m_vecW1.reserve(iW1Size);
	for (int iCnt = 0; iCnt < iW1Size; iCnt++) {
		fscanf_s(fp, "%lf\n", &dVal);
		m_vecW1.push_back(dVal);
	}

	// 中間層-出力層の重み
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

// 重みの読み込み
int	FF_Neural::SaveWeight()
{
	FILE *fp = NULL;
	fopen_s(&fp, FFNN_WEIGHT_FILE_PATH, "w");
	if (!fp) {
		AfxMessageBox(_T("Save:WeightFile Open Failed"));
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

// 逆方向伝搬
// ExpZ2 ... 教師ラベル
double FF_Neural::BackProp(const doubleVec &vecZ0, const doubleVec &ExpZ2)
{
	// 順方向伝搬
	doubleVec vecZ1, vecZ2;
	ForwardProp(vecZ0, vecZ1, vecZ2);

	// クロスエントロピーで誤差算出

	// 出力層の勾配を算出
	double dLoss = 0;
	doubleVec vecDel2;	// 出力層のデルタEの算出
	vecDel2.reserve(m_il2Size);
	for (int iCnt = 0; iCnt < m_il2Size; iCnt++) {
		dLoss += ExpZ2[iCnt] * log(vecZ2[iCnt]);
		vecDel2.push_back(vecZ2[iCnt] - ExpZ2[iCnt]);
	}

	// 中間層の勾配を算出
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

	// 誤差逆伝搬
	double dEps = 0.015; //学習係数

	for (int il2 = 0; il2 < m_il2Size; il2++) {
		// W2の重み更新
		int iW2Offset = il2 * m_il1Size;
		for (int iCnt = 0; iCnt < m_il1Size; iCnt++) {
			m_vecW2[iW2Offset + iCnt] -= dEps * vecDel2[il2] * vecZ1[iCnt];
		}
	}

	for (int il1 = 0; il1 < m_il1Size; il1++) {
		// W1の重み更新
		int iW1Offset = il1 * m_il0Size;
		for (int iCnt = 0; iCnt < m_il0Size; iCnt++) {
			m_vecW1[iW1Offset + iCnt] -= dEps * vecDel1[il1] * vecZ0[iCnt];
		}
	}

	return dLoss;
}

// 順方向伝搬
void FF_Neural::ForwardProp(const doubleVec &vecZ0, doubleVec &vecZ1, doubleVec &vecZ2)
{
	// 中間層への順方向伝搬
	vecZ1.clear();
	vecZ1.reserve(m_il1Size);
	for (int il1 = 0; il1 < m_il1Size; il1++) {

		double dSum = 0;
		int iW1Offset = il1 * m_il0Size;
		for (int iCnt = 0; iCnt < m_il0Size; iCnt++) {
			dSum += vecZ0[iCnt] * m_vecW1[iW1Offset + iCnt];
		}
		// 活性化(シグモイド関数)
		vecZ1.push_back(Sigmoid(dSum));
	}

	// 出力層への順方向伝搬
	vecZ2.clear();
	vecZ2.reserve(m_il2Size);
	for (int il2 = 0; il2 < m_il2Size; il2++) {

		double dSum = m_dB1;	// バイアス値を加える
		int iW2Offset = il2 * m_il1Size;
		for (int iCnt = 0; iCnt < m_il1Size; iCnt++) {
			dSum += vecZ1[iCnt] * m_vecW2[iW2Offset + iCnt];
		}

		vecZ2.push_back(dSum);
	}

	// ソフトマックス関数で確率値の計算
	VecSoftMax(vecZ2);
}

// 推論結果の正解チェック
BOOL FF_Neural::CheckInference(const MNIST_DATA &Mn, doubleVec &vecOut)
{
	int iAnsNum = -1;
	double dMaxLabel = -1;
	for (int iCnt = 0; iCnt < 10; iCnt++) {
		if (vecOut[iCnt] > dMaxLabel) {
			dMaxLabel = vecOut[iCnt];
			iAnsNum = iCnt;
		}
		//printf(" [%02d]...%f%%\n", iCnt, vecOut[iCnt]);
	}

	return (iAnsNum == Mn.sLabel);
}

// 学習処理
int	FF_Neural::Training(CStatic &m_stResDisp)
{
	CString strDispMsg;
	m_il0Size = MNIST_IMG_SIZE;
	FILE *fpImg = NULL;
	FILE *fpLabel = NULL;

	// 教師データのオープン
	CString strImgPath, strLabelPath;
	strImgPath.Format(_T("%s\\" MNIST_TRAINING_IMG_PATH), m_pszTrainFolderPath);
	strLabelPath.Format(_T("%s\\" MNIST_TRAINING_LABEL_PATH), m_pszTrainFolderPath);
	if(GetMnistDataFP(&fpImg, &fpLabel, strImgPath, strLabelPath)!= 0){
		AfxMessageBox(_T("Failed to Load Training Data"));
		return -1;
	}

	// 重みをランダムで初期化
	InitWeight();

	// 訓練
	for(int iCnt = 0; iCnt < MNIST_TRANING_IMG_CNT; iCnt++){

		MNIST_DATA sTraningData;
		GetMnistImgLabel(fpImg, fpLabel, sTraningData, iCnt);

		doubleVec vecImg, vecLabel;
		ConvMNISTtoVec(sTraningData, vecImg, vecLabel);

		double dErr = BackProp(vecImg, vecLabel);



		if(iCnt % 500 == 0) {
			// メッセージ処理を行う
			MSG msg;
			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
				if (msg.message == WM_QUIT || msg.message == WM_DESTROY || msg.message == WM_CLOSE) continue;
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			// 途中経過を表示
			strDispMsg.Format(_T("Training processing...\n %d/%d : miss:%f"), 
					iCnt, MNIST_TRANING_IMG_CNT, dErr);
			m_stResDisp.SetWindowText(strDispMsg);
		}
	}

	// 教師データのClose
	ReleaseMnistDataFP(&fpImg, &fpLabel);

	// テストデータのオープン
	strImgPath.Format(_T("%s\\" MNIST_TEST_IMG_PATH), m_pszTestFolderPath);
	strLabelPath.Format(_T("%s\\" MNIST_TEST_LABEL_PATH), m_pszTestFolderPath);
	if(GetMnistDataFP(&fpImg, &fpLabel, strImgPath, strLabelPath)!= 0){
		AfxMessageBox(_T("Failed to Load Test Data"));
		return -1;
	}

	// テスト
	UINT uiOKNum = 0;
	for(int iCnt = 0; iCnt < MNIST_TEST_IMG_CNT; iCnt++){

		MNIST_DATA sTraningData;
		GetMnistImgLabel(fpImg, fpLabel, sTraningData, iCnt);  

		doubleVec vecImg, vecLabel;
		ConvMNISTtoVec(sTraningData, vecImg, vecLabel);

		doubleVec vecZ1, vecOut;
		ForwardProp(vecImg, vecZ1, vecOut);

		if(CheckInference(sTraningData, vecOut)){
			uiOKNum++;
		}

		if(iCnt % 500 == 0) {
			// メッセージ処理を行う
			MSG msg;
			while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
				TranslateMessage(&msg);
				DispatchMessage(&msg);
			}

			// 途中経過を表示
			strDispMsg.Format(_T("Test Processing...\n %d/%d"), iCnt, MNIST_TEST_IMG_CNT);
			m_stResDisp.SetWindowText(strDispMsg);
		}		
	}

	// テストデータのClose
	ReleaseMnistDataFP(&fpImg, &fpLabel);

	// テスト結果表示
	strDispMsg.Format(_T("Training done...\n Accuracy:%5.2f%%"), (double)uiOKNum * 100 /MNIST_TEST_IMG_CNT);
	m_stResDisp.SetWindowText(strDispMsg);

	// 重みの保存
	SaveWeight();

	return 0;
}


// 推論処理
void FF_Neural::Inference(const doubleVec& vecIn, CStatic &m_stResDisp)
{
	// 重みの読み込み
	if(m_vecW1.size() == 0 && LoadWeight() != 0){
		return;
	}

	// 順方向伝搬による推論
	doubleVec vecZ1, vecOut;
	ForwardProp(vecIn, vecZ1, vecOut);

	// 認識結果の表示
	CString strResult, strPercentage;
	int iAnsNum = -1;
	double dMaxLabel = -1;
	for (int iCnt = 0; iCnt < 10; iCnt++) {
		if (vecOut[iCnt] > dMaxLabel) {
			dMaxLabel = vecOut[iCnt];
			iAnsNum = iCnt;
		}
		CString strBuf;
		strBuf.Format(_T("[%01d]...%5.2f%%\n"), iCnt, vecOut[iCnt] * 100);
		strPercentage += strBuf;
	}

	// 正解率の表示
	strResult.Format(_T("Estimate...%d\n\n%s"), iAnsNum, strPercentage);
	m_stResDisp.SetWindowText(strResult);

	return;
}
