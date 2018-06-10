// MnistFileSelDlg.cpp : 実装ファイル
//

#include "stdafx.h"
#include "NeuralNetwork_NumOCR.h"
#include "MnistFileSelDlg.h"
#include "afxdialogex.h"


// MnistFileSelDlg ダイアログ

IMPLEMENT_DYNAMIC(MnistFileSelDlg, CDialogEx)
CString MnistFileSelDlg::m_strTrainFilePath;
CString MnistFileSelDlg::m_strTestFilePath;

MnistFileSelDlg::MnistFileSelDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_MNIST_SELDLG, pParent)
{

}

MnistFileSelDlg::~MnistFileSelDlg()
{
}

void MnistFileSelDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}


BEGIN_MESSAGE_MAP(MnistFileSelDlg, CDialogEx)
	ON_BN_CLICKED(IDC_BTN_TRAINBRW, &MnistFileSelDlg::OnBnClickedBtnTrainbrw)
	ON_BN_CLICKED(IDC_BTN_TESTBRW, &MnistFileSelDlg::OnBnClickedBtnTestbrw)
	ON_BN_CLICKED(IDOK, &MnistFileSelDlg::OnBnClickedOk)
	ON_BN_CLICKED(IDCANCEL, &MnistFileSelDlg::OnBnClickedCancel)
END_MESSAGE_MAP()


// MnistFileSelDlg メッセージ ハンドラー


BOOL MnistFileSelDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	if (!m_strTrainFilePath.IsEmpty()) {
		((CEdit*)GetDlgItem(IDC_EDIT_TRAINPATH))->SetWindowTextW(m_strTrainFilePath);
	}
	if (!m_strTestFilePath.IsEmpty()) {
		((CEdit*)GetDlgItem(IDC_EDIT_TESTPATH))->SetWindowTextW(m_strTestFilePath);
	}

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 例外 : OCX プロパティ ページは必ず FALSE を返します。
}


BOOL MnistFileSelDlg::FileSelDlg(CString &strSelFilePath, LPCTSTR lpszDlgTitle/* = NULL*/)
{
	strSelFilePath.Empty();
	OPENFILENAME ofn = { sizeof(ofn) };
	ofn.hwndOwner = GetSafeHwnd();
	ofn.lpstrFilter = _T(".gz\0*.gz\0");
	ofn.lpstrFile = strSelFilePath.GetBuffer(MAX_PATH + 1);
	ofn.nMaxFile = MAX_PATH;
	ofn.nMaxFileTitle = MAX_PATH;
	ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	ofn.lpstrTitle = lpszDlgTitle ? lpszDlgTitle:_T("Select MNIST Training/Test Image");

	BOOL bSelRes = GetOpenFileName(&ofn);
	strSelFilePath.ReleaseBuffer();
	return bSelRes;
}

void MnistFileSelDlg::OnBnClickedBtnTrainbrw()
{
	if (FileSelDlg(m_strTrainFilePath, _T("Select MNIST Training Image File"))) {
		((CEdit*)GetDlgItem(IDC_EDIT_TRAINPATH))->SetWindowTextW(m_strTrainFilePath);
	}
}


void MnistFileSelDlg::OnBnClickedBtnTestbrw()
{
	if (FileSelDlg(m_strTestFilePath, _T("Select MNIST Test Image File"))) {
		((CEdit*)GetDlgItem(IDC_EDIT_TESTPATH))->SetWindowTextW(m_strTestFilePath);
	}
}


void MnistFileSelDlg::OnBnClickedOk()
{
	// ファイルパスの実在チェック
	if (GetFileAttributes(m_strTrainFilePath) != -1
		&& GetFileAttributes(m_strTestFilePath) != -1) {
		CDialogEx::OnOK();
	}
	else {
		AfxMessageBox(_T("File Not Found!"));
	}
}


void MnistFileSelDlg::OnBnClickedCancel()
{
	m_strTestFilePath.Empty();
	m_strTrainFilePath.Empty();
	CDialogEx::OnCancel();
}
