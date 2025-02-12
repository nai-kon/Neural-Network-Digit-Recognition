// MnistFileSelDlg.cpp : 実装ファイル
//

#include "stdafx.h"
#include "NeuralNetwork_NumOCR.h"
#include "MnistFileSelDlg.h"
#include "afxdialogex.h"


// MnistFileSelDlg ダイアログ

IMPLEMENT_DYNAMIC(MnistFileSelDlg, CDialogEx)
CString MnistFileSelDlg::m_strTrainFolderPath;
CString MnistFileSelDlg::m_strTestFolderPath;

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

	if (!m_strTrainFolderPath.IsEmpty()) {
		((CEdit*)GetDlgItem(IDC_EDIT_TRAINPATH))->SetWindowTextW(m_strTrainFolderPath);
	}
	if (!m_strTestFolderPath.IsEmpty()) {
		((CEdit*)GetDlgItem(IDC_EDIT_TESTPATH))->SetWindowTextW(m_strTestFolderPath);
	}

	return TRUE;  // return TRUE unless you set the focus to a control
				  // 例外 : OCX プロパティ ページは必ず FALSE を返します。
}


BOOL MnistFileSelDlg::FolderSelDlg(CString &strSelFilePath, LPCTSTR lpszDlgTitle/* = NULL*/)
{
	CFolderPickerDialog dlg;
	dlg.m_ofn.lpstrTitle = _T("Select MNIST Training/Test Image Folder");
	if (dlg.DoModal() == IDOK) {
		strSelFilePath = dlg.GetPathName();
		return TRUE;
	}
	return FALSE;
}

void MnistFileSelDlg::OnBnClickedBtnTrainbrw()
{
	if (FolderSelDlg(m_strTrainFolderPath, _T("Select MNIST Training Image/Label Folder"))) {
		((CEdit*)GetDlgItem(IDC_EDIT_TRAINPATH))->SetWindowTextW(m_strTrainFolderPath);
	}
}


void MnistFileSelDlg::OnBnClickedBtnTestbrw()
{
	if (FolderSelDlg(m_strTestFolderPath, _T("Select MNIST Test Image/Label Folder"))) {
		((CEdit*)GetDlgItem(IDC_EDIT_TESTPATH))->SetWindowTextW(m_strTestFolderPath);
	}
}


void MnistFileSelDlg::OnBnClickedOk()
{
	// ファイルパスの実在チェック
	if (GetFileAttributes(m_strTrainFolderPath) != -1
		&& GetFileAttributes(m_strTestFolderPath) != -1) {
		CDialogEx::OnOK();
	}
	else {
		AfxMessageBox(_T("File Not Found!"));
	}
}


void MnistFileSelDlg::OnBnClickedCancel()
{
	m_strTestFolderPath.Empty();
	m_strTrainFolderPath.Empty();
	CDialogEx::OnCancel();
}
