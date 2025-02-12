#pragma once


// MnistFileSelDlg ダイアログ

class MnistFileSelDlg : public CDialogEx
{
	DECLARE_DYNAMIC(MnistFileSelDlg)

public:
	MnistFileSelDlg(CWnd* pParent = nullptr);   // 標準コンストラクター
	virtual ~MnistFileSelDlg();

// ダイアログ データ
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_MNIST_SELDLG };
#endif

public:
	static CString m_strTrainFolderPath;
	static CString m_strTestFolderPath;
	LPCTSTR GetTrainFolderPath() { return (LPCTSTR)m_strTrainFolderPath; }
	LPCTSTR GetTestFolderPath() { return (LPCTSTR)m_strTestFolderPath; }

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV サポート

	BOOL FolderSelDlg(CString &strSelFilePath, LPCTSTR lpszDlgTitle = NULL);



	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnTrainbrw();
	afx_msg void OnBnClickedBtnTestbrw();
	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
};
