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
	static CString m_strTrainFilePath;
	static CString m_strTestFilePath;
	LPCTSTR GetTraiFilePath() { return (LPCTSTR)m_strTrainFilePath; }
	LPCTSTR GetTestFilePath() { return (LPCTSTR)m_strTestFilePath; }

protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV サポート

	BOOL FileSelDlg(CString &strSelFilePath, LPCTSTR lpszDlgTitle = NULL);



	DECLARE_MESSAGE_MAP()
public:
	afx_msg void OnBnClickedBtnTrainbrw();
	afx_msg void OnBnClickedBtnTestbrw();
	virtual BOOL OnInitDialog();
	afx_msg void OnBnClickedOk();
	afx_msg void OnBnClickedCancel();
};
