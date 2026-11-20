Option Explicit

Dim files, shell, appDir, python, mainFile, command, index
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
appDir = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(appDir, ".venv\Scripts\pythonw.exe")
mainFile = files.BuildPath(appDir, "main.py")

If Not files.FileExists(python) Then
    MsgBox "Khong tim thay Python: " & python & vbCrLf & _
        "Hay tao moi truong .venv va cai thu vien truoc khi chay.", vbCritical, "ReviewTrans"
    WScript.Quit 1
End If

If Not files.FileExists(mainFile) Then
    MsgBox "Khong tim thay main.py trong thu muc ung dung.", vbCritical, "ReviewTrans"
    WScript.Quit 1
End If

shell.CurrentDirectory = appDir
command = Quote(python) & " " & Quote(mainFile)
For index = 0 To WScript.Arguments.Count - 1
    command = command & " " & Quote(WScript.Arguments(index))
Next

On Error Resume Next
shell.Run command, 1, False
If Err.Number <> 0 Then
    MsgBox "Khong mo duoc ReviewTrans: " & Err.Description, vbCritical, "ReviewTrans"
    WScript.Quit 1
End If
On Error GoTo 0

Function Quote(value)
    Quote = Chr(34) & Replace(value, Chr(34), "") & Chr(34)
End Function
