Option Explicit

' ---- HTTP GET: platform-switched, with timeouts ----------------------------

#If Mac Then
Private Function HttpGetJson(url As String) As String
    HttpGetJson = AppleScriptTask("httpGet.scpt", "httpGet", url)
End Function
#Else
Private Function HttpGetJson(url As String) As String
    Dim http As Object
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    http.setTimeouts 5000, 5000, 30000, 30000
    http.Open "GET", url, False
    http.send
    HttpGetJson = http.responseText
End Function
#End If

' ---- JSON string extractor -------------------------------------------------
' Extracts the value of a string-typed key from a flat JSON object fragment.
' Handles \" escape sequences inside values.

Private Function JsonStr(obj As String, key As String) As String
    Dim k As String, p As Long, q As Long
    k = """" & key & """:"""
    p = InStr(obj, k)
    If p = 0 Then JsonStr = "": Exit Function
    p = p + Len(k)
    q = p
    Do While q <= Len(obj)
        Select Case Mid(obj, q, 1)
            Case "\"
                q = q + 2           ' skip backslash + escaped char
            Case """"
                Exit Do             ' closing quote found
            Case Else
                q = q + 1
        End Select
    Loop
    JsonStr = Mid(obj, p, q - p)
End Function

' ---- JSON number extractor (matches RunPredictions pattern) ----------------

Private Function JsonNum(resp As String, key As String) As Variant
    Dim k As String, p As Long, q As Long, s As String
    k = """" & key & """:"
    p = InStr(resp, k)
    If p = 0 Then JsonNum = CVErr(xlErrValue): Exit Function
    p = p + Len(k)
    q = p
    Do While q <= Len(resp)
        Select Case Mid(resp, q, 1)
            Case ",", "}", " "
                Exit Do
        End Select
        q = q + 1
    Loop
    s = Trim(Mid(resp, p, q - p))
    If IsNumeric(s) Then JsonNum = CDbl(s) Else JsonNum = CVErr(xlErrValue)
End Function

' ---- Find matching close-brace (handles nesting) ---------------------------

Private Function FindCloseBrace(s As String, openPos As Long) As Long
    Dim depth As Long, i As Long, c As String
    depth = 0
    For i = openPos To Len(s)
        c = Mid(s, i, 1)
        If c = "{" Then depth = depth + 1
        If c = "}" Then
            depth = depth - 1
            If depth = 0 Then FindCloseBrace = i: Exit Function
        End If
    Next i
    FindCloseBrace = 0
End Function

' ---- Main: GET /models and populate destination table ----------------------

Public Sub GetModels(TABLE_NAME As String, Optional brandFilter As String = "")
    Const BASE_URL As String = "http://127.0.0.1:8000/models"

    Dim url As String, resp As String, obj As String
    Dim lo As ListObject, r As ListRow
    Dim p As Long, closePos As Long, i As Long

    url = BASE_URL
    If Len(Trim(brandFilter)) > 0 Then
        url = url & "?brand=" & Replace(Trim(brandFilter), " ", "%20")
    End If

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual
    Application.StatusBar = "Fetching models from API..."
    DoEvents

    On Error GoTo CleanFail

    resp = HttpGetJson(url)

    Debug.Print "GetModels resp length=" & Len(resp)
    Debug.Print "GetModels resp preview=" & Left(resp, 200)

    ' Clear existing rows from the destination table
    Set lo = ActiveSheet.ListObjects(TABLE_NAME)
    If Not lo.DataBodyRange Is Nothing Then
        lo.DataBodyRange.Delete
    End If

    ' Walk the JSON array, extracting one { } object at a time
    i = 0
    p = 1
    Do
        p = InStr(p, resp, "{")
        If p = 0 Then Exit Do

        closePos = FindCloseBrace(resp, p)
        If closePos = 0 Then Exit Do

        obj = Mid(resp, p, closePos - p + 1)

        Set r = lo.ListRows.Add
        r.Range.Cells(1, lo.ListColumns("brand").Index).Value          = JsonStr(obj, "brand")
        r.Range.Cells(1, lo.ListColumns("model").Index).Value          = JsonStr(obj, "model")
        r.Range.Cells(1, lo.ListColumns("estimator").Index).Value      = JsonStr(obj, "estimator")
        r.Range.Cells(1, lo.ListColumns("base_estimator").Index).Value = JsonStr(obj, "base_estimator")
        r.Range.Cells(1, lo.ListColumns("params").Index).Value         = JsonStr(obj, "params")
        r.Range.Cells(1, lo.ListColumns("score_r2").Index).Value       = JsonNum(obj, "score_r2")

        i = i + 1
        p = closePos + 1

        Application.StatusBar = "Loaded " & i & " models..."
        DoEvents
    Loop

    Application.StatusBar = "Done — " & i & " models loaded."

CleanExit:
    Application.StatusBar = False
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    Exit Sub

CleanFail:
    MsgBox "GetModels failed: " & Err.Description, vbExclamation
    Resume CleanExit
End Sub
