Option Explicit

' ---- HTTP: platform-switched, with timeouts --------------------------------

#If Mac Then
Private Function HttpPostJson(url As String, body As String) As String
    Dim arg As String
    arg = url & vbTab & body          ' single string param, tab-delimited
    HttpPostJson = AppleScriptTask("httpPost.scpt", "httpPost", arg)
End Function
#Else
Private Function HttpPostJson(url As String, body As String) As String
    Dim http As Object
    ' ServerXMLHTTP supports setTimeouts; plain XMLHTTP does not
    Set http = CreateObject("MSXML2.ServerXMLHTTP.6.0")
    ' resolve, connect, send, receive — all in ms
    http.setTimeouts 5000, 5000, 30000, 30000
    http.Open "POST", url, False
    http.setRequestHeader "Content-Type", "application/json"
    http.send body
    HttpPostJson = http.responseText
End Function
#End If

' ---- JSON number extractor (unchanged) -------------------------------------

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

' ---- Main loop with progress + DoEvents ------------------------------------

Public Sub RunPredictions(TABLE_NAME)
    Const url As String = "http://127.0.0.1:8000/predict"

    Dim lo As ListObject, r As ListRow
    Dim brand As String, model As String, weekstart As String, spend As Double
    Dim body As String, resp As String
    Dim i As Long, total As Long

    Set lo = ActiveSheet.ListObjects(TABLE_NAME)
    total = lo.ListRows.Count

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual

    On Error GoTo CleanFail

    i = 0
    For Each r In lo.ListRows
        i = i + 1
        Application.StatusBar = "Predicting " & i & " / " & total & "..."
        DoEvents
    
        Dim vBrand As Variant, vModel As Variant, vWeek As Variant, vSpend As Variant
        vBrand = r.Range.Cells(1, lo.ListColumns("brand").Index).Value
        vModel = r.Range.Cells(1, lo.ListColumns("model").Index).Value
        vWeek = r.Range.Cells(1, lo.ListColumns("weekstart").Index).Value
        vSpend = r.Range.Cells(1, lo.ListColumns("spend").Index).Value
    
        Debug.Print "row " & i, _
                    "table=[" & TypeName(TABLE_NAME) & ":" & TABLE_NAME & "]", _
                    "brand=[" & TypeName(vBrand) & ":" & vBrand & "]", _
                    "model=[" & TypeName(vModel) & ":" & vModel & "]", _
                    "weekstart=[" & TypeName(vWeek) & ":" & vWeek & "]", _
                    "spend=[" & TypeName(vSpend) & ":" & vSpend & "]"
    
        If Not IsDate(vWeek) Then Err.Raise vbObjectError + 1, , "weekstart is not a date"
        If Not IsNumeric(vSpend) Then Err.Raise vbObjectError + 2, , "spend is not numeric"
    
        brand = CStr(vBrand)
        model = CStr(vModel)
        weekstart = Format(CDate(vWeek), "yyyy-mm-dd")
        spend = CDbl(vSpend)

        If Len(Trim(CStr(vModel))) = 0 Then
            r.Range.Cells(1, lo.ListColumns("nd_pred_lo").Index).Value = 0
            r.Range.Cells(1, lo.ListColumns("nd_pred_med").Index).Value = 0
            r.Range.Cells(1, lo.ListColumns("nd_pred_hi").Index).Value = 0
            GoTo NextRow
        End If
    
        body = "{""brand"":""" & brand & """,""model"":""" & model & _
               """,""weekstart"":""" & weekstart & """,""spend"":" & spend & "}"
    
        Dim vResp As Variant
        vResp = HttpPostJson(url, body)
        Debug.Print "  resp TypeName=" & TypeName(vResp) & " IsNull=" & IsNull(vResp) & _
                    " IsEmpty=" & IsEmpty(vResp) & " IsError=" & IsError(vResp)
        On Error Resume Next
        Debug.Print "  resp value=" & CStr(vResp)
        On Error GoTo CleanFail
        resp = ""
        If VarType(vResp) = vbString Then resp = CStr(vResp)
    
        r.Range.Cells(1, lo.ListColumns("nd_pred_lo").Index).Value = JsonNum(resp, "nd_lo")
        r.Range.Cells(1, lo.ListColumns("nd_pred_med").Index).Value = JsonNum(resp, "nd_med")
        r.Range.Cells(1, lo.ListColumns("nd_pred_hi").Index).Value = JsonNum(resp, "nd_hi")
NextRow:
    Next r

CleanExit:
    Application.StatusBar = False
    Application.Calculation = xlCalculationAutomatic
    Application.ScreenUpdating = True
    Exit Sub

CleanFail:
    MsgBox "Row " & i & " failed: " & Err.Description, vbExclamation
    Resume CleanExit
End Sub

