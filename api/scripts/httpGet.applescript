on httpGet(url_)
    set url_ to url_ as string
    set cmd to "curl -sS --connect-timeout 5 --max-time 30 -X GET " & quoted form of url_ & " 2>&1"
    try
        set out_ to do shell script cmd
    on error errMsg number errNum
        set out_ to "{\"error\":\"applescript\",\"num\":" & errNum & ",\"msg\":\"" & errMsg & "\"}"
    end try
    return out_ as string
end httpGet
