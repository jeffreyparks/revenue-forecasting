on httpPost(arg)
    set arg to arg as string
    -- arg is "url\tbody"; tab is the delimiter
    set AppleScript's text item delimiters to (ASCII character 9)
    set parts to text items of arg
    set AppleScript's text item delimiters to ""
    set url_  to item 1 of parts
    set body_ to item 2 of parts
    set cmd to "curl -sS --connect-timeout 5 --max-time 30 -X POST " & Â
        quoted form of url_ & Â
        " -H 'Content-Type: application/json' --data-binary " & Â
        quoted form of body_ & " 2>&1"
    try
        set out_ to do shell script cmd
    on error errMsg number errNum
        set out_ to "{\"error\":\"applescript\",\"num\":" & errNum & ",\"msg\":\"" & errMsg & "\"}"
    end try
    return out_ as string
end httpPost