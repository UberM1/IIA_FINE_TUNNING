function Image()
    return {}
end

function Figure()
    return {}
end

function Str(el)
    if el.text:match("^%[image%]$") or el.text:match("^%[fig.*%]$") then
        return {}
    end
    return el
end

function Para(el)
    if #el.content == 0 then
        return {}
    end
    return el
end