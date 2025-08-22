import chardet

with open("inference.py", "rb") as f:
    raw = f.read()
    result = chardet.detect(raw)
    print(result)
