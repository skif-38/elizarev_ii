import re
pat = re.compile(r'\b\w+[ое]\s\w+(?:[л]\w*|[тч][ьи][ся]*)\b|\b\w+(?:[л]\w*|[тч][ьи][ся]*)\s\w+[ое]\b')

with open("a.txt", encoding="utf-8") as f:
    text = f.read()
for i in pat.finditer(text):
    print(i.group())