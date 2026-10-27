with open('src/dh360_hi/api/static/demo.html', 'r') as f:
    content = f.read()

content = content.replace("display=\\'block\\'", "display='block'")

with open('src/dh360_hi/api/static/demo.html', 'w') as f:
    f.write(content)
