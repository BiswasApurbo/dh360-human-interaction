import re
with open("tests/frontend/test_demo_logic.js", "r") as f:
    content = f.read()

content = content.replace('scriptCode = scriptCode.replace(/let isPreflightMode = false;/, \'global.isPreflightMode = false;\');', 
'''scriptCode = scriptCode.replace(/let isPreflightMode = false;/, 'global.isPreflightMode = false;');
scriptCode = scriptCode.replace(/let abortController = null;/, 'global.abortController = null;');
scriptCode = scriptCode.replace(/abortController = currentAbortController;/g, 'global.abortController = currentAbortController;');
scriptCode = scriptCode.replace(/abortController = null;/g, 'global.abortController = null;');
scriptCode = scriptCode.replace(/if \\(abortController\\)/g, 'if (global.abortController)');
scriptCode = scriptCode.replace(/abortController\\.abort\\(\\);/g, 'global.abortController.abort();');
scriptCode = scriptCode.replace(/if \\(isConnecting || abortController\\)/g, 'if (global.isConnecting || global.abortController)');
''')

with open("tests/frontend/test_demo_logic.js", "w") as f:
    f.write(content)
