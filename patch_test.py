import re

with open("tests/frontend/test_demo_logic.js", "r") as f:
    content = f.read()

content = content.replace('domState["total-progress"] = { textContent: "" };', 'domState["total-progress"] = { textContent: "" };\ndomState["notice-box"] = { style: {}, textContent: "", className: "" };')
content = content.replace('domState["total-progress"]: { textContent: "" }', 'domState["total-progress"]: { textContent: "" },\n    "notice-box": { style: {}, textContent: "", className: "" }')

# Replace global.lastNotice checks with notice-box checks
content = content.replace('global.lastNotice = "";', 'domState["notice-box"].textContent = "";')
content = content.replace('console.log("lastNotice is:", global.lastNotice); if (!global.lastNotice.includes', 'if (!domState["notice-box"].textContent.includes')
content = content.replace('if (!global.lastNotice.includes', 'if (!domState["notice-box"].textContent.includes')

with open("tests/frontend/test_demo_logic.js", "w") as f:
    f.write(content)
