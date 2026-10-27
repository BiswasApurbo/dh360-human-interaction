import re

with open("tests/frontend/test_demo_logic.js", "r") as f:
    content = f.read()

test_old = """    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 },
        { status: 409, json: { detail: "Enrollment active" } },
        { status: 200 }
    ];
    await startEnrollment("test-token", "person-123");
    const deleteArg = global.fetchArgs.find(a => a.options.method === "DELETE" && a.url === "/persons/person-123");
    if (!deleteArg) throw new Error("Failed enrollment should rollback the person creation");"""

test_new = """    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 },
        { status: 409, json: { detail: "Enrollment active" } },
        { status: 200 }
    ];
    const enrollOk = await startEnrollment("test-token", "person-123");
    const deleteArg = global.fetchArgs.find(a => a.options.method === "DELETE" && a.url === "/persons/person-123");
    if (deleteArg) throw new Error("startEnrollment MUST NOT delete a pre-existing person!");
    if (enrollOk) throw new Error("startEnrollment should return false on failure");"""

content = content.replace(test_old, test_new)

with open("tests/frontend/test_demo_logic.js", "w") as f:
    f.write(content)
