const fs = require('fs');

const html = fs.readFileSync('src/dh360_hi/api/static/demo.html', 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("Script not found");
let scriptCode = scriptMatch[1];

// We use string replacement to expose private variables to global scope for the test
scriptCode = scriptCode.replace(/let isPreflightMode = false;/, 'global.isPreflightMode = false;');
scriptCode = scriptCode.replace(/isPreflightMode = true;/g, 'global.isPreflightMode = true;');
scriptCode = scriptCode.replace(/isPreflightMode = false;/g, 'global.isPreflightMode = false;');
scriptCode = scriptCode.replace(/if \(isPreflightMode\)/g, 'if (global.isPreflightMode)');

// Expose abortController
scriptCode = scriptCode.replace(/let abortController = null;/, 'global.abortController = null;');
scriptCode = scriptCode.replace(/abortController = currentAbortController;/g, 'global.abortController = currentAbortController;');
scriptCode = scriptCode.replace(/abortController = null;/g, 'global.abortController = null;');
scriptCode = scriptCode.replace(/if \(abortController\)/g, 'if (global.abortController)');
scriptCode = scriptCode.replace(/abortController\.abort\(\);/g, 'global.abortController.abort();');
scriptCode = scriptCode.replace(/if \(isConnecting \|\| abortController\)/g, 'if (global.isConnecting || global.abortController)');
scriptCode = scriptCode.replace(/abortController === currentAbortController/g, 'global.abortController === currentAbortController');

global.abortController = null;
global.isConnecting = false;
global.fetchResponses = [];
global.fetchArgs = [];
global.fetch = async (url, options) => {
    global.fetchArgs.push({url, options});
    const res = global.fetchResponses.shift() || { status: 200, json: {} };
    if (res.error) throw new Error(res.error);
    return {
        ok: res.status >= 200 && res.status < 300,
        status: res.status,
        json: async () => res.json || {},
        body: res.body || { getReader: () => ({ read: async () => ({ done: true }) }) }
    };
};

const domState = {
    "token-input": { value: "test-token" },
    "auth-status": { textContent: "", className: "" },
    "auth-person": { textContent: "" },
    "auth-id": { textContent: "" },
    "auth-started": { textContent: "" },
    "auth-updated": { textContent: "" },
    "auth-ended": { textContent: "" },
    "registration-section": { style: { display: 'none' } },
    "show-register-btn": { style: { display: 'block' }, disabled: false },
    "start-btn": { disabled: false },
    "display-name": { value: "Test" },
    "preferred-name": { value: "" },
    "role": { value: "" },
    "department": { value: "" },
    "office-room": { value: "" },
    "working-hours": { value: "" },
    "profile-description": { value: "" },
    "register-btn": { disabled: false },
    "current-direction": { textContent: "" },
    "current-instruction": { textContent: "" },
    "direction-progress": { textContent: "" },
    "total-progress": { textContent: "" },
    "notice-box": { style: {}, textContent: "", className: "" }
};

global.document = {
    getElementById: (id) => domState[id] || { style: {}, textContent: "" },
    querySelectorAll: () => []
};
global.appendLog = () => {};
global.setSseIndicator = () => {};
global.clearContextUI = () => {};
global.TextDecoder = class { decode() { return ""; } };

global.setInterval = () => 1;
global.clearInterval = () => {};

eval(scriptCode);

async function runTests() {
    console.log("Running frontend logic regression tests...");

    global.fetchResponses = [
        { status: 200 },
        { status: 409, json: { detail: "No active enrollment session" } },
        { status: 201, json: { authorization_id: "auth-123" } },
        { status: 200 }
    ];
    await showRegistrationForm();
    if (!global.isPreflightMode) throw new Error("Should be in preflight mode");
    if (!domState["show-register-btn"].disabled) throw new Error("Button should be disabled during preflight");
    
    global.fetchResponses = [{ status: 200 }];
    await handleSseBlock('event: authorization_updated\ndata: {"status": "AUTHORIZED", "display_name": "Apurbo"}');
    if (domState["registration-section"].style.display === 'block') throw new Error("Registration should be hidden");
    if (!domState["notice-box"].textContent.includes("already registered")) throw new Error("Should notify already registered");
    if (domState["show-register-btn"].disabled) throw new Error("Button should be re-enabled");
    console.log("✓ Existing person detection prevents duplicate creation");

    global.isPreflightMode = true;
    domState["notice-box"].textContent = "";
    await handleSseBlock('event: authorization_updated\ndata: {"status": "UNCERTAIN"}');
    if (domState["registration-section"].style.display === 'block') throw new Error("Registration should be hidden on UNCERTAIN");
    if (!domState["notice-box"].textContent.includes("Identity could not be confirmed")) throw new Error("Should notify uncertain");
    
    global.fetchResponses = [{ status: 200 }];
    await handleSseBlock('event: authorization_updated\ndata: {"status": "UNKNOWN"}');
    if (domState["registration-section"].style.display !== 'block') throw new Error("Registration should be shown on UNKNOWN");
    if (global.isPreflightMode) throw new Error("Should exit preflight mode");
    console.log("✓ UNKNOWN and UNCERTAIN have different outcomes");

    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 },
        { status: 200 },
        { status: 201, json: { authorization_id: "auth-456" } }
    ];
    await startAuthorization();
    if (global.fetchArgs[0].url !== "/authorization/sessions/current" || global.fetchArgs[0].options.method !== "DELETE") throw new Error("Auth cancel must be first");
    if (global.fetchArgs[1].url !== "/persons/enrollment/sessions/current" || global.fetchArgs[1].options.method !== "DELETE") throw new Error("Enroll cancel must be second");
    if (global.fetchArgs[2].url !== "/authorization/sessions" || global.fetchArgs[2].options.method !== "POST") throw new Error("Auth start must be third");
    console.log("✓ Session cancellation completes before a replacement starts");

    global.abortController = { abort: () => {} };
    global.fetchResponses = [{ status: 200 }];
    await cancelActiveSession("test");
    if (global.abortController !== null) throw new Error("abortController should be forcefully nulled to prevent stale cleanup races");
    console.log("✓ Stale SSE events cannot overwrite the current UI");

    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 },
        { status: 409, json: { detail: "Enrollment active" } },
        { status: 200 }
    ];
    const enrollOk = await startEnrollment("test-token", "person-123");
    const deleteArg = global.fetchArgs.find(a => a.options.method === "DELETE" && a.url === "/persons/person-123");
    if (deleteArg) throw new Error("startEnrollment MUST NOT delete a pre-existing person!");
    if (enrollOk) throw new Error("startEnrollment should return false on failure");
    console.log("✓ startEnrollment never deletes a pre-existing person on failure");

    // Test handleRegister rollback
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 201, json: { person_id: "new-person-456", display_name: "Test Person" } }, // POST /persons
        { status: 409, json: { detail: "Enrollment active" } }, // PATCH consent fails
        { status: 200 } // DELETE /persons/new-person-456
    ];
    
    // Simulate DOM for qa-table
    global.document.querySelectorAll = (sel) => {
        if (sel === "#qa-table tbody tr") return [];
        return [];
    };
    
    await handleRegister();
    
    const deleteNewArg = global.fetchArgs.find(a => a.options.method === "DELETE" && a.url === "/persons/new-person-456");
    if (!deleteNewArg) throw new Error("handleRegister MUST rollback the newly created person if enrollment fails!");
    console.log("✓ handleRegister safely rolls back ONLY the newly created profile");

    console.log("All frontend logic tests passed successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
