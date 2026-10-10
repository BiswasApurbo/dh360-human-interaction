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
    "enrollment-feedback": { textContent: "", style: {} },
    "people-tbody": {
        innerHTML: "",
        children: [],
        appendChild: function(c) { this.children.push(c); }
    },
    "notice-box": { style: {}, textContent: "", className: "" }
};

global.document = {
    getElementById: (id) => domState[id] || { style: {}, textContent: "" },
    querySelectorAll: () => [],
    createElement: (tag) => ({
        tag,
        style: {},
        textContent: "",
        children: [],
        appendChild: function(c) { this.children.push(c); }
    })
};
global.window = {
    confirm: () => true,
    addEventListener: () => {}
};
global.appendLog = () => {};
global.setSseIndicator = () => {};
global.clearContextUI = () => {};
global.TextDecoder = class { decode() { return ""; } };

global.setInterval = (fn) => { global._latestIntervalFn = fn; return 1; };
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
    if (global.isPreflightMode) throw new Error("UNCERTAIN must reach terminal state and exit preflight mode");
    if (domState["show-register-btn"].disabled) throw new Error("Button should be re-enabled for retry on UNCERTAIN");

    global.isPreflightMode = true;
    global.fetchResponses = [{ status: 200 }];
    await handleSseBlock('event: authorization_updated\ndata: {"status": "UNKNOWN"}');
    if (domState["registration-section"].style.display !== 'block') throw new Error("Registration should be shown on UNKNOWN");
    if (global.isPreflightMode) throw new Error("Should exit preflight mode on UNKNOWN");
    console.log("✓ UNKNOWN and UNCERTAIN have different outcomes and both reach terminal states");

    // Test TIMEOUT reaches terminal state
    global.isPreflightMode = true;
    domState["show-register-btn"].disabled = true;
    global.fetchResponses = [{ status: 200 }];
    await handleSseBlock('event: authorization_updated\ndata: {"status": "TIMEOUT"}');
    if (global.isPreflightMode) throw new Error("TIMEOUT must exit preflight mode");
    if (!domState["notice-box"].textContent.includes("timed out")) throw new Error("TIMEOUT should show timeout notice");
    if (domState["show-register-btn"].disabled) throw new Error("Button should be re-enabled on TIMEOUT");
    console.log("✓ Preflight TIMEOUT reaches terminal outcome and offers retry");

    // Test polling fallback catches terminal status without SSE
    global.isPreflightMode = true;
    domState["registration-section"].style.display = 'none';
    domState["show-register-btn"].disabled = true;
    global.fetchResponses = [
        { status: 200, json: { authorization_id: "auth-123", status: "authorized", display_name: "Apurbo" } },
        { status: 200 }
    ];
    await global._latestIntervalFn();
    if (global.isPreflightMode) throw new Error("Polling fallback must resolve preflight mode");
    if (!domState["notice-box"].textContent.includes("already registered")) throw new Error("Polling fallback should show already registered notice");
    if (domState["show-register-btn"].disabled) throw new Error("Button should be re-enabled after polling resolution");
    console.log("✓ Preflight polling fallback resolves terminal outcome without SSE events");

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

    // Test Issue 1: updateEnrollmentProgressUI displays rejection feedback and stage instructions
    updateEnrollmentProgressUI({
        current_direction: "FRONT",
        direction_counts: { front: 0, right: 0, left: 0 },
        direction_targets: { front: 10, right: 10, left: 10 },
        samples_collected: 0,
        target_samples: 30,
        last_rejection_reason: "NO_FACE",
        last_rejection_details: "No face detected in camera frame"
    });
    if (!domState["enrollment-feedback"].textContent.includes("No face detected")) {
        throw new Error("Enrollment feedback should display rejection details");
    }
    if (!domState["current-instruction"].textContent.includes("Stage 1/3")) {
        throw new Error("Current instruction should indicate stage number and direction prompt");
    }

    updateEnrollmentProgressUI({
        current_direction: "RIGHT",
        direction_counts: { front: 10, right: 2, left: 0 },
        direction_targets: { front: 10, right: 10, left: 10 },
        samples_collected: 12,
        target_samples: 30,
        last_rejection_reason: null,
        last_rejection_details: null
    });
    if (!domState["enrollment-feedback"].textContent.includes("Sample accepted (12/30)")) {
        throw new Error("Enrollment feedback should display sample acceptance when accepted");
    }
    if (!domState["current-instruction"].textContent.includes("Stage 2/3")) {
        throw new Error("Current instruction should indicate Stage 2/3 for RIGHT");
    }
    console.log("✓ updateEnrollmentProgressUI correctly displays rejection feedback and stage instructions");

    // Test Issue 2: loadRegisteredPersons populates table without exposing embeddings
    global.fetchArgs = [];
    global.fetchResponses = [
        {
            status: 200,
            json: [
                { person_id: "p-apurbo", display_name: "Apurbo", preferred_name: "Apurbo B." },
                { person_id: "p-jane", display_name: "Jane Doe", preferred_name: null }
            ]
        }
    ];
    domState["people-tbody"].children = [];
    await loadRegisteredPersons();
    if (global.fetchArgs[0].url !== "/persons" || global.fetchArgs[0].options.headers.Authorization !== "Bearer test-token") {
        throw new Error("loadRegisteredPersons must query /persons with Authorization header");
    }
    if (domState["people-tbody"].children.length !== 2) {
        throw new Error("loadRegisteredPersons should create rows for each registered person");
    }
    const row0 = domState["people-tbody"].children[0];
    const row0Text = row0.children.map(c => c.textContent).join(" ");
    if (!row0Text.includes("Apurbo") || !row0Text.includes("p-apurbo")) {
        throw new Error("Table row must contain person name and person ID");
    }
    if (row0Text.includes("embedding")) {
        throw new Error("Table row must not expose biometric embeddings");
    }
    console.log("✓ loadRegisteredPersons queries /persons and displays person profiles safely");

    // Test Issue 2: confirmAndDeletePerson requires confirmation and calls DELETE /persons/{id}
    global.window.confirm = () => false; // User cancels confirmation
    global.fetchArgs = [];
    await confirmAndDeletePerson("p-jane", "Jane Doe");
    if (global.fetchArgs.length !== 0) {
        throw new Error("Deletion must NOT proceed when confirmation is cancelled");
    }

    global.window.confirm = () => true; // User confirms
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 204 }, // DELETE /persons/p-jane
        { status: 200, json: [{ person_id: "p-apurbo", display_name: "Apurbo" }] } // reload GET /persons
    ];
    await confirmAndDeletePerson("p-jane", "Jane Doe");
    if (global.fetchArgs.length < 2) {
        throw new Error("confirmAndDeletePerson must issue DELETE and reload table");
    }
    if (global.fetchArgs[0].url !== "/persons/p-jane" || global.fetchArgs[0].options.method !== "DELETE") {
        throw new Error("confirmAndDeletePerson must send DELETE /persons/{personId}");
    }
    console.log("✓ confirmAndDeletePerson asks for confirmation and deletes via DELETE /persons/{id}");

    console.log("All frontend logic tests passed successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
