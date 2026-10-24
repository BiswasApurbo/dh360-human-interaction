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

// Expose registeredPersonsCount
scriptCode = scriptCode.replace(/let registeredPersonsCount = null;/, 'global.registeredPersonsCount = null;');

// Expose enrolledIdentitiesCount
scriptCode = scriptCode.replace(/let enrolledIdentitiesCount = null;/, 'global.enrolledIdentitiesCount = null;');

// Expose isNewRegistration
scriptCode = scriptCode.replace(/let isNewRegistration = false;/, 'global.isNewRegistration = false;');
scriptCode = scriptCode.replace(/isNewRegistration = true;/g, 'global.isNewRegistration = true;');
scriptCode = scriptCode.replace(/isNewRegistration = false;/g, 'global.isNewRegistration = false;');
scriptCode = scriptCode.replace(/if \(isNewRegistration/g, 'if (global.isNewRegistration');

// Expose isWorking
scriptCode = scriptCode.replace(/let isWorking = false;/, 'global.isWorking = false;');
scriptCode = scriptCode.replace(/isWorking = true;/g, 'global.isWorking = true;');
scriptCode = scriptCode.replace(/isWorking = false;/g, 'global.isWorking = false;');
scriptCode = scriptCode.replace(/if \(isWorking\)/g, 'if (global.isWorking)');

// Expose enrollmentIntervalId
scriptCode = scriptCode.replace(/let enrollmentIntervalId = null;/, 'global.enrollmentIntervalId = null;');
scriptCode = scriptCode.replace(/enrollmentIntervalId = /g, 'global.enrollmentIntervalId = ');
scriptCode = scriptCode.replace(/if \(enrollmentIntervalId/g, 'if (global.enrollmentIntervalId');

// Expose isStepInProgress
scriptCode = scriptCode.replace(/let isStepInProgress = false;/, 'global.isStepInProgress = false;');
scriptCode = scriptCode.replace(/isStepInProgress = true;/g, 'global.isStepInProgress = true;');
scriptCode = scriptCode.replace(/isStepInProgress = false;/g, 'global.isStepInProgress = false;');
scriptCode = scriptCode.replace(/if \(isStepInProgress\)/g, 'if (global.isStepInProgress)');

// Expose currentPersonId
scriptCode = scriptCode.replace(/let currentPersonId = null;/, 'global.currentPersonId = null;');
scriptCode = scriptCode.replace(/currentPersonId = /g, 'global.currentPersonId = ');
scriptCode = scriptCode.replace(/currentPersonId ===/g, 'global.currentPersonId ===');

global.abortController = null;
global.isConnecting = false;
global.registeredPersonsCount = null;
global.enrolledIdentitiesCount = null;
global.isNewRegistration = false;
global.isWorking = false;
global.enrollmentIntervalId = null;
global.isStepInProgress = false;
global.currentPersonId = null;
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
    "enrollment-section": { style: { display: 'none' } },
    "auth-action-section": { style: { display: 'block' } },
    "show-register-btn": { style: { display: 'block' }, disabled: false },
    "start-btn": { disabled: false },
    "display-name": { value: "Test" },
    "preferred-name": { value: "" },
    "role": { value: "" },
    "department": { value: "" },
    "office-room": { value: "" },
    "working-hours": { value: "" },
    "profile-description": { value: "" },
    "register-btn": { disabled: false, title: "" },
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
    "notice-box": { style: {}, textContent: "", className: "" },
    "hud-stage-instruction": { textContent: "" },
    "hud-feedback-text": { textContent: "", style: {} },
    "stage-card-front": { className: "stage-card" },
    "stage-card-right": { className: "stage-card" },
    "stage-card-left": { className: "stage-card" },
    "stage-count-front": { textContent: "" },
    "stage-count-right": { textContent: "" },
    "stage-count-left": { textContent: "" },
    "stage-bar-front": { style: {} },
    "stage-bar-right": { style: {} },
    "stage-bar-left": { style: {} },
    "hud-total-samples": { textContent: "" },
    "hud-arrow-right": { style: { display: "none" } },
    "hud-arrow-left": { style: { display: "none" } },
    "hud-front-indicator": { style: { display: "inline" } },
    "hud-wireframe-mesh": { style: {} },
    "hud-face-guide-group": {
        attributes: {},
        setAttribute(k, v) { this.attributes[k] = v; },
        getAttribute(k) { return this.attributes[k]; }
    },
    "hud-target-reticle": {
        attributes: {},
        setAttribute(k, v) { this.attributes[k] = v; },
        getAttribute(k) { return this.attributes[k]; }
    },
    "hud-camera-badge": { textContent: "", className: "hud-badge" },
    "hud-camera-toggle-btn": { textContent: "" },
    "cancel-enrollment-btn": { disabled: false },
    "cancel-enrollment-top-btn": { disabled: false },
    "enrollment-camera-preview": { style: {}, src: "" },
    "enrollment-local-video": { style: {}, srcObject: null }
};

global.document = {
    getElementById: (id) => domState[id] || {
        style: {},
        textContent: "",
        attributes: {},
        setAttribute(k, v) { this.attributes[k] = v; },
        getAttribute(k) { return this.attributes[k]; }
    },
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

    // Test empty gallery: showRegistrationForm skips preflight check directly to form
    global.registeredPersonsCount = 0;
    domState["registration-section"].style.display = 'none';
    domState["show-register-btn"].style.display = 'block';
    domState["show-register-btn"].disabled = false;
    global.fetchResponses = [
        { status: 200 }, // DELETE /authorization/sessions/current
        { status: 409, json: { detail: "No active enrollment session" } } // DELETE /persons/enrollment/sessions/current
    ];
    await showRegistrationForm();
    if (global.isPreflightMode) throw new Error("Should NOT enter preflight mode when gallery is empty");
    if (domState["registration-section"].style.display !== 'block') throw new Error("Registration section should be shown immediately when gallery is empty");
    if (domState["show-register-btn"].style.display !== 'none') throw new Error("Show register button should be hidden when form is open");
    console.log("✓ Empty gallery skips duplicate preflight and immediately displays registration form");

    // Test empty gallery: startAuthorization blocks and notifies operator
    global.registeredPersonsCount = 0;
    domState["notice-box"].textContent = "";
    domState["start-btn"].disabled = false;
    global.fetchResponses = [
        { status: 200 }, // DELETE /authorization/sessions/current
        { status: 409, json: { detail: "No active enrollment session" } } // DELETE /persons/enrollment/sessions/current
    ];
    await startAuthorization();
    if (!domState["notice-box"].textContent.includes("No registered people found")) {
        throw new Error("startAuthorization should notify operator when gallery is empty");
    }
    if (domState["start-btn"].disabled) throw new Error("Start button should remain enabled");
    console.log("✓ Empty gallery blocks startAuthorization and prompts operator to register first");

    // Test non-empty gallery: preflight duplicate detection runs
    global.registeredPersonsCount = 1;
    domState["registration-section"].style.display = 'none';
    domState["show-register-btn"].style.display = 'block';
    domState["show-register-btn"].disabled = false;
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

    global.registeredPersonsCount = 1;
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

    global.isPreflightMode = false;
    domState["notice-box"].textContent = "";
    await handleSseBlock('event: authorization_updated\ndata: {"status": "TIMEOUT"}');
    if (!domState["notice-box"].textContent.includes("Authorization timed out")) {
        throw new Error("Normal authorization TIMEOUT should show retryable timeout notice");
    }
    console.log("✓ Normal authorization TIMEOUT shows clear retryable message");

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

    // Test empty recognition gallery via health check: 0 enrolled identities blocks startAuthorization even if registeredPersonsCount === 1
    global.enrolledIdentitiesCount = 0;
    global.registeredPersonsCount = 1;
    domState["notice-box"].textContent = "";
    domState["start-btn"].disabled = false;
    global.fetchResponses = [
        { status: 200 }, // DELETE /authorization/sessions/current
        { status: 409, json: { detail: "No active enrollment session" } } // DELETE /persons/enrollment/sessions/current
    ];
    await startAuthorization();
    if (!domState["notice-box"].textContent.includes("No registered people found")) {
        throw new Error("startAuthorization should block and notify when recognition gallery has 0 enrolled identities");
    }
    console.log("✓ Empty recognition gallery (0 enrolled identities) blocks startAuthorization");

    // Test empty recognition gallery via health check skips preflight
    global.enrolledIdentitiesCount = 0;
    global.registeredPersonsCount = 1;
    domState["registration-section"].style.display = 'none';
    domState["show-register-btn"].style.display = 'block';
    domState["show-register-btn"].disabled = false;
    global.fetchResponses = [
        { status: 200 }, // DELETE /authorization/sessions/current
        { status: 409, json: { detail: "No active enrollment session" } } // DELETE /persons/enrollment/sessions/current
    ];
    await showRegistrationForm();
    if (global.isPreflightMode) throw new Error("Should NOT enter preflight mode when gallery has 0 enrolled identities");
    if (domState["registration-section"].style.display !== 'block') throw new Error("Registration section should be shown immediately when gallery has 0 enrolled identities");
    console.log("✓ Empty recognition gallery skips preflight duplicate check and displays registration form");

    // Test polling step failure (state === 'FAILED') stops polling and rolls back new registration
    global.isNewRegistration = true;
    pollEnrollmentStep("test-token", "failed-person-999");
    global.fetchResponses = [
        { status: 200, json: { state: "FAILED", last_rejection_reason: "timeout", last_rejection_details: "Enrollment timed out" } },
        { status: 200 }, // DELETE /persons/failed-person-999
        { status: 200, json: [] } // loadRegisteredPersons
    ];
    global.fetchArgs = [];
    await global._latestIntervalFn();
    const rollbackCall = global.fetchArgs.find(a => a.options && a.options.method === "DELETE" && a.url.includes("failed-person-999"));
    if (!rollbackCall) throw new Error("pollEnrollmentStep MUST roll back newly registered person on FAILED state");
    if (!domState["notice-box"].textContent.includes("Enrollment ended")) throw new Error("Failure notice must be shown");
    console.log("✓ Enrollment step FAILED state cleanly stops polling and rolls back incomplete person");

    // Test startEnrollmentForExistingPerson initiates enrollment without rollback flag
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 }, // cancel active auth
        { status: 200 }, // cancel active enroll
        { status: 200 }, // PATCH consent
        { status: 201, json: {} } // POST enrollment session
    ];
    await startEnrollmentForExistingPerson("existing-p1", "Existing Person");
    if (global.isNewRegistration) throw new Error("Existing person enrollment MUST NOT set isNewRegistration");
    console.log("✓ startEnrollmentForExistingPerson safely enrolls existing person without new registration rollback flag");

    // Test Bug 2: updateRegisterButtonState enforces validation and reactive state
    global.enrollmentIntervalId = null;
    global.isStepInProgress = false;
    global.isWorking = false;
    global.isPreflightMode = false;
    domState["token-input"].value = "";
    domState["display-name"].value = "Apurbo";
    updateRegisterButtonState();
    if (!domState["register-btn"].disabled) throw new Error("Register button must be disabled when token is empty");
    if (!domState["register-btn"].title.includes("token")) throw new Error("Button title must state token required");

    domState["token-input"].value = "valid-token";
    domState["display-name"].value = "";
    updateRegisterButtonState();
    if (!domState["register-btn"].disabled) throw new Error("Register button must be disabled when display name is empty");
    if (!domState["register-btn"].title.includes("Display Name")) throw new Error("Button title must state Display Name required");

    domState["display-name"].value = "Apurbo";
    updateRegisterButtonState();
    if (domState["register-btn"].disabled) throw new Error("Register button must be enabled when token and display name are valid");

    global.isWorking = true;
    updateRegisterButtonState();
    if (!domState["register-btn"].disabled) throw new Error("Register button must be disabled when operation is in progress");
    global.isWorking = false;
    updateRegisterButtonState();
    if (domState["register-btn"].disabled) throw new Error("Register button must recover and be enabled after operation completes");
    console.log("✓ updateRegisterButtonState reactively validates token, display name, and operation state");

    // Test cancelRegistrationForm hides form and restores buttons
    domState["registration-section"].style.display = "block";
    domState["show-register-btn"].style.display = "none";
    cancelRegistrationForm();
    if (domState["registration-section"].style.display !== "none") throw new Error("cancelRegistrationForm must hide registration section");
    if (domState["show-register-btn"].style.display !== "inline-block") throw new Error("cancelRegistrationForm must restore show register button");
    console.log("✓ cancelRegistrationForm cleanly closes registration form");

    // Test Bug 1: confirmAndDeletePerson cleans up active authorization session if matching deleted person
    domState["auth-person"].textContent = "Apurbo";
    global.window.confirm = () => true;
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 204 }, // DELETE /persons/p-apurbo
        { status: 200 }, // DELETE /authorization/sessions/current (cleanup active session)
        { status: 200, json: [] } // loadRegisteredPersons
    ];
    await confirmAndDeletePerson("p-apurbo", "Apurbo");
    const authCancelCall = global.fetchArgs.find(a => a.options && a.options.method === "DELETE" && a.url === "/authorization/sessions/current");
    if (!authCancelCall) throw new Error("confirmAndDeletePerson must cancel active session when deleted person matches active session");
    console.log("✓ confirmAndDeletePerson invalidates active authorization session when matching person is deleted");

    // Test confirmAndDeletePerson does NOT cancel active authorization session for an unrelated person
    domState["auth-person"].textContent = "Apurbo";
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 204 }, // DELETE /persons/p-unrelated
        { status: 200, json: [] } // loadRegisteredPersons
    ];
    await confirmAndDeletePerson("p-unrelated", "Unrelated Person");
    const unrelatedCancelCall = global.fetchArgs.find(a => a.options && a.options.method === "DELETE" && a.url === "/authorization/sessions/current");
    if (unrelatedCancelCall) throw new Error("confirmAndDeletePerson must NOT cancel session when deleting unrelated person");
    console.log("✓ confirmAndDeletePerson preserves active session when deleting unrelated person");

    // Test HUD Stage 1 (FRONT): verifies stage cards, arrows, and instruction banner
    updateEnrollmentProgressUI({
        current_direction: "FRONT",
        direction_counts: { front: 5, right: 0, left: 0 },
        direction_targets: { front: 10, right: 10, left: 10 },
        samples_collected: 5,
        target_samples: 30,
        face_detected: true,
        face_bbox: [200, 150, 100, 100]
    });
    if (!domState["hud-stage-instruction"].textContent.includes("STAGE 1/3: LOOK FRONT (5/10)")) {
        throw new Error("HUD stage banner must show FRONT stage and counts");
    }
    if (domState["stage-card-front"].className !== "stage-card active") {
        throw new Error("Front card must be active");
    }
    if (domState["hud-front-indicator"].style.display !== "inline") {
        throw new Error("Front indicator must be displayed during FRONT stage");
    }
    if (domState["hud-arrow-right"].style.display !== "none" || domState["hud-arrow-left"].style.display !== "none") {
        throw new Error("Directional arrows must be hidden during FRONT stage");
    }
    if (domState["hud-face-guide-group"].getAttribute("transform") !== "translate(-70, -20)") {
        throw new Error("Face guide group must translate to tracked face center");
    }
    if (domState["hud-target-reticle"].getAttribute("stroke") !== "#38bdf8") {
        throw new Error("Target reticle must show locked cyan stroke when face is detected");
    }
    console.log("✓ HUD correctly renders FRONT stage, tracking reticle, and card status");

    // Test HUD Stage 2 (RIGHT): verifies right arrows (>>>), active right card, completed front card
    updateEnrollmentProgressUI({
        current_direction: "RIGHT",
        direction_counts: { front: 10, right: 4, left: 0 },
        direction_targets: { front: 10, right: 10, left: 10 },
        samples_collected: 14,
        target_samples: 30,
        face_detected: true,
        face_bbox: [320, 220, 100, 100]
    });
    if (!domState["hud-stage-instruction"].textContent.includes("STAGE 2/3: TURN HEAD TO YOUR RIGHT >>>")) {
        throw new Error("HUD stage banner must show RIGHT turn instruction with chevrons");
    }
    if (domState["stage-card-front"].className !== "stage-card complete") {
        throw new Error("Front card must be marked complete in Stage 2");
    }
    if (domState["stage-card-right"].className !== "stage-card active") {
        throw new Error("Right card must be active in Stage 2");
    }
    if (domState["hud-arrow-right"].style.display !== "inline") {
        throw new Error("Right arrow group must be displayed during RIGHT stage");
    }
    if (domState["hud-arrow-left"].style.display !== "none") {
        throw new Error("Left arrow group must be hidden during RIGHT stage");
    }
    if (domState["stage-bar-right"].style.width !== "40%") {
        throw new Error("Right stage progress bar must reflect 4/10 (40%)");
    }
    console.log("✓ HUD correctly renders RIGHT stage with rightward guide chevrons and progress");

    // Test HUD Stage 3 (LEFT): verifies left arrows (<<<), completed right card, and no face state
    updateEnrollmentProgressUI({
        current_direction: "LEFT",
        direction_counts: { front: 10, right: 10, left: 7 },
        direction_targets: { front: 10, right: 10, left: 10 },
        samples_collected: 27,
        target_samples: 30,
        face_detected: false,
        face_bbox: null,
        last_rejection_reason: "NO_FACE"
    });
    if (!domState["hud-stage-instruction"].textContent.includes("STAGE 3/3: <<< TURN HEAD TO YOUR LEFT")) {
        throw new Error("HUD stage banner must show LEFT turn instruction with chevrons");
    }
    if (domState["stage-card-right"].className !== "stage-card complete") {
        throw new Error("Right card must be marked complete in Stage 3");
    }
    if (domState["stage-card-left"].className !== "stage-card active") {
        throw new Error("Left card must be active in Stage 3");
    }
    if (domState["hud-arrow-left"].style.display !== "inline") {
        throw new Error("Left arrow group must be displayed during LEFT stage");
    }
    if (domState["hud-arrow-right"].style.display !== "none") {
        throw new Error("Right arrow group must be hidden during LEFT stage");
    }
    if (domState["hud-face-guide-group"].getAttribute("transform") !== "translate(0, 0)") {
        throw new Error("Face guide group must reset to center when no face is detected");
    }
    if (domState["hud-target-reticle"].getAttribute("stroke") !== "#f59e0b") {
        throw new Error("Target reticle must show searching amber stroke when no face is detected");
    }
    console.log("✓ HUD correctly renders LEFT stage with leftward guide chevrons and searching reticle");

    // Test handleCancelEnrollment cleanly cancels active session and cleans up UI
    global.currentPersonId = "p-cancel-test";
    global.isNewRegistration = true;
    domState["enrollment-section"].style.display = "block";
    domState["auth-action-section"].style.display = "none";
    global.fetchArgs = [];
    global.fetchResponses = [
        { status: 200 }, // DELETE /persons/enrollment/sessions/current
        { status: 204 }, // DELETE /persons/p-cancel-test (rollback newly created person)
        { status: 200, json: [] } // loadRegisteredPersons
    ];
    await handleCancelEnrollment();
    const sessionCancelReq = global.fetchArgs.find(a => a.url === "/persons/enrollment/sessions/current" && a.options.method === "DELETE");
    if (!sessionCancelReq) throw new Error("handleCancelEnrollment must send DELETE to /persons/enrollment/sessions/current");
    const personRollbackReq = global.fetchArgs.find(a => a.url === "/persons/p-cancel-test" && a.options.method === "DELETE");
    if (!personRollbackReq) throw new Error("handleCancelEnrollment must roll back newly registered person");
    if (domState["enrollment-section"].style.display !== "none") throw new Error("Enrollment section must be hidden after cancel");
    if (domState["auth-action-section"].style.display !== "block") throw new Error("Auth action section must be restored after cancel");
    console.log("✓ handleCancelEnrollment safely cancels session, rolls back new profile, and restores UI");

    console.log("All frontend logic tests passed successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
