const fs = require('fs');

// Read the script content from demo.html
const html = fs.readFileSync('src/dh360_hi/api/static/demo.html', 'utf8');
const scriptMatch = html.match(/<script>([\s\S]*?)<\/script>/);
if (!scriptMatch) throw new Error("Script not found");
let scriptCode = scriptMatch[1];

// Extract just the functions we want to test to avoid DOM dependency hell
const cancelMatch = scriptCode.match(/async function cancelActiveSession\(token\) \{[\s\S]*?\n    \}/);
const regMatch = scriptCode.match(/async function showRegistrationForm\(\) \{[\s\S]*?\n    \}/);

if (!cancelMatch || !regMatch) throw new Error("Functions not found");

// Setup a mock environment
global.abortController = null;
global.isConnecting = true;
global.fetchResponses = [];
global.fetch = async (url, options) => {
    const res = global.fetchResponses.shift();
    if (res.error) throw new Error(res.error);
    return {
        status: res.status,
        json: async () => res.json || {}
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
    "show-register-btn": { style: { display: 'block' } }
};

global.document = {
    getElementById: (id) => domState[id] || { style: {}, textContent: "" }
};
global.appendLog = () => {};
global.setSseIndicator = () => {};
global.clearContextUI = () => {};
global.clearNotice = () => {};
global.showNotice = (msg, type) => { global.lastNotice = msg; };

// Evaluate the isolated functions
eval(cancelMatch[0]);
eval(regMatch[0]);

async function runTests() {
    console.log("Running frontend logic tests...");

    // Test 1: Successful cancellation permits registration
    global.fetchResponses = [{ status: 200 }];
    global.abortController = { abort: () => { global.abortController.aborted = true; } };
    domState['registration-section'].style.display = 'none';
    
    await showRegistrationForm();
    
    if (domState['registration-section'].style.display !== 'block') {
        throw new Error("Test 1 Failed: Registration form should be visible on 200 OK");
    }
    if (!global.abortController.aborted) {
        throw new Error("Test 1 Failed: abortController was not aborted");
    }
    console.log("✓ Test 1: Successful cancellation permits registration");

    // Test 2: Failed cancellation (500) blocks registration
    global.fetchResponses = [{ status: 500 }];
    domState['registration-section'].style.display = 'none';
    global.lastNotice = null;

    await showRegistrationForm();

    if (domState['registration-section'].style.display === 'block') {
        throw new Error("Test 2 Failed: Registration form should be hidden on 500 Error");
    }
    if (!global.lastNotice.includes("Registration is blocked")) {
        throw new Error("Test 2 Failed: Did not show correct error notice");
    }
    console.log("✓ Test 2: Failed cancellation (500) blocks registration");

    // Test 3: Failed cancellation (409 conflict, actual session running) blocks registration
    global.fetchResponses = [{ status: 409, json: { detail: "Another session holds lock" } }];
    domState['registration-section'].style.display = 'none';
    global.lastNotice = null;

    await showRegistrationForm();

    if (domState['registration-section'].style.display === 'block') {
        throw new Error("Test 3 Failed: Registration form should be hidden on actual 409 Conflict");
    }
    console.log("✓ Test 3: Failed cancellation (actual 409 conflict) blocks registration");

    // Test 4: Missing token blocks registration entirely
    domState['token-input'].value = "   "; // Empty token
    domState['registration-section'].style.display = 'none';
    global.lastNotice = null;

    await showRegistrationForm();

    if (domState['registration-section'].style.display === 'block') {
        throw new Error("Test 4 Failed: Registration form should be hidden when token is missing");
    }
    if (!global.lastNotice.includes("Bearer token with required scopes to register")) {
        throw new Error("Test 4 Failed: Did not show missing token error");
    }
    console.log("✓ Test 4: Missing token blocks registration entirely and skips cancellation");

    // Restore token for next test
    domState['token-input'].value = "test-token";

    // Test 5: "No active session" (409) is treated as success and permits registration
    global.fetchResponses = [{ status: 409, json: { detail: "No active authorization session to cancel" } }];
    domState['registration-section'].style.display = 'none';
    
    await showRegistrationForm();
    
    if (domState['registration-section'].style.display !== 'block') {
        throw new Error("Test 5 Failed: Registration form should be visible when no active session exists");
    }
    console.log("✓ Test 5: 'No active session' 409 permits registration");

    console.log("All frontend logic tests passed successfully!");
}

runTests().catch(err => {
    console.error(err);
    process.exit(1);
});
