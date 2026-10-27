/**
 * Comprehensive Automated Security Test Suite for DH360 Next.js BFF.
 *
 * Verifies:
 * 1. Missing or invalid Firebase token returns HTTP 401.
 * 2. Disabled user is rejected with HTTP 401.
 * 3. Emulator authentication is explicitly separated from production (blocked in NODE_ENV=production).
 * 4. Authenticated user with insufficient privileges receives HTTP 403.
 * 5. Unknown path (404) or unsupported method (405) is rejected.
 * 6. An allowed request reaches the configured DH360 upstream.
 * 7. Upstream receives server-side operator token; browser never receives it.
 * 8. Malformed request data or unapproved content-type is rejected safely.
 * 9. Upstream unavailable or timeout responses handled without leaking secrets.
 * 10. Query strings, Accept header, and Last-Event-ID are forwarded correctly.
 * 11. Destructive operations (delete person, biometrics reset, consent update) strictly gated to superAdmin.
 * 12. Streaming (SSE) and binary (JPEG) response pipes function correctly without buffering.
 */

import http from "node:http";
import assert from "node:assert";
import { handleDh360ProxyRequest } from "../src/lib/dh360/proxy.ts";
import { resetConfigCacheForTesting } from "../src/lib/dh360/config.ts";

function createMockJwt(payloadObj) {
  const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(JSON.stringify(payloadObj)).toString("base64url");
  return `${header}.${payload}.mockSignature`;
}

const nowSec = Math.floor(Date.now() / 1000);

const SUPER_ADMIN_TOKEN = createMockJwt({
  sub: "super-123",
  email: "super@kbcon.local",
  role: "superAdmin",
  exp: nowSec + 3600,
});

const ADMIN_TOKEN = createMockJwt({
  sub: "admin-456",
  email: "admin@kbcon.local",
  role: "admin",
  exp: nowSec + 3600,
});

const DISABLED_USER_TOKEN = createMockJwt({
  sub: "disabled-789",
  email: "disabled@kbcon.local",
  role: "admin",
  exp: nowSec + 3600,
});

const UNPRIVILEGED_TOKEN = createMockJwt({
  sub: "user-789",
  email: "user@kbcon.local",
  role: "viewer", // Not admin or superAdmin
  exp: nowSec + 3600,
});

const EXPIRED_TOKEN = createMockJwt({
  sub: "admin-456",
  email: "admin@kbcon.local",
  role: "admin",
  exp: nowSec - 3600, // Expired
});

const FORGED_TOKEN = createMockJwt({
  sub: "attacker-999",
  email: "attacker@evil.com",
  role: "superAdmin",
  exp: nowSec + 3600,
});

const TEST_OPERATOR_TOKEN = "test-secret-operator-token-xyz-987";

let mockServer;
let mockPort;
let lastDh360Request = null;

async function startMockServer() {
  return new Promise((resolve) => {
    mockServer = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        const recorded = {
          method: req.method,
          url: req.url,
          headers: req.headers,
          body,
        };

        // Mock Firebase Auth Emulator lookup endpoint
        if (req.url.startsWith("/identitytoolkit.googleapis.com/v1/accounts:lookup")) {
          try {
            const parsed = JSON.parse(body);
            const token = parsed.idToken;

            if (token === SUPER_ADMIN_TOKEN) {
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  users: [
                    {
                      localId: "super-123",
                      email: "super@kbcon.local",
                      disabled: false,
                      customAttributes: JSON.stringify({ role: "superAdmin" }),
                    },
                  ],
                })
              );
              return;
            }

            if (token === ADMIN_TOKEN) {
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  users: [
                    {
                      localId: "admin-456",
                      email: "admin@kbcon.local",
                      disabled: false,
                      customAttributes: JSON.stringify({ role: "admin" }),
                    },
                  ],
                })
              );
              return;
            }

            if (token === DISABLED_USER_TOKEN) {
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  users: [
                    {
                      localId: "disabled-789",
                      email: "disabled@kbcon.local",
                      disabled: true, // Account disabled
                      customAttributes: JSON.stringify({ role: "admin" }),
                    },
                  ],
                })
              );
              return;
            }

            if (token === UNPRIVILEGED_TOKEN) {
              res.writeHead(200, { "Content-Type": "application/json" });
              res.end(
                JSON.stringify({
                  users: [
                    {
                      localId: "user-789",
                      email: "user@kbcon.local",
                      disabled: false,
                      customAttributes: JSON.stringify({ role: "viewer" }),
                    },
                  ],
                })
              );
              return;
            }

            // Forged or unknown token
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: "INVALID_ID_TOKEN" } }));
            return;
          } catch {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ error: { message: "BAD_REQUEST" } }));
            return;
          }
        }

        // Mock DH360 FastAPI Endpoints
        lastDh360Request = recorded;
        if (req.headers["authorization"] !== `Bearer ${TEST_OPERATOR_TOKEN}`) {
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ detail: "Invalid operator credentials" }));
          return;
        }

        if (req.url === "/health") {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ status: "healthy", mode: "local-test" }));
        } else if (req.url.includes("/preview")) {
          res.writeHead(200, {
            "Content-Type": "image/jpeg",
            "Cache-Control": "no-store",
          });
          res.end(Buffer.from([0xff, 0xd8, 0xff, 0xe0]));
        } else if (req.url === "/interaction/events") {
          res.writeHead(200, {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
          });
          res.write("event: heartbeat\ndata: {}\n\n");
          res.end();
        } else if (req.url.startsWith("/sessions")) {
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(JSON.stringify([{ session_id: "s-1", count: 1 }]));
        } else if (req.url.startsWith("/persons")) {
          if (req.method === "DELETE") {
            res.writeHead(204);
            res.end();
          } else if (req.method === "PATCH") {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify({ updated: true }));
          } else {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify([{ person_id: "p-1" }]));
          }
        } else {
          res.writeHead(404, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "NOT_FOUND" }));
        }
      });
    });

    mockServer.listen(0, "127.0.0.1", () => {
      mockPort = mockServer.address().port;
      process.env.DH360_API_URL = `http://127.0.0.1:${mockPort}`;
      process.env.DH360_OPERATOR_TOKEN = TEST_OPERATOR_TOKEN;
      process.env.FIREBASE_AUTH_EMULATOR_HOST = `127.0.0.1:${mockPort}`;
      process.env.NEXT_PUBLIC_USE_EMULATOR = "true";
      process.env.NODE_ENV = "development";
      resetConfigCacheForTesting();
      resolve();
    });
  });
}

async function stopMockServer() {
  return new Promise((resolve) => {
    if (mockServer) {
      mockServer.close(resolve);
    } else {
      resolve();
    }
  });
}

async function runTests() {
  console.log("Starting DH360 BFF Security Verification Suite...\n");
  await startMockServer();

  let passed = 0;
  let failed = 0;

  async function test(name, fn) {
    try {
      await fn();
      console.log(`✓ ${name}`);
      passed++;
    } catch (err) {
      console.error(`✗ ${name}`);
      console.error(err);
      failed++;
    }
  }

  // ---------------------------------------------------------------------------
  // 1. Authentication & Token Verification Gate
  // ---------------------------------------------------------------------------
  await test("1.1 Missing Authorization header returns HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", { method: "GET" });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
    const body = await res.json();
    assert.strictEqual(body.error, "UNAUTHENTICATED");
  });

  await test("1.2 Non-Bearer scheme returns HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: "Basic dXNlcjpwYXNz" },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
  });

  await test("1.3 Malformed JWT string returns HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: "Bearer malformed.jwt" },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
  });

  await test("1.4 Expired token returns HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${EXPIRED_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
  });

  await test("1.5 Disabled user account is rejected with HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${DISABLED_USER_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
  });

  await test("1.6 Forged unverified token in emulator mode is rejected with HTTP 401", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${FORGED_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401);
  });

  await test("1.7 Emulator authentication is strictly blocked in production mode (NODE_ENV=production)", async () => {
    process.env.NODE_ENV = "production";
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401); // Rejects emulator shortcut in production
    process.env.NODE_ENV = "development";
  });

  await test("1.8 Emulator authentication is disabled when NEXT_PUBLIC_USE_EMULATOR=false", async () => {
    process.env.NEXT_PUBLIC_USE_EMULATOR = "false";
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 401); // Rejects emulator lookup when NEXT_PUBLIC_USE_EMULATOR is false
    process.env.NEXT_PUBLIC_USE_EMULATOR = "true";
  });

  // ---------------------------------------------------------------------------
  // 2. Privilege Boundaries & RBAC
  // ---------------------------------------------------------------------------
  await test("2.1 Unprivileged role ('viewer') is rejected fail-closed with HTTP 403", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${UNPRIVILEGED_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 403);
    const body = await res.json();
    assert.strictEqual(body.error, "FORBIDDEN");
  });

  await test("2.2 Admin cannot reset biometrics (Super Admin required) -> HTTP 403", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100/biometrics", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100", "biometrics"]);
    assert.strictEqual(res.status, 403);
    const body = await res.json();
    assert.strictEqual(body.error, "FORBIDDEN");
  });

  await test("2.3 Admin cannot delete person profile (Super Admin required) -> HTTP 403", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100"]);
    assert.strictEqual(res.status, 403);
    const body = await res.json();
    assert.strictEqual(body.error, "FORBIDDEN");
  });

  await test("2.4 Admin cannot modify consent_status (Super Admin required) -> HTTP 403", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100", {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ consent_status: "granted" }),
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100"]);
    assert.strictEqual(res.status, 403);
    const body = await res.json();
    assert.strictEqual(body.error, "FORBIDDEN");
  });

  await test("2.5 Super Admin CAN modify consent_status -> HTTP 200", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100", {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${SUPER_ADMIN_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ consent_status: "granted" }),
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100"]);
    assert.strictEqual(res.status, 200);
  });

  await test("2.6 Super Admin CAN delete person profile -> HTTP 204", async () => {
    lastDh360Request = null;
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100"]);
    assert.strictEqual(res.status, 204);
    assert.strictEqual(lastDh360Request.method, "DELETE");
  });

  // ---------------------------------------------------------------------------
  // 3. Route Allowlist & Method Verification
  // ---------------------------------------------------------------------------
  await test("3.1 Unknown path returns HTTP 404 (Fail Closed)", async () => {
    const req = new Request("http://localhost:3000/api/dh360/unregistered/path", {
      method: "GET",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["unregistered", "path"]);
    assert.strictEqual(res.status, 404);
  });

  await test("3.2 Unsupported method on valid route returns HTTP 405", async () => {
    const req = new Request("http://localhost:3000/api/dh360/health", {
      method: "DELETE",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 405);
  });

  // ---------------------------------------------------------------------------
  // 4. Request Forwarding & Secret Protection
  // ---------------------------------------------------------------------------
  await test("4.1 Upstream receives operator token; browser never receives it", async () => {
    lastDh360Request = null;
    const req = new Request("http://localhost:3000/api/dh360/health", {
      method: "GET",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(lastDh360Request.headers["authorization"], `Bearer ${TEST_OPERATOR_TOKEN}`);

    const resJson = await res.json();
    assert.strictEqual(JSON.stringify(resJson).includes(TEST_OPERATOR_TOKEN), false);
    for (const [, val] of res.headers.entries()) {
      assert.strictEqual(val.includes(TEST_OPERATOR_TOKEN), false);
    }
  });

  await test("4.2 Non-JSON content-type on request body is rejected with HTTP 415", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-100", {
      method: "PATCH",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        "Content-Type": "text/plain",
      },
      body: "consent_status=granted",
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-100"]);
    assert.strictEqual(res.status, 415);
  });

  await test("4.3 Malformed JSON body returns HTTP 400 without forwarding", async () => {
    lastDh360Request = null;
    const req = new Request("http://localhost:3000/api/dh360/persons", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: "{ invalid json ...",
    });
    const res = await handleDh360ProxyRequest(req, ["persons"]);
    assert.strictEqual(res.status, 400);
    assert.strictEqual(lastDh360Request, null);
  });

  await test("4.4 Path traversal attempt with '..' is rejected with HTTP 400", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/../secret", {
      method: "GET",
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "..", "secret"]);
    assert.strictEqual(res.status, 400);
  });

  await test("4.5 Upstream connection offline returns HTTP 503 without leaking secrets", async () => {
    process.env.DH360_API_URL = "http://127.0.0.1:59999";
    resetConfigCacheForTesting();

    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 503);
    const body = await res.json();
    assert.strictEqual(body.error, "UPSTREAM_UNAVAILABLE");
    assert.strictEqual(JSON.stringify(body).includes(TEST_OPERATOR_TOKEN), false);

    process.env.DH360_API_URL = `http://127.0.0.1:${mockPort}`;
    resetConfigCacheForTesting();
  });

  await test("4.6 Query parameters, Accept header, and Last-Event-ID are forwarded", async () => {
    lastDh360Request = null;
    const req = new Request("http://localhost:3000/api/dh360/sessions?person_id=alice&limit=10", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        Accept: "application/json",
        "Last-Event-ID": "evt-888",
      },
    });
    const res = await handleDh360ProxyRequest(req, ["sessions"]);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(lastDh360Request.url, "/sessions?person_id=alice&limit=10");
    assert.strictEqual(lastDh360Request.headers["accept"], "application/json");
    assert.strictEqual(lastDh360Request.headers["last-event-id"], "evt-888");
  });

  await test("4.7 Upstream operator credential rejection maps to HTTP 502 Bad Gateway", async () => {
    process.env.DH360_OPERATOR_TOKEN = "invalid-operator-token";
    resetConfigCacheForTesting();

    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 502);
    const body = await res.json();
    assert.strictEqual(body.error, "UPSTREAM_AUTH_ERROR");
    assert.strictEqual(body.upstreamStatus, 401);

    process.env.DH360_OPERATOR_TOKEN = TEST_OPERATOR_TOKEN;
    resetConfigCacheForTesting();
  });

  await test("4.8 Missing operator token maps to HTTP 503 CONFIGURATION_ERROR", async () => {
    process.env.DH360_OPERATOR_TOKEN = "";
    resetConfigCacheForTesting();

    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 503);
    const body = await res.json();
    assert.strictEqual(body.error, "CONFIGURATION_ERROR");
    assert.strictEqual(body.message.includes("DH360_OPERATOR_TOKEN"), true);

    process.env.DH360_OPERATOR_TOKEN = TEST_OPERATOR_TOKEN;
    resetConfigCacheForTesting();
  });

  await test("4.9 Vercel deployment with localhost DH360_API_URL maps to HTTP 503 CONFIGURATION_ERROR", async () => {
    process.env.VERCEL = "1";
    process.env.DH360_API_URL = "http://127.0.0.1:8000";
    resetConfigCacheForTesting();

    const req = new Request("http://localhost:3000/api/dh360/health", {
      headers: { Authorization: `Bearer ${ADMIN_TOKEN}` },
    });
    const res = await handleDh360ProxyRequest(req, ["health"]);
    assert.strictEqual(res.status, 503);
    const body = await res.json();
    assert.strictEqual(body.error, "CONFIGURATION_ERROR");
    assert.strictEqual(body.message.includes("localhost/127.0.0.1"), true);

    delete process.env.VERCEL;
    process.env.DH360_API_URL = `http://127.0.0.1:${mockPort}`;
    resetConfigCacheForTesting();
  });

  // ---------------------------------------------------------------------------
  // 5. Streaming and Binary Verification
  // ---------------------------------------------------------------------------
  await test("5.1 SSE stream is piped as text/event-stream without buffering", async () => {
    const req = new Request("http://localhost:3000/api/dh360/interaction/events", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        Accept: "text/event-stream",
      },
    });
    const res = await handleDh360ProxyRequest(req, ["interaction", "events"]);
    assert.strictEqual(res.status, 200);
    assert.match(res.headers.get("content-type"), /text\/event-stream/);
    const text = await res.text();
    assert.match(text, /heartbeat/);
  });

  await test("5.2 Binary JPEG preview stream is piped with image/jpeg content-type", async () => {
    const req = new Request("http://localhost:3000/api/dh360/persons/p-1/enrollment/sessions/preview", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${ADMIN_TOKEN}`,
        Accept: "image/jpeg",
      },
    });
    const res = await handleDh360ProxyRequest(req, ["persons", "p-1", "enrollment", "sessions", "preview"]);
    assert.strictEqual(res.status, 200);
    assert.match(res.headers.get("content-type"), /image\/jpeg/);
    const buf = await res.arrayBuffer();
    assert.strictEqual(buf.byteLength, 4);
  });

  await stopMockServer();

  console.log(`\n========================================`);
  console.log(`Security Suite: ${passed} passed, ${failed} failed`);
  console.log(`========================================\n`);

  if (failed > 0) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error("Fatal test error:", err);
  process.exit(1);
});
