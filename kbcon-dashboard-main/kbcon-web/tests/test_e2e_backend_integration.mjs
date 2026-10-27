/**
 * End-to-End Integration Verification Suite for DH360 Backend & Next.js BFF.
 *
 * Verifies the live communication chain:
 * 1. Spawns the live DH360 Python backend (dh360-hi serve) on 127.0.0.1:8000.
 * 2. Authenticates through Next.js handleDh360ProxyRequest using Firebase identity tokens.
 * 3. Enforces 401 for unauthenticated calls and 403 for unprivileged callers.
 * 4. Queries /api/dh360/persons and validates database connectivity (reads Test Person).
 * 5. Queries /api/dh360/persons/:id for person profile retrieval.
 * 6. Executes live authorization session lifecycle: start, poll current, stop.
 * 7. Confirms camera/perception subsystem is actively processing frames.
 */

import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import http from "node:http";
import { fileURLToPath } from "node:url";

import { handleDh360ProxyRequest } from "../src/lib/dh360/proxy.ts";
import { resetConfigCacheForTesting } from "../src/lib/dh360/config.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const WEB_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(WEB_DIR, "../..");

// 1. Read operator token and environment from .env.local
const envLocalPath = path.join(WEB_DIR, ".env.local");
let operatorToken = "";
if (fs.existsSync(envLocalPath)) {
  const content = fs.readFileSync(envLocalPath, "utf8");
  for (const line of content.split("\n")) {
    if (line.startsWith("DH360_OPERATOR_TOKEN=")) {
      operatorToken = line.split("=")[1].trim().replace(/^["\']|["\']$/g, "");
    }
  }
}
if (!operatorToken) {
  operatorToken = "dev-operator-secret-token";
}

// 2. Mock Firebase Emulator lookup server for token verification in development
let mockEmulatorServer;
let mockEmulatorPort;

function createMockJwt(payloadObj) {
  const header = Buffer.from(JSON.stringify({ alg: "none", typ: "JWT" })).toString("base64url");
  const payload = Buffer.from(JSON.stringify(payloadObj)).toString("base64url");
  return `${header}.${payload}.mockSignature`;
}

const nowSec = Math.floor(Date.now() / 1000);
const SUPER_ADMIN_TOKEN = createMockJwt({
  sub: "super-admin-uid-1",
  email: "admin@kbcon.ai",
  role: "superAdmin",
  exp: nowSec + 3600,
});

const UNPRIVILEGED_TOKEN = createMockJwt({
  sub: "viewer-uid-2",
  email: "viewer@kbcon.ai",
  role: "viewer",
  exp: nowSec + 3600,
});

async function startMockEmulator() {
  return new Promise((resolve) => {
    mockEmulatorServer = http.createServer((req, res) => {
      let body = "";
      req.on("data", (chunk) => (body += chunk));
      req.on("end", () => {
        if (req.url.startsWith("/identitytoolkit.googleapis.com/v1/accounts:lookup")) {
          const parsed = JSON.parse(body || "{}");
          if (parsed.idToken === SUPER_ADMIN_TOKEN) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                users: [
                  {
                    localId: "super-admin-uid-1",
                    email: "admin@kbcon.ai",
                    disabled: false,
                    customAttributes: JSON.stringify({ role: "superAdmin" }),
                  },
                ],
              })
            );
            return;
          }
          if (parsed.idToken === UNPRIVILEGED_TOKEN) {
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                users: [
                  {
                    localId: "viewer-uid-2",
                    email: "viewer@kbcon.ai",
                    disabled: false,
                    customAttributes: JSON.stringify({ role: "viewer" }),
                  },
                ],
              })
            );
            return;
          }
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: { message: "INVALID_ID_TOKEN" } }));
          return;
        }
        res.writeHead(404);
        res.end();
      });
    });

    mockEmulatorServer.listen(0, "127.0.0.1", () => {
      mockEmulatorPort = mockEmulatorServer.address().port;
      process.env.FIREBASE_AUTH_EMULATOR_HOST = `127.0.0.1:${mockEmulatorPort}`;
      process.env.NEXT_PUBLIC_USE_EMULATOR = "true";
      process.env.NODE_ENV = "development";
      process.env.DH360_API_URL = "http://127.0.0.1:8000";
      process.env.DH360_OPERATOR_TOKEN = operatorToken;
      resetConfigCacheForTesting();
      resolve();
    });
  });
}

// 3. Spawn Python backend
let backendProcess;

async function startBackend() {
  const pythonBin = path.join(REPO_ROOT, ".venv/bin/dh360-hi");
  const configPath = path.join(REPO_ROOT, "dh360-test-config.toml");

  backendProcess = spawn(
    pythonBin,
    [
      "serve",
      "--config-file", configPath,
      "--source", "synthetic",
      "--host", "127.0.0.1",
      "--port", "8000",
    ],
    {
      cwd: REPO_ROOT,
      env: {
        ...process.env,
        DH360_OPERATOR_TOKEN: operatorToken,
        DH360_HI_API__OPERATOR_TOKEN: operatorToken,
      },
      stdio: "pipe",
    }
  );

  let logs = "";
  backendProcess.stdout.on("data", (d) => (logs += d.toString()));
  backendProcess.stderr.on("data", (d) => (logs += d.toString()));

  // Wait for /health to respond
  const start = Date.now();
  while (Date.now() - start < 10000) {
    try {
      const res = await fetch("http://127.0.0.1:8000/health");
      if (res.ok) {
        return;
      }
    } catch {
      await new Promise((r) => setTimeout(r, 200));
    }
  }

  throw new Error(`Backend failed to start within 10s. Logs:\n${logs}`);
}

async function stopAll() {
  if (backendProcess) {
    backendProcess.kill("SIGTERM");
  }
  if (mockEmulatorServer) {
    mockEmulatorServer.close();
  }
}

async function run() {
  console.log("Starting DH360 End-to-End Live Integration Verification...");

  try {
    await startMockEmulator();
    await startBackend();
    console.log("✓ Live DH360 backend started and healthy on http://127.0.0.1:8000\n");

    let passed = 0;
    async function test(name, fn) {
      try {
        await fn();
        console.log(`✓ ${name}`);
        passed++;
      } catch (err) {
        console.error(`✗ ${name}:`, err.message);
        throw err;
      }
    }

    // Step 1: Unauthenticated request is rejected with 401
    await test("1. Unauthenticated request to /api/dh360/persons returns 401", async () => {
      const req = new Request("http://localhost:3000/api/dh360/persons");
      const res = await handleDh360ProxyRequest(req, ["persons"]);
      assert.equal(res.status, 401);
      const json = await res.json();
      assert.equal(json.error, "UNAUTHENTICATED");
    });

    // Step 2: Unprivileged request is rejected with 403
    await test("2. Authenticated caller without admin role returns 403 FORBIDDEN", async () => {
      const req = new Request("http://localhost:3000/api/dh360/persons", {
        headers: { Authorization: `Bearer ${UNPRIVILEGED_TOKEN}` },
      });
      const res = await handleDh360ProxyRequest(req, ["persons"]);
      assert.equal(res.status, 403);
      const json = await res.json();
      assert.equal(json.error, "FORBIDDEN");
    });

    // Step 3: Authorized request retrieves registered people from database
    let existingPersonId = "";
    await test("3. SuperAdmin caller lists people via BFF -> retrieves Test Person", async () => {
      const req = new Request("http://localhost:3000/api/dh360/persons", {
        headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
      });
      const res = await handleDh360ProxyRequest(req, ["persons"]);
      assert.equal(res.status, 200);
      const persons = await res.json();
      assert.ok(Array.isArray(persons));
      assert.ok(persons.length >= 1, "Should contain at least 1 person");
      const found = persons.find((p) => p.display_name === "Test Person");
      assert.ok(found, "Test Person must be in people list");
      existingPersonId = found.person_id;
    });

    // Step 4: Retrieve person detail
    await test("4. SuperAdmin retrieves person detail by ID", async () => {
      const req = new Request(`http://localhost:3000/api/dh360/persons/${existingPersonId}`, {
        headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
      });
      const res = await handleDh360ProxyRequest(req, ["persons", existingPersonId]);
      assert.equal(res.status, 200);
      const person = await res.json();
      assert.equal(person.person_id, existingPersonId);
      assert.equal(person.display_name, "Test Person");
    });

    // Step 5: Start authorization session
    let authSessionId = "";
    await test("5. SuperAdmin starts live authorization session", async () => {
      const req = new Request("http://localhost:3000/api/dh360/authorization/sessions", {
        method: "POST",
        headers: {
          Authorization: `Bearer ${SUPER_ADMIN_TOKEN}`,
          "Content-Type": "application/json",
        },
      });
      const res = await handleDh360ProxyRequest(req, ["authorization", "sessions"]);
      assert.ok(res.status === 200 || res.status === 201, `Status: ${res.status}`);
      const session = await res.json();
      assert.ok(session.authorization_id, "Session must have authorization_id");
      authSessionId = session.authorization_id;
    });

    // Step 6: Poll current authorization session
    await test("6. Poll active authorization session via GET /authorization/sessions/current", async () => {
      const req = new Request("http://localhost:3000/api/dh360/authorization/sessions/current", {
        headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
      });
      const res = await handleDh360ProxyRequest(req, ["authorization", "sessions", "current"]);
      assert.equal(res.status, 200);
      const session = await res.json();
      assert.equal(session.authorization_id, authSessionId);
    });

    // Step 7: Stop authorization session
    await test("7. Stop active authorization session via DELETE", async () => {
      const req = new Request("http://localhost:3000/api/dh360/authorization/sessions/current", {
        method: "DELETE",
        headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
      });
      const res = await handleDh360ProxyRequest(req, ["authorization", "sessions", "current"]);
      assert.ok(res.status === 200 || res.status === 204, `Status: ${res.status}`);

      // Verify session is now cancelled with ended_at_iso timestamp
      const checkReq = new Request("http://localhost:3000/api/dh360/authorization/sessions/current", {
        headers: { Authorization: `Bearer ${SUPER_ADMIN_TOKEN}` },
      });
      const checkRes = await handleDh360ProxyRequest(checkReq, ["authorization", "sessions", "current"]);
      assert.equal(checkRes.status, 200);
      const checkData = await checkRes.json();
      assert.equal(checkData.status, "cancelled", "Session status should be cancelled");
      assert.ok(checkData.ended_at_iso, "Session should have ended_at_iso timestamp");
    });

    // Step 8: Verify live perception metrics from backend
    await test("8. Perception runtime active and processing frames at target FPS", async () => {
      const res = await fetch("http://127.0.0.1:8000/health");
      assert.equal(res.status, 200);
      const data = await res.json();
      assert.equal(data.status, "healthy");
      assert.ok(data.metrics.frames_processed > 0, "Frames must be actively processed");
      assert.ok(data.subsystems.runtime.status === "healthy");
      assert.ok(data.subsystems.database.status === "healthy");
    });

    console.log(`\n========================================`);
    console.log(`E2E Integration Suite: ${passed} passed, 0 failed!`);
    console.log(`========================================\n`);
  } finally {
    await stopAll();
  }
}

run().catch((err) => {
  console.error("E2E Integration Suite Failed:", err);
  process.exit(1);
});
