/**
 * Dual-Project Firebase Token Verification Bridge Security Suite
 */

import assert from "node:assert";
import crypto from "node:crypto";
import { authenticateFirebaseToken } from "../src/lib/dh360/auth.ts";

console.log("Starting Dual-Project Token Bridge Security Verification...\n");

// Generate local RSA test keypair
const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", {
  modulusLength: 2048,
  publicKeyEncoding: { type: "spki", format: "pem" },
  privateKeyEncoding: { type: "pkcs8", format: "pem" }
});

const nowSec = Math.floor(Date.now() / 1000);
const testKid = "test-key-id-2026";

function createSignedJwt(payloadObj, kid = testKid, privKey = privateKey) {
  const header = Buffer.from(JSON.stringify({ alg: "RS256", typ: "JWT", kid })).toString("base64url");
  const payload = Buffer.from(JSON.stringify(payloadObj)).toString("base64url");
  const signingInput = `${header}.${payload}`;
  const sign = crypto.createSign("RSA-SHA256");
  sign.update(signingInput);
  const signature = sign.sign(privKey).toString("base64url");
  return `${header}.${payload}.${signature}`;
}

// We mock the public cert resolver by intercepting global fetch for Google public certs
const originalFetch = globalThis.fetch;
globalThis.fetch = async function(url, options) {
  if (typeof url === "string" && url.includes("robot/v1/metadata/x509/securetoken@system.gserviceaccount.com")) {
    return {
      ok: true,
      headers: { get: () => "max-age=3600" },
      json: async () => ({ [testKid]: publicKey })
    };
  }
  return originalFetch(url, options);
};

let passed = 0;
let total = 0;

async function test(name, fn) {
  total++;
  try {
    await fn();
    console.log(`✓ [PASS] ${name}`);
    passed++;
  } catch (err) {
    console.error(`✗ [FAIL] ${name}:`, err.message);
  }
}

async function runTests() {
  // 1. Positive: Destination project token (kbcon-production-dhbd360)
  await test("1. Valid token for destination project (kbcon-production-dhbd360) is accepted", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "SWUY3vnq4MblnDvSencCYS7iOeH2",
      email: "digitalhealthbd360@gmail.com",
      role: "superAdmin",
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, true);
    assert.strictEqual(res.principal.role, "superAdmin");
    assert.strictEqual(res.principal.email, "digitalhealthbd360@gmail.com");
  });

  // 2. Positive: Legacy project token (kbcon-production)
  await test("2. Valid token for legacy project (kbcon-production) is accepted during bridge window", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production",
      iss: "https://securetoken.google.com/kbcon-production",
      sub: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
      email: "supplier-salute46@bravealias.com",
      role: "admin",
      permissions: { dashboard: true, robots: true, assistant: true, people: true, knowledgeBase: true, telemetry: true, settings: true },
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, true);
    assert.strictEqual(res.principal.role, "admin");
    assert.strictEqual(res.principal.permissions.robots, true);
  });

  // 3. Negative: Unrelated project token
  await test("3. Token from unrelated project (unrelated-rogue-project) is strictly rejected", async () => {
    const token = createSignedJwt({
      aud: "unrelated-rogue-project",
      iss: "https://securetoken.google.com/unrelated-rogue-project",
      sub: "rogue-user-999",
      email: "rogue@attacker.com",
      role: "superAdmin",
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
    assert.match(res.message, /Token issuer or audience does not match/);
  });

  // 4. Negative: Mismatched issuer and audience
  await test("4. Token with mismatched issuer and audience is strictly rejected", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production", // mismatched!
      sub: "user-123",
      email: "test@kbcon.local",
      role: "admin",
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
  });

  // 5. Negative: Expired token
  await test("5. Expired token is rejected", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "user-123",
      email: "test@kbcon.local",
      role: "superAdmin",
      iat: nowSec - 7200,
      auth_time: nowSec - 7200,
      exp: nowSec - 3600, // expired 1 hour ago
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
    assert.strictEqual(res.code, "UNAUTHENTICATED");
  });

  // 6. Negative: Future issuance time (iat)
  await test("6. Token issued in the far future is rejected", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "user-123",
      email: "test@kbcon.local",
      role: "superAdmin",
      iat: nowSec + 600, // 10 minutes in the future
      auth_time: nowSec + 600,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
  });

  // 7. Negative: Corrupted signature
  await test("7. Token with corrupted cryptographic signature is rejected", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "user-123",
      email: "test@kbcon.local",
      role: "superAdmin",
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const corruptedToken = token.slice(0, -6) + "XXXXXX";
    const res = await authenticateFirebaseToken(`Bearer ${corruptedToken}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
    assert.match(res.message, /signature/i);
  });

  // 8. Negative: Unprivileged role
  await test("8. Authenticated token with unprivileged role (viewer) is rejected with HTTP 403", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "viewer-123",
      email: "viewer@external.local",
      role: "viewer", // not admin or superAdmin
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 403);
    assert.strictEqual(res.code, "FORBIDDEN");
  });

  // 9. Negative: Missing subject (sub)
  await test("9. Token with empty subject identifier is rejected", async () => {
    const token = createSignedJwt({
      aud: "kbcon-production-dhbd360",
      iss: "https://securetoken.google.com/kbcon-production-dhbd360",
      sub: "",
      email: "empty@kbcon.local",
      role: "admin",
      iat: nowSec - 60,
      auth_time: nowSec - 60,
      exp: nowSec + 3600,
    });
    const res = await authenticateFirebaseToken(`Bearer ${token}`);
    assert.strictEqual(res.success, false);
    assert.strictEqual(res.status, 401);
  });

  console.log(`\n========================================`);
  console.log(`Results: ${passed} / ${total} tests passed.`);
  console.log(`========================================\n`);

  if (passed !== total) {
    process.exit(1);
  }
}

runTests().catch(console.error);
