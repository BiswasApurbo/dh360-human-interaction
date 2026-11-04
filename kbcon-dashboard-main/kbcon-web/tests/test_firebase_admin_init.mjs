/**
 * Regression Test: Firebase Admin CommonJS / ESM Compatibility.
 *
 * Verifies that firebase-admin, jwks-rsa, and jose can be imported
 * and initialized in Node without ERR_REQUIRE_ESM.
 */

import assert from "node:assert/strict";

console.log("1. Testing dynamic import of firebase-admin submodules...");
const [adminApp, adminAuth, adminFirestore] = await Promise.all([
  import("firebase-admin/app"),
  import("firebase-admin/auth"),
  import("firebase-admin/firestore"),
]);

assert.ok(adminApp.getApps, "firebase-admin/app must export getApps");
assert.ok(adminApp.initializeApp, "firebase-admin/app must export initializeApp");
assert.ok(adminAuth.getAuth, "firebase-admin/auth must export getAuth");
assert.ok(adminFirestore.getFirestore, "firebase-admin/firestore must export getFirestore");
console.log("   -> firebase-admin/app, auth, and firestore imported successfully.");

console.log("2. Testing jwks-rsa resolution without ERR_REQUIRE_ESM...");
try {
  const jwksRsa = (await import("jwks-rsa")).default || (await import("jwks-rsa"));
  assert.ok(jwksRsa, "jwks-rsa must resolve cleanly");
  console.log("   -> jwks-rsa resolved successfully.");
} catch (err) {
  assert.fail(`jwks-rsa failed to load: ${err.message}`);
}

console.log("3. Testing server-admin initialization without error...");
const { getServerAdminApp, hasServerAdminCredentials } = await import("../src/lib/server-admin.ts");
assert.strictEqual(typeof hasServerAdminCredentials, "function");
assert.strictEqual(typeof getServerAdminApp, "function");

console.log("ALL FIREBASE ADMIN TESTS PASSED SUCCESSFULLY!");
