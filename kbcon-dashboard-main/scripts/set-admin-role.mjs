#!/usr/bin/env node
/**
 * Server-Side Administrator Role Provisioning CLI
 *
 * Usage:
 *   node scripts/set-admin-role.mjs <email> [role] [--emulator]
 *
 * Arguments:
 *   <email>       The email of the existing Firebase Auth user to provision.
 *   [role]        The target role: "superAdmin" (default) or "admin".
 *   --emulator    Force using local Firebase Auth/Firestore emulator.
 *
 * Environment Variables (Production):
 *   FIREBASE_PROJECT_ID            Target Firebase Project ID (default: kbcon-dashboard-4b009)
 *   GOOGLE_APPLICATION_CREDENTIALS Path to service account JSON file
 *   FIREBASE_SERVICE_ACCOUNT_KEY   Path to service account JSON file, or inline JSON string
 *
 * Environment Variables (Emulator):
 *   FIREBASE_AUTH_EMULATOR_HOST    e.g. 127.0.0.1:9099
 *   FIRESTORE_EMULATOR_HOST        e.g. 127.0.0.1:8080
 *   NEXT_PUBLIC_USE_EMULATOR       "true"
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, "..");
const WEB_DIR = path.resolve(ROOT_DIR, "kbcon-web");

// Load .env.local if present in kbcon-web
const envLocalPath = path.join(WEB_DIR, ".env.local");
if (fs.existsSync(envLocalPath)) {
  const envContent = fs.readFileSync(envLocalPath, "utf-8");
  for (const line of envContent.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eqIdx = trimmed.indexOf("=");
    if (eqIdx > 0) {
      const key = trimmed.slice(0, eqIdx).trim();
      const val = trimmed.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
      if (!process.env[key]) {
        process.env[key] = val;
      }
    }
  }
}

// Resolve firebase-admin from functions directory
const req = createRequire(new URL("../functions/package.json", import.meta.url));
let admin;
try {
  admin = req("firebase-admin");
} catch {
  console.error(
    "Error: firebase-admin is not installed. Please run `cd functions && npm install` first."
  );
  process.exit(1);
}

// Parse arguments
const args = process.argv.slice(2);
const isEmulatorFlag = args.includes("--emulator");
const cleanArgs = args.filter((a) => !a.startsWith("--"));

const email = cleanArgs[0];
const targetRole = cleanArgs[1] || "superAdmin";

if (!email) {
  console.log(`
Usage:
  node scripts/set-admin-role.mjs <email> [role] [--emulator]

Examples:
  # Provision production superAdmin:
  GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json node scripts/set-admin-role.mjs admin@kbcon.ai superAdmin

  # Provision emulator admin:
  node scripts/set-admin-role.mjs dev@example.com admin --emulator
`);
  process.exit(1);
}

if (targetRole !== "superAdmin" && targetRole !== "admin") {
  console.error(`Error: Role must be either "superAdmin" or "admin". Received: "${targetRole}"`);
  process.exit(1);
}

const projectId =
  process.env.FIREBASE_PROJECT_ID ||
  process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID ||
  "kbcon-dashboard-4b009";

const isEmulator =
  isEmulatorFlag ||
  process.env.NEXT_PUBLIC_USE_EMULATOR === "true" ||
  Boolean(process.env.FIREBASE_AUTH_EMULATOR_HOST && !process.env.GOOGLE_APPLICATION_CREDENTIALS);

console.log(`Target Firebase Project: ${projectId}`);
console.log(`Mode: ${isEmulator ? "Firebase Emulator (Local)" : "Firebase Production (Cloud)"}`);
console.log(`User Email: ${email}`);
console.log(`Target Role: ${targetRole}\n`);

// Initialize Firebase Admin SDK
try {
  if (isEmulator) {
    if (!process.env.FIREBASE_AUTH_EMULATOR_HOST) {
      process.env.FIREBASE_AUTH_EMULATOR_HOST = "127.0.0.1:9099";
    }
    if (!process.env.FIRESTORE_EMULATOR_HOST) {
      process.env.FIRESTORE_EMULATOR_HOST = "127.0.0.1:8080";
    }
    admin.initializeApp({ projectId });
  } else {
    // Production / Cloud initialization
    let credential = null;
    const saKeyEnv = process.env.FIREBASE_SERVICE_ACCOUNT_KEY;

    if (saKeyEnv) {
      if (fs.existsSync(saKeyEnv)) {
        credential = admin.credential.cert(JSON.parse(fs.readFileSync(saKeyEnv, "utf-8")));
      } else {
        try {
          credential = admin.credential.cert(JSON.parse(saKeyEnv));
        } catch {
          console.error("Error: FIREBASE_SERVICE_ACCOUNT_KEY is not a valid JSON string or existing file path.");
          process.exit(1);
        }
      }
    } else if (process.env.GOOGLE_APPLICATION_CREDENTIALS) {
      const saPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
      if (!fs.existsSync(saPath)) {
        console.error(`Error: GOOGLE_APPLICATION_CREDENTIALS path not found: ${saPath}`);
        process.exit(1);
      }
      credential = admin.credential.applicationDefault();
    } else {
      console.error(`
================================================================================
PRODUCTION CREDENTIALS REQUIRED:
To provision roles in the production Firebase project ("${projectId}"),
provide a service account key using one of the following methods:

Option 1: Download Service Account Key JSON from Firebase Console:
  1. Go to Firebase Console -> Project Settings -> Service Accounts.
  2. Click "Generate new private key".
  3. Run:
     GOOGLE_APPLICATION_CREDENTIALS=/path/to/serviceAccountKey.json node scripts/set-admin-role.mjs ${email} ${targetRole}

Option 2: Inline or environment variable:
  FIREBASE_SERVICE_ACCOUNT_KEY=/path/to/key.json node scripts/set-admin-role.mjs ${email} ${targetRole}

Note: If you intended to test against the local emulator instead, run:
  node scripts/set-admin-role.mjs ${email} ${targetRole} --emulator
================================================================================
`);
      process.exit(1);
    }

    admin.initializeApp({
      credential,
      projectId,
    });
  }
} catch (err) {
  console.error("Failed to initialize Firebase Admin SDK:", err.message);
  process.exit(1);
}

async function main() {
  const auth = admin.auth();
  const db = admin.firestore();

  // 1. Fetch user by email
  let user;
  try {
    user = await auth.getUserByEmail(email);
  } catch (err) {
    if (err.code === "auth/user-not-found") {
      console.error(`\nError: User "${email}" was not found in Firebase Authentication for project "${projectId}".`);
      console.error("Please create the user first in Firebase Console or sign up through the application.");
      process.exit(1);
    }
    throw err;
  }

  // 2. Set Custom User Claims
  await auth.setCustomUserClaims(user.uid, { role: targetRole });
  console.log(`✓ Custom claim set: { role: "${targetRole}" } on UID: ${user.uid}`);

  // 3. Update or create Firestore user record
  await db
    .collection("users")
    .doc(user.uid)
    .set(
      {
        name: user.displayName || user.email?.split("@")[0] || "Admin",
        email: user.email,
        role: targetRole,
        active: true,
        updatedAt: new Date().toISOString(),
        provisionedBy: "set-admin-role-script",
      },
      { merge: true }
    );
  console.log(`✓ Firestore record updated in collection "users" for doc ID: ${user.uid}`);

  console.log(`
================================================================================
SUCCESS!
User ${email} (UID: ${user.uid}) has been provisioned as "${targetRole}".

IMPORTANT:
If the user is currently signed in, they must log out and sign back in
(or refresh their ID token with user.getIdToken(true)) for the new custom
claim to take effect in their session.
================================================================================
`);
}

main().catch((err) => {
  console.error("Provisioning failed:", err);
  process.exit(1);
});
