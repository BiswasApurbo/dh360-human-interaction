/**
 * Server-Side Firebase Admin SDK Singleton.
 *
 * Exclusively executed on the server in Next.js Route Handlers and Server Actions.
 * Never bundled or exposed to the client browser.
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { getApps, initializeApp, cert, type App } from "firebase-admin/app";
import { getAuth, type Auth } from "firebase-admin/auth";
import { getFirestore, type Firestore } from "firebase-admin/firestore";

let adminApp: App | null = null;

function resolveServiceAccount(): Record<string, unknown> | null {
  // 1. Check FIREBASE_SERVICE_ACCOUNT_KEY environment variable (raw JSON or path)
  const envKey = process.env.FIREBASE_SERVICE_ACCOUNT_KEY;
  if (envKey) {
    if (fs.existsSync(envKey)) {
      try {
        return JSON.parse(fs.readFileSync(envKey, "utf-8"));
      } catch {
        // Fall through
      }
    }
    try {
      return JSON.parse(envKey);
    } catch {
      // Fall through
    }
  }

  // 2. Check GOOGLE_APPLICATION_CREDENTIALS
  const gCred = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (gCred && fs.existsSync(gCred)) {
    try {
      return JSON.parse(fs.readFileSync(gCred, "utf-8"));
    } catch {
      // Fall through
    }
  }

  // 3. Check ~/.config/kbcon/firebase-service-account.json
  const defaultLocalPath = path.join(os.homedir(), ".config", "kbcon", "firebase-service-account.json");
  if (fs.existsSync(defaultLocalPath)) {
    try {
      return JSON.parse(fs.readFileSync(defaultLocalPath, "utf-8"));
    } catch {
      // Fall through
    }
  }

  return null;
}

export function getServerAdminApp(): App {
  if (adminApp) return adminApp;

  const existingApps = getApps();
  if (existingApps.length > 0) {
    adminApp = existingApps[0];
    return adminApp;
  }

  const projectId =
    process.env.FIREBASE_PROJECT_ID ||
    process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID ||
    "kbcon-dashboard-4b009";

  const sa = resolveServiceAccount();
  if (sa) {
    adminApp = initializeApp({
      credential: cert(sa),
      projectId,
    });
  } else {
    // Emulator or Application Default Credentials
    adminApp = initializeApp({
      projectId,
    });
  }

  return adminApp;
}

export function getAdminAuth(): Auth {
  return getAuth(getServerAdminApp());
}

export function getAdminFirestore(): Firestore {
  return getFirestore(getServerAdminApp());
}
