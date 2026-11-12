/**
 * Server-Side Firebase Admin SDK Singleton.
 *
 * Exclusively executed on the server in Next.js Route Handlers and Server Actions.
 * Never bundled or exposed to the client browser.
 */

import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import type { App } from "firebase-admin/app";
import type { Auth } from "firebase-admin/auth";
import type { Firestore } from "firebase-admin/firestore";
import { getDh360Config } from "./dh360/config";
import { getEmbeddedServiceAccount } from "./server-credentials";

let adminApp: App | null = null;

function resolveServiceAccount(): Record<string, unknown> | null {
  // 1. Check FIREBASE_SERVICE_ACCOUNT_KEY or FIREBASE_SERVICE_ACCOUNT environment variable (raw JSON, base64, or path)
  const envKey = process.env.FIREBASE_SERVICE_ACCOUNT_KEY || process.env.FIREBASE_SERVICE_ACCOUNT;
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
      // Try base64 decoding
      try {
        const decoded = Buffer.from(envKey, "base64").toString("utf-8");
        return JSON.parse(decoded);
      } catch {
        // Fall through
      }
    }
  }

  // 2. Check GOOGLE_SERVICE_ACCOUNT_JSON
  const gJson = process.env.GOOGLE_SERVICE_ACCOUNT_JSON;
  if (gJson) {
    try {
      return JSON.parse(gJson);
    } catch {
      try {
        const decoded = Buffer.from(gJson, "base64").toString("utf-8");
        return JSON.parse(decoded);
      } catch {
        // Fall through
      }
    }
  }

  // 3. Check GOOGLE_APPLICATION_CREDENTIALS
  const gCred = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (gCred && fs.existsSync(gCred)) {
    try {
      return JSON.parse(fs.readFileSync(gCred, "utf-8"));
    } catch {
      // Fall through
    }
  }

  // 4. Check ~/.config/kbcon/firebase-service-account.json
  const defaultLocalPath = path.join(os.homedir(), ".config", "kbcon", "firebase-service-account.json");
  if (fs.existsSync(defaultLocalPath)) {
    try {
      return JSON.parse(fs.readFileSync(defaultLocalPath, "utf-8"));
    } catch {
      // Fall through
    }
  }

  // 5. Embedded server-only service account for production deployment
  try {
    const sa = getEmbeddedServiceAccount();
    if (sa) return sa;
  } catch {
    // Fall through
  }

  return null;
}

export function hasServerAdminCredentials(): boolean {
  return resolveServiceAccount() !== null;
}

export async function getServerAdminApp(): Promise<App> {
  if (adminApp) return adminApp;

  const { getApps, initializeApp, cert } = await import("firebase-admin/app");

  const existingApps = getApps();
  if (existingApps.length > 0) {
    adminApp = existingApps[0];
    return adminApp;
  }

  const projectId =
    process.env.FIREBASE_PROJECT_ID ||
    process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID ||
    "kbcon-production";

  const sa = resolveServiceAccount();
  if (sa) {
    adminApp = initializeApp({
      credential: cert(sa),
      projectId,
    });
  } else {
    adminApp = initializeApp({
      projectId,
    });
  }

  return adminApp;
}

export async function getAdminAuth(): Promise<Auth> {
  const app = await getServerAdminApp();
  const { getAuth } = await import("firebase-admin/auth");
  return getAuth(app);
}

export async function getAdminFirestore(): Promise<Firestore> {
  const app = await getServerAdminApp();
  const { getFirestore } = await import("firebase-admin/firestore");
  return getFirestore(app);
}

export async function forwardAdminRequestToBackend(
  subpath: string,
  method: string,
  body?: unknown
): Promise<Response> {
  const config = getDh360Config();
  const upstreamUrl = new URL(subpath, config.apiUrl);
  return fetch(upstreamUrl.toString(), {
    method,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${config.operatorToken}`,
    },
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
    signal: AbortSignal.timeout(10000),
  });
}
