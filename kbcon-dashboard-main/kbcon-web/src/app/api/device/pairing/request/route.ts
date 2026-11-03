import { NextResponse } from "next/server";
import crypto from "node:crypto";
import {
  generatePairingCode,
  hashPairingCode,
  generateChallenge,
  getMemoryPairingSessions,
  type PairingSession,
} from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "BAD_REQUEST", message: "Invalid JSON body." },
      { status: 400 }
    );
  }

  const deviceId = typeof body.deviceId === "string" ? body.deviceId.trim() : "";
  const publicKey = typeof body.publicKey === "string" ? body.publicKey.trim() : "";
  const deviceInfo =
    body.deviceInfo && typeof body.deviceInfo === "object"
      ? (body.deviceInfo as Record<string, unknown>)
      : {};

  if (!deviceId || deviceId.length < 8 || deviceId.length > 64) {
    return NextResponse.json(
      {
        error: "INVALID_ARGUMENT",
        message: "deviceId must be between 8 and 64 characters.",
      },
      { status: 400 }
    );
  }

  if (!publicKey || !publicKey.includes("PUBLIC KEY")) {
    return NextResponse.json(
      {
        error: "INVALID_ARGUMENT",
        message: "publicKey must be a valid PEM-encoded public key.",
      },
      { status: 400 }
    );
  }

  const code = generatePairingCode();
  const normalizedCode = code.replace(/-/g, "");
  const codeHash = hashPairingCode(normalizedCode);
  const challenge = generateChallenge();
  const sessionId = crypto.randomUUID();
  const now = new Date();
  const expiresAt = new Date(now.getTime() + 10 * 60 * 1000).toISOString();

  const sessionData: PairingSession = {
    sessionId,
    deviceId,
    publicKey,
    deviceInfo,
    codeHash,
    challenge,
    status: "pending",
    createdAt: now.toISOString(),
    expiresAt,
  };

  if (hasServerAdminCredentials()) {
    try {
      const db = await getAdminFirestore();
      await db.collection("pairing_sessions").doc(sessionId).set(sessionData);
    } catch (err: unknown) {
      return NextResponse.json(
        {
          error: "INTERNAL_ERROR",
          message:
            err instanceof Error
              ? err.message
              : "Failed to create pairing session in database.",
        },
        { status: 500 }
      );
    }
  } else {
    getMemoryPairingSessions().set(sessionId, sessionData);
  }

  return NextResponse.json({
    success: true,
    code,
    sessionId,
    challenge,
    expiresAt,
  });
}
