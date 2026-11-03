import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  hashPairingCode,
  hashToken,
  normalizePairingCode,
  getMemoryPairingSessions,
  getMemoryDevices,
  type PairingSession,
  type PairedDevice,
} from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";
import type { DocumentReference } from "firebase-admin/firestore";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.mustChangePassword) {
    return NextResponse.json(
      {
        error: "PASSWORD_CHANGE_REQUIRED",
        message: "Mandatory password change required before pairing devices.",
      },
      { status: 403 }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can pair physical devices to robots." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json(
      { error: "BAD_REQUEST", message: "Invalid JSON body." },
      { status: 400 }
    );
  }

  const rawCode = typeof body.code === "string" ? body.code.trim() : "";
  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";

  if (!rawCode) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Pairing code is required." },
      { status: 400 }
    );
  }

  if (!robotId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Target robotId is required." },
      { status: 400 }
    );
  }

  const normalizedCode = normalizePairingCode(rawCode);
  const targetCodeHash = hashPairingCode(normalizedCode);
  const now = new Date();

  let session: PairingSession | null = null;
  let sessionRef: DocumentReference | null = null;
  let robotRef: DocumentReference | null = null;
  let robotData: Record<string, unknown> | null = null;

  if (hasServerAdminCredentials()) {
    try {
      const db = await getAdminFirestore();

      // Find pending session matching code hash
      const sessionQuery = await db
        .collection("pairing_sessions")
        .where("codeHash", "==", targetCodeHash)
        .where("status", "==", "pending")
        .limit(1)
        .get();

      if (!sessionQuery.empty) {
        const snap = sessionQuery.docs[0];
        session = snap.data() as PairingSession;
        sessionRef = snap.ref;
      }

      if (!session) {
        return NextResponse.json(
          {
            error: "INVALID_CODE",
            message: "Invalid, expired, or already used pairing code.",
          },
          { status: 400 }
        );
      }

      // Check 10-minute expiry
      if (new Date(session.expiresAt).getTime() < now.getTime()) {
        if (sessionRef) {
          await sessionRef.set({ status: "expired" }, { merge: true });
        }
        return NextResponse.json(
          {
            error: "EXPIRED_CODE",
            message: "Pairing code has expired. Generate a new code with 'kbcon-device pair'.",
          },
          { status: 400 }
        );
      }

      // Verify robot exists in Firestore
      robotRef = db.collection("robots").doc(robotId);
      let robotSnap = await robotRef.get();
      if (!robotSnap.exists) {
        // Try canonical normalization match
        const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
        const allRobots = await db.collection("robots").get();
        for (const d of allRobots.docs) {
          if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
            robotRef = d.ref;
            robotSnap = d;
            break;
          }
        }
      }

      if (!robotSnap.exists) {
        return NextResponse.json(
          { error: "NOT_FOUND", message: `Target robot '${robotId}' not found.` },
          { status: 404 }
        );
      }

      robotData = robotSnap.data() || {};
    } catch (err: unknown) {
      return NextResponse.json(
        {
          error: "INTERNAL_ERROR",
          message: err instanceof Error ? err.message : "Database error during pairing verification.",
        },
        { status: 500 }
      );
    }
  } else {
    // Memory fallback for mock/tests
    for (const s of getMemoryPairingSessions().values()) {
      if (s.codeHash === targetCodeHash && s.status === "pending") {
        session = s;
        break;
      }
    }

    if (!session) {
      return NextResponse.json(
        { error: "INVALID_CODE", message: "Invalid or expired pairing code." },
        { status: 400 }
      );
    }

    if (new Date(session.expiresAt).getTime() < now.getTime()) {
      session.status = "expired";
      return NextResponse.json(
        { error: "EXPIRED_CODE", message: "Pairing code has expired." },
        { status: 400 }
      );
    }

    robotData = {
      id: robotId,
      name: `KBCON Robot ${robotId}`,
      language: "bn",
      mode: "education",
      type: "Humanoid",
    };
  }

  // Generate high-entropy device token (SHA-256 stored in DB, raw returned to device)
  const rawDeviceToken = crypto.randomBytes(32).toString("hex");
  const deviceTokenHash = hashToken(rawDeviceToken);
  const resolvedRobotId = (robotData?.id as string) || robotId;
  const resolvedTenantId =
    (robotData?.assignedAdminId as string) || (robotData?.tenantId as string) || null;

  const robotConfig = {
    robotId: resolvedRobotId,
    name: (robotData?.name as string) || `KBCON ${resolvedRobotId}`,
    language: (robotData?.language as string) || "bn",
    mode: (robotData?.mode as string) || "education",
    type: (robotData?.type as string) || "Humanoid",
  };

  const deviceRecord: PairedDevice = {
    deviceId: session.deviceId,
    robotId: resolvedRobotId,
    tenantId: resolvedTenantId,
    publicKey: session.publicKey,
    deviceTokenHash,
    status: "paired",
    pairedAt: now.toISOString(),
    pairedBy: authResult.principal.uid,
    deviceInfo: session.deviceInfo || {},
    updatedAt: now.toISOString(),
  };

  if (hasServerAdminCredentials()) {
    try {
      const db = await getAdminFirestore();

      // 1. Record device registration
      await db.collection("devices").doc(session.deviceId).set(deviceRecord, { merge: true });

      // 2. Bind robot to device
      if (robotRef) {
        await robotRef.set(
          {
            deviceId: session.deviceId,
            deviceStatus: "paired",
            updatedAt: now.toISOString(),
          },
          { merge: true }
        );
      }

      // 3. Mark session as approved
      if (sessionRef) {
        await sessionRef.set(
          {
            status: "approved",
            robotId: resolvedRobotId,
            tenantId: resolvedTenantId,
            deviceToken: rawDeviceToken,
            robotConfig,
            approvedAt: now.toISOString(),
            approvedBy: authResult.principal.uid,
          },
          { merge: true }
        );
      }
    } catch (err: unknown) {
      return NextResponse.json(
        {
          error: "INTERNAL_ERROR",
          message: err instanceof Error ? err.message : "Failed to record pairing in database.",
        },
        { status: 500 }
      );
    }
  } else {
    getMemoryDevices().set(session.deviceId, deviceRecord);
    session.status = "approved";
    session.robotId = resolvedRobotId;
    session.tenantId = resolvedTenantId;
    session.deviceToken = rawDeviceToken;
    session.robotConfig = robotConfig;
  }

  return NextResponse.json({
    success: true,
    message: `Physical device ${session.deviceId} successfully paired to robot ${resolvedRobotId}.`,
    deviceId: session.deviceId,
    robotId: resolvedRobotId,
    tenantId: resolvedTenantId,
  });
}
