import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can revoke device pairings." },
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

  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";
  const deviceId = typeof body.deviceId === "string" ? body.deviceId.trim() : "";

  if (!robotId && !deviceId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Either robotId or deviceId is required." },
      { status: 400 }
    );
  }

  const now = new Date().toISOString();

  if (hasServerAdminCredentials()) {
    try {
      const db = await getAdminFirestore();

      if (deviceId) {
        await db.collection("devices").doc(deviceId).set(
          {
            status: "revoked",
            deviceTokenHash: null,
            robotId: null,
            revokedAt: now,
            revokedBy: authResult.principal.uid,
            updatedAt: now,
          },
          { merge: true }
        );
      }

      if (robotId) {
        await db.collection("robots").doc(robotId).set(
          {
            deviceId: null,
            deviceStatus: "unpaired",
            status: "offline",
            updatedAt: now,
          },
          { merge: true }
        );
      }
    } catch (err: unknown) {
      return NextResponse.json(
        {
          error: "INTERNAL_ERROR",
          message: err instanceof Error ? err.message : "Failed to revoke device pairing.",
        },
        { status: 500 }
      );
    }
  } else {
    if (deviceId) {
      const d = getMemoryDevices().get(deviceId);
      if (d) {
        d.status = "revoked";
        d.robotId = "";
      }
    }
  }

  return NextResponse.json({
    success: true,
    message: `Device pairing revoked successfully.`,
  });
}
