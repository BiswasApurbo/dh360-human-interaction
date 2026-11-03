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
      { error: "FORBIDDEN", message: "Only Super Admin can unbind physical devices." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const deviceId = typeof body.deviceId === "string" ? body.deviceId.trim() : "";
  if (!deviceId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "deviceId is required." }, { status: 400 });
  }

  const now = new Date().toISOString();

  if (!hasServerAdminCredentials()) {
    const memDev = getMemoryDevices().get(deviceId);
    if (memDev) {
      memDev.robotId = "";
      memDev.status = "unpaired";
      memDev.updatedAt = now;
    }
    return NextResponse.json({
      success: true,
      message: `Device ${deviceId} successfully unbound.`,
    });
  }

  try {
    const adminDb = await getAdminFirestore();
    const deviceRef = adminDb.collection("devices").doc(deviceId);
    const devSnap = await deviceRef.get();

    if (!devSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Device ${deviceId} not found.` }, { status: 404 });
    }

    const devData = devSnap.data() || {};
    const previousRobotId = devData.robotId;

    if (previousRobotId) {
      const robotRef = adminDb.collection("robots").doc(previousRobotId);
      await robotRef.set(
        {
          deviceId: null,
          deviceStatus: "unpaired",
          updatedAt: now,
        },
        { merge: true }
      );
    }

    await deviceRef.set(
      {
        robotId: null,
        robotName: null,
        tenantId: null,
        tenantName: null,
        status: "unpaired",
        bindingStatus: "unbound",
        updatedAt: now,
      },
      { merge: true }
    );

    return NextResponse.json({
      success: true,
      message: `Device ${deviceId} successfully unbound.`,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to unbind device." },
      { status: 500 }
    );
  }
}
