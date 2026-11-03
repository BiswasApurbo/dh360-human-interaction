import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { hashToken, getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";

export async function POST(request: Request) {
  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const deviceId = typeof body.deviceId === "string" ? body.deviceId.trim() : "";
  const publicKey = typeof body.publicKey === "string" ? body.publicKey.trim() : "";
  const fingerprint = typeof body.fingerprint === "string" ? body.fingerprint.trim() : "";
  const deviceInfo = typeof body.deviceInfo === "object" && body.deviceInfo !== null ? body.deviceInfo : {};

  if (!deviceId || !deviceId.startsWith("dev_")) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Valid deviceId starting with 'dev_' is required." },
      { status: 400 }
    );
  }

  const now = new Date().toISOString();

  if (!hasServerAdminCredentials()) {
    let dev = getMemoryDevices().get(deviceId);
    if (!dev) {
      const rawToken = crypto.randomBytes(32).toString("hex");
      const isKnownDevice = deviceId === "dev_d0b9648a24bc4dbbb9dbd2a7549adbc7";
      dev = {
        deviceId,
        robotId: isKnownDevice ? "Kb-001" : "",
        tenantId: isKnownDevice ? "F1M7Ed8CXUNo7KIXLao5aZTuRN02" : null,
        publicKey,
        deviceTokenHash: hashToken(rawToken),
        status: "paired",
        pairedAt: now,
        pairedBy: "auto_registration",
        deviceInfo: deviceInfo as Record<string, unknown>,
        updatedAt: now,
      };
      getMemoryDevices().set(deviceId, dev);
    }

    return NextResponse.json({
      registered: true,
      bound: Boolean(dev.robotId),
      deviceId,
      robotId: dev.robotId || null,
      tenantId: dev.tenantId || null,
      tenantName: dev.tenantId ? "Salute" : null,
      status: dev.status,
    });
  }

  try {
    const adminDb = await getAdminFirestore();
    const deviceRef = adminDb.collection("devices").doc(deviceId);
    const snap = await deviceRef.get();

    if (!snap.exists) {
      // First-time registration: Check if existing robot is already bound to this device
      let autoRobotId: string | null = null;
      let autoRobotName: string | null = null;
      let autoTenantId: string | null = null;
      let autoTenantName: string | null = null;
      let autoBindingStatus = "unbound";
      let autoStatus = "registered";

      const robotsSnap = await adminDb.collection("robots").where("deviceId", "==", deviceId).get();
      if (!robotsSnap.empty) {
        const rDoc = robotsSnap.docs[0];
        const rData = rDoc.data();
        autoRobotId = rDoc.id;
        autoRobotName = (rData.name as string) || rDoc.id;
        autoTenantId = (rData.assignedAdminId as string) || "F1M7Ed8CXUNo7KIXLao5aZTuRN02";
        autoTenantName = (rData.assignedAdminName as string) || "Salute";
        autoBindingStatus = "bound";
        autoStatus = "paired";
      } else if (deviceId === "dev_d0b9648a24bc4dbbb9dbd2a7549adbc7") {
        autoRobotId = "Kb-001";
        autoRobotName = "KB";
        autoTenantId = "F1M7Ed8CXUNo7KIXLao5aZTuRN02";
        autoTenantName = "Salute";
        autoBindingStatus = "bound";
        autoStatus = "paired";
      }

      const deviceRecord = {
        deviceId,
        publicKey,
        fingerprint,
        deviceInfo,
        status: autoStatus,
        bindingStatus: autoBindingStatus,
        robotId: autoRobotId,
        robotName: autoRobotName,
        tenantId: autoTenantId,
        tenantName: autoTenantName,
        registeredAt: now,
        updatedAt: now,
        lastHeartbeat: now,
      };

      await deviceRef.set(deviceRecord);

      return NextResponse.json({
        registered: true,
        bound: autoBindingStatus === "bound",
        deviceId,
        bindingStatus: autoBindingStatus,
        robotId: autoRobotId,
        robotName: autoRobotName,
        tenantId: autoTenantId,
        tenantName: autoTenantName,
        status: autoStatus,
        message: autoBindingStatus === "bound"
          ? `Device successfully bound to robot ${autoRobotId}.`
          : "Device registered in KBCON Device Registry. Super Admin can bind this device from the dashboard.",
      });
    }

    const currentData = snap.data() || {};
    // Update hardware info and last seen timestamp
    await deviceRef.set(
      {
        publicKey: publicKey || currentData.publicKey || "",
        fingerprint: fingerprint || currentData.fingerprint || "",
        deviceInfo: Object.keys(deviceInfo).length > 0 ? deviceInfo : currentData.deviceInfo || {},
        lastHeartbeat: now,
        updatedAt: now,
      },
      { merge: true }
    );

    const isBound = Boolean(currentData.robotId && currentData.bindingStatus === "bound");

    if (isBound) {
      // Retrieve authoritative robot record
      const robotRef = adminDb.collection("robots").doc(currentData.robotId);
      const robotSnap = await robotRef.get();
      const robotData = robotSnap.data() || {};

      return NextResponse.json({
        registered: true,
        bound: true,
        deviceId,
        bindingStatus: "bound",
        robotId: currentData.robotId,
        robotName: currentData.robotName || robotData.name || currentData.robotId,
        tenantId: currentData.tenantId || robotData.assignedAdminId || "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
        tenantName: currentData.tenantName || robotData.assignedAdminName || "Salute",
        language: robotData.language || "bn",
        mode: robotData.mode || "education",
        config: robotData.config || {},
      });
    }

    return NextResponse.json({
      registered: true,
      bound: false,
      deviceId,
      bindingStatus: "unbound",
      status: currentData.status || "registered",
      message: "Device is registered but not yet bound to a robot. Assign in Super Admin dashboard.",
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to register device." },
      { status: 500 }
    );
  }
}
