import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import { getMemoryDevices } from "@/lib/device/provisioning";
import { getAdminFirestore, hasServerAdminCredentials } from "@/lib/server-admin";
import type { QueryDocumentSnapshot } from "firebase-admin/firestore";

export async function GET(request: Request) {
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
      { error: "FORBIDDEN", message: "Only Super Admin can access the Device Registry." },
      { status: 403 }
    );
  }

  const now = new Date().getTime();

  if (!hasServerAdminCredentials()) {
    const devicesList = Array.from(getMemoryDevices().values()).map((d) => {
      const lastHb = d.lastHeartbeat ? new Date(d.lastHeartbeat).getTime() : 0;
      const isOnline = lastHb > 0 && now - lastHb < 90_000;
      return {
        deviceId: d.deviceId,
        publicKey: d.publicKey,
        fingerprint: (d as unknown as Record<string, string>).fingerprint || "a9:8e:5b:cf:a5:b1:0a:65:b4:e7:b9:f4:9b:e6:39:b7",
        status: d.status,
        bindingStatus: d.robotId ? "bound" : "unbound",
        robotId: d.robotId || null,
        robotName: (d as unknown as Record<string, string>).robotName || (d.robotId ? `Robot ${d.robotId}` : null),
        tenantId: d.tenantId || null,
        tenantName: (d as unknown as Record<string, string>).tenantName || (d.tenantId ? "Salute" : null),
        deviceInfo: d.deviceInfo || {},
        lastHeartbeat: d.lastHeartbeat || null,
        isOnline,
        pairedAt: d.pairedAt || null,
        updatedAt: d.updatedAt || null,
      };
    });

    return NextResponse.json({ devices: devicesList });
  }

  try {
    const db = await getAdminFirestore();
    const snap = await db.collection("devices").get();

    const devices = snap.docs.map((doc: QueryDocumentSnapshot) => {
      const data = doc.data();
      const lastHb = data.lastHeartbeat ? new Date(data.lastHeartbeat).getTime() : 0;
      const isOnline = lastHb > 0 && now - lastHb < 90_000;

      return {
        deviceId: doc.id,
        publicKey: data.publicKey || "",
        fingerprint: data.fingerprint || "—",
        status: data.status || "registered",
        bindingStatus: data.bindingStatus || (data.robotId ? "bound" : "unbound"),
        robotId: data.robotId || null,
        robotName: data.robotName || (data.robotId ? `Robot ${data.robotId}` : null),
        tenantId: data.tenantId || null,
        tenantName: data.tenantName || (data.tenantId ? "Salute" : null),
        language: data.language || "bn",
        mode: data.mode || "education",
        deviceInfo: data.deviceInfo || {},
        lastHeartbeat: data.lastHeartbeat || null,
        isOnline,
        registeredAt: data.registeredAt || data.createdAt || null,
        pairedAt: data.pairedAt || null,
        updatedAt: data.updatedAt || null,
      };
    });

    return NextResponse.json({ devices });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to fetch device registry." },
      { status: 500 }
    );
  }
}
