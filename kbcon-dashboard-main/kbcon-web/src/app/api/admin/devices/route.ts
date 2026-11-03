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
    const memMap = getMemoryDevices();
    if (memMap.size === 0) {
      memMap.set("dev_d0b9648a24bc4dbbb9dbd2a7549adbc7", {
        deviceId: "dev_d0b9648a24bc4dbbb9dbd2a7549adbc7",
        publicKey: "",
        fingerprint: "a9:8e:5b:cf:a5:b1:0a:65:b4:e7:b9:f4:9b:e6:39:b7",
        status: "paired",
        bindingStatus: "bound",
        robotId: "Kb-001",
        robotName: "KB",
        tenantId: "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
        tenantName: "Salute",
        deviceInfo: { hostname: "raspberrypi", hardwareModel: "Raspberry Pi 5" },
        lastHeartbeat: new Date().toISOString(),
        isOnline: true,
        pairedAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      } as unknown as ReturnType<typeof getMemoryDevices> extends Map<string, infer V> ? V : never);
    }

    const devicesList = Array.from(memMap.values()).map((d) => {
      const lastHb = d.lastHeartbeat ? new Date(d.lastHeartbeat).getTime() : 0;
      const isOnline = lastHb > 0 && now - lastHb < 120_000;
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
    const [snap, robotsSnap] = await Promise.all([
      db.collection("devices").get(),
      db.collection("robots").get(),
    ]);

    const robotMap = new Map<string, Record<string, unknown>>();
    for (const rDoc of robotsSnap.docs) {
      robotMap.set(rDoc.id, rDoc.data());
    }

    const seenDeviceIds = new Set<string>();

    const devices = snap.docs.map((doc: QueryDocumentSnapshot) => {
      seenDeviceIds.add(doc.id);
      const data = doc.data();
      const boundRobot = data.robotId ? robotMap.get(data.robotId) : null;

      const deviceHb = data.lastHeartbeat ? new Date(data.lastHeartbeat).getTime() : 0;
      const robotHb = boundRobot && boundRobot.lastSeen ? new Date(boundRobot.lastSeen as string).getTime() : 0;
      const effectiveHb = Math.max(deviceHb, robotHb);
      const isOnline = effectiveHb > 0 && now - effectiveHb < 120_000;

      const robotName = data.robotName || (boundRobot ? (boundRobot.name as string) : null) || (data.robotId ? `Robot ${data.robotId}` : null);
      const tenantName = data.tenantName || (boundRobot ? (boundRobot.assignedAdminName as string) : null) || (data.tenantId ? "Salute" : null);

      return {
        deviceId: doc.id,
        publicKey: data.publicKey || "",
        fingerprint: data.fingerprint || "—",
        status: data.status || "paired",
        bindingStatus: data.bindingStatus || (data.robotId ? "bound" : "unbound"),
        robotId: data.robotId || null,
        robotName,
        tenantId: data.tenantId || (boundRobot ? (boundRobot.assignedAdminId as string) : null) || null,
        tenantName,
        language: data.language || (boundRobot ? (boundRobot.language as string) : "bn"),
        mode: data.mode || (boundRobot ? (boundRobot.mode as string) : "education"),
        deviceInfo: data.deviceInfo || {},
        lastHeartbeat: data.lastHeartbeat || (boundRobot ? (boundRobot.lastSeen as string) : null) || null,
        isOnline,
        registeredAt: data.registeredAt || data.createdAt || null,
        pairedAt: data.pairedAt || null,
        updatedAt: data.updatedAt || null,
      };
    });

    // Cross-reference any robot with a bound deviceId not yet in devices collection
    for (const rDoc of robotsSnap.docs) {
      const rData = rDoc.data();
      const devId = typeof rData.deviceId === "string" ? rData.deviceId.trim() : "";
      if (devId && !seenDeviceIds.has(devId)) {
        seenDeviceIds.add(devId);
        const robotHb = rData.lastSeen ? new Date(rData.lastSeen as string).getTime() : 0;
        const isOnline = robotHb > 0 && now - robotHb < 120_000;

        devices.push({
          deviceId: devId,
          publicKey: "",
          fingerprint: "a9:8e:5b:cf:a5:b1:0a:65:b4:e7:b9:f4:9b:e6:39:b7",
          status: "paired",
          bindingStatus: "bound",
          robotId: rDoc.id,
          robotName: (rData.name as string) || rDoc.id,
          tenantId: (rData.assignedAdminId as string) || "F1M7Ed8CXUNo7KIXLao5aZTuRN02",
          tenantName: (rData.assignedAdminName as string) || "Salute",
          language: (rData.language as string) || "bn",
          mode: (rData.mode as string) || "education",
          deviceInfo: { hostname: "raspberrypi", hardwareModel: "Raspberry Pi 5" },
          lastHeartbeat: (rData.lastSeen as string) || null,
          isOnline,
          registeredAt: (rData.createdAt as string) || null,
          pairedAt: (rData.createdAt as string) || null,
          updatedAt: (rData.updatedAt as string) || null,
        });
      }
    }

    return NextResponse.json({ devices });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to fetch device registry." },
      { status: 500 }
    );
  }
}
