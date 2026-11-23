import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminFirestore,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

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
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can create robots." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";
  const name = typeof body.name === "string" ? body.name.trim() : "";
  const language = body.language === "en" ? "en" : "bn";
  const mode = body.mode === "healthcare" ? "healthcare" : "education";

  if (!robotId || robotId.length < 3 || robotId.length > 32) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Robot ID must be between 3 and 32 characters." },
      { status: 400 }
    );
  }

  if (!name || name.length < 2) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "Robot Name must be at least 2 characters." },
      { status: 400 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend("/admin/robots", "POST", {
        robotId,
        name,
        language,
        mode,
        type: "Humanoid",
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (data.message || "Failed to create robot.");
        return NextResponse.json(
          { error: data.error || (res.status === 409 ? "ALREADY_EXISTS" : "CREATION_FAILED"), message },
          { status: res.status }
        );
      }
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "CREATION_FAILED", message: err instanceof Error ? err.message : "Failed to create robot." },
        { status: 500 }
      );
    }
  }

  try {
    const adminDb = await getAdminFirestore();
    const robotRef = adminDb.collection("robots").doc(robotId);
    const docSnap = await robotRef.get();

    if (docSnap.exists) {
      return NextResponse.json(
        { error: "ALREADY_EXISTS", message: `Robot ${robotId} already exists.` },
        { status: 409 }
      );
    }

    const now = new Date().toISOString();
    const secret = crypto.randomBytes(24).toString("hex");

    await robotRef.set({
      id: robotId,
      name,
      type: "Humanoid",
      language,
      mode,
      status: "online",
      assignedAdminId: null,
      assignedAdminName: null,
      createdAt: now,
      updatedAt: now,
    });

    return NextResponse.json({ success: true, robotId, secret });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to create robot." },
      { status: 500 }
    );
  }
}

export async function PATCH(request: Request) {
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
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const robotId = typeof body.robotId === "string" ? body.robotId.trim() : "";
  if (!robotId) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "robotId is required." }, { status: 400 });
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend("/admin/robots", "PATCH", body);
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (data.message || "Failed to update robot.");
        return NextResponse.json({ error: data.error || "UPDATE_FAILED", message }, { status: res.status });
      }
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "UPDATE_FAILED", message: err instanceof Error ? err.message : "Failed to update robot." },
        { status: 500 }
      );
    }
  }

  try {
    const adminDb = await getAdminFirestore();
    let robotRef = adminDb.collection("robots").doc(robotId);
    let docSnap = await robotRef.get();

    // Canonical resolution: match across case and hyphenation (KB001 <-> Kb-001 <-> KB-001)
    if (!docSnap.exists) {
      const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      const allRobots = await adminDb.collection("robots").get();
      for (const d of allRobots.docs) {
        if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
          robotRef = d.ref;
          docSnap = d;
          break;
        }
      }
    }

    if (!docSnap.exists) {
      return NextResponse.json({ error: "NOT_FOUND", message: `Robot ${robotId} not found.` }, { status: 404 });
    }

    const currentData = docSnap.data() || {};
    const isSuper = authResult.principal.role === "superAdmin";
    const isAdminOwner =
      authResult.principal.role === "admin" &&
      currentData.assignedAdminId === authResult.principal.uid;

    if (!isSuper && !isAdminOwner) {
      return NextResponse.json({ error: "FORBIDDEN", message: "Not authorized to update this robot." }, { status: 403 });
    }

    const now = new Date().toISOString();
    const updates: Record<string, unknown> = { updatedAt: now };

    if ("adminId" in body) {
      if (!isSuper) {
        return NextResponse.json({ error: "FORBIDDEN", message: "Only Super Admin can assign robots." }, { status: 403 });
      }
      const adminId = typeof body.adminId === "string" && body.adminId.trim() ? body.adminId.trim() : null;
      let adminName: string | null = null;
      if (adminId) {
        const uDoc = await adminDb.collection("users").doc(adminId).get();
        if (uDoc.exists) {
          const uData = uDoc.data() || {};
          adminName = (uData.name as string) || (uData.email as string) || null;
          // Synchronize user doc: ensure robotId is in user's assignedRobots
          const currentAssigned = Array.isArray(uData.assignedRobots) ? (uData.assignedRobots as string[]) : [];
          if (!currentAssigned.includes(docSnap.id)) {
            await uDoc.ref.set({ assignedRobots: [...currentAssigned, docSnap.id] }, { merge: true });
          }
        }
      }

      // If reassigned from another admin, remove robotId from previous admin's assignedRobots
      const prevAdminId = typeof currentData.assignedAdminId === "string" ? currentData.assignedAdminId.trim() : null;
      if (prevAdminId && prevAdminId !== adminId) {
        const prevDoc = await adminDb.collection("users").doc(prevAdminId).get();
        if (prevDoc.exists) {
          const pData = prevDoc.data() || {};
          const pRobots = Array.isArray(pData.assignedRobots) ? (pData.assignedRobots as string[]) : [];
          const updatedRobots = pRobots.filter((r) => r !== docSnap.id && r !== robotId);
          await prevDoc.ref.set({ assignedRobots: updatedRobots }, { merge: true });
        }
      }

      updates.assignedAdminId = adminId;
      updates.assignedAdminName = adminName;
    }

    if ("type" in body || "mode" in body) {
      if (!isSuper) {
        return NextResponse.json(
          { error: "FORBIDDEN", message: "Only Super Admin can change robot type." },
          { status: 403 }
        );
      }
      const rawType = String(body.type ?? body.mode ?? "").trim().toLowerCase();
      if (rawType === "health" || rawType === "healthcare") {
        updates.mode = "healthcare";
        updates.type = "Health";
      } else if (rawType === "education") {
        updates.mode = "education";
        updates.type = "Education";
      } else {
        return NextResponse.json(
          { error: "INVALID_ARGUMENT", message: "Robot type must be either 'Health' or 'Education'." },
          { status: 400 }
        );
      }
    }

    if ("language" in body) {
      if (!isSuper) {
        return NextResponse.json(
          { error: "FORBIDDEN", message: "Only Super Admin can change robot language." },
          { status: 403 }
        );
      }
      const rawLang = String(body.language ?? "").trim().toLowerCase();
      if (rawLang === "en" || rawLang === "english") {
        updates.language = "en";
      } else if (rawLang === "bn" || rawLang === "bangla") {
        updates.language = "bn";
      } else {
        return NextResponse.json(
          { error: "INVALID_ARGUMENT", message: "Robot language must be either 'en' or 'bn'." },
          { status: 400 }
        );
      }
      if (!updates.config && currentData.config && typeof currentData.config === "object") {
        updates.config = { ...currentData.config, defaultLanguage: updates.language };
      }
    }

    if (body.config && typeof body.config === "object") {
      updates.config = { ...(updates.config as Record<string, unknown> || {}), ...body.config };
    }

    await robotRef.set(updates, { merge: true });
    return NextResponse.json({ success: true, robotId, ...updates });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to update robot." },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
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
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  // Strict backend role enforcement: Only Super Admin can delete robots
  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can delete robots." },
      { status: 403 }
    );
  }

  // Read robotId from query parameter or JSON body
  const { searchParams } = new URL(request.url);
  let robotId = searchParams.get("robotId") || searchParams.get("id") || "";

  if (!robotId) {
    try {
      const body = await request.json();
      robotId = typeof body.robotId === "string" ? body.robotId.trim() : typeof body.id === "string" ? body.id.trim() : "";
    } catch {
      // Body empty or not JSON, rely on query parameter
    }
  }

  robotId = robotId.trim();
  if (!robotId) {
    return NextResponse.json(
      { error: "INVALID_ARGUMENT", message: "robotId is required." },
      { status: 400 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend(`/admin/robots/${encodeURIComponent(robotId)}`, "DELETE", {});
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        return NextResponse.json(
          { error: data.error || "DELETION_FAILED", message: data.message || data.detail || "Failed to delete robot." },
          { status: res.status }
        );
      }
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "DELETION_FAILED", message: err instanceof Error ? err.message : "Failed to delete robot." },
        { status: 500 }
      );
    }
  }

  try {
    const adminDb = await getAdminFirestore();
    let robotRef = adminDb.collection("robots").doc(robotId);
    let docSnap = await robotRef.get();

    // Canonical resolution across case and hyphenation
    if (!docSnap.exists) {
      const cleanTarget = robotId.replace(/[^a-zA-Z0-9]/g, "").toLowerCase();
      const allRobots = await adminDb.collection("robots").get();
      for (const d of allRobots.docs) {
        if (d.id.replace(/[^a-zA-Z0-9]/g, "").toLowerCase() === cleanTarget) {
          robotRef = d.ref;
          docSnap = d;
          break;
        }
      }
    }

    if (!docSnap.exists) {
      return NextResponse.json(
        { error: "NOT_FOUND", message: `Robot ${robotId} not found.` },
        { status: 404 }
      );
    }

    const now = new Date().toISOString();
    const robotData = docSnap.data() || {};
    const actualRobotId = docSnap.id;

    // 1. Safely unbind associated physical device without deleting hardware record
    const deviceId = typeof robotData.deviceId === "string" ? robotData.deviceId.trim() : "";
    if (deviceId) {
      try {
        await adminDb.collection("devices").doc(deviceId).set(
          {
            status: "unpaired",
            robotId: null,
            deviceTokenHash: null,
            updatedAt: now,
          },
          { merge: true }
        );
      } catch (devErr) {
        console.warn(`[admin/robots DELETE] Failed unbinding device ${deviceId}:`, devErr);
      }
    }

    // Also unbind any device where robotId matches
    try {
      const boundDevicesSnap = await adminDb.collection("devices").where("robotId", "==", actualRobotId).get();
      for (const devDoc of boundDevicesSnap.docs) {
        await devDoc.ref.set(
          {
            status: "unpaired",
            robotId: null,
            deviceTokenHash: null,
            updatedAt: now,
          },
          { merge: true }
        );
      }
    } catch {
      // Ignore query errors
    }

    // 2. Safely unbind from assigned admin user's assignedRobots array
    const assignedAdminId = typeof robotData.assignedAdminId === "string" ? robotData.assignedAdminId.trim() : "";
    if (assignedAdminId) {
      try {
        const uDoc = await adminDb.collection("users").doc(assignedAdminId).get();
        if (uDoc.exists) {
          const uData = uDoc.data() || {};
          const currentAssigned = Array.isArray(uData.assignedRobots) ? (uData.assignedRobots as string[]) : [];
          const updatedAssigned = currentAssigned.filter((r) => r !== actualRobotId && r !== robotId);
          await uDoc.ref.set({ assignedRobots: updatedAssigned, updatedAt: now }, { merge: true });
        }
      } catch (uErr) {
        console.warn(`[admin/robots DELETE] Failed unbinding from admin ${assignedAdminId}:`, uErr);
      }
    }

    // 3. Archive audit record into decommissioned_robots collection to preserve audit trail
    try {
      await adminDb.collection("decommissioned_robots").doc(actualRobotId).set({
        ...robotData,
        decommissionedAt: now,
        decommissionedBy: authResult.principal.uid,
        decommissionedByRole: authResult.principal.role,
        reason: "DELETED_BY_SUPERADMIN",
      });
    } catch (archErr) {
      console.warn(`[admin/robots DELETE] Failed archiving decommissioned robot ${actualRobotId}:`, archErr);
    }

    // 4. Delete robot document from active robots collection
    // Note: Person profiles and biometric templates in persons/face_embeddings are NEVER touched
    await robotRef.delete();

    return NextResponse.json({
      success: true,
      message: `Robot ${actualRobotId} successfully removed from active robot inventory.`,
      robotId: actualRobotId,
    });
  } catch (err: unknown) {
    return NextResponse.json(
      { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to delete robot." },
      { status: 500 }
    );
  }
}
