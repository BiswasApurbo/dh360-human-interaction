import { NextResponse } from "next/server";
import { authenticateFirebaseToken } from "@/lib/dh360/auth";
import {
  forwardAdminRequestToBackend,
  getAdminAuth,
  getAdminFirestore,
  hasServerAdminCredentials,
} from "@/lib/server-admin";

const DEFAULT_ADMIN_PERMS = {
  dashboard: true,
  robots: true,
  assistant: true,
  people: true,
  knowledgeBase: true,
  telemetry: true,
  settings: false,
};

export async function GET(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  // Enforce mandatory password change for bootstrap accounts
  if (authResult.principal.mustChangePassword) {
    return NextResponse.json(
      {
        error: "PASSWORD_CHANGE_REQUIRED",
        message: "Mandatory password change required before accessing administrative endpoints.",
      },
      { status: 403 }
    );
  }

  // Only superAdmin can access admin management
  if (authResult.principal.role !== "superAdmin") {
    return NextResponse.json(
      { error: "FORBIDDEN", message: "Only Super Admin can manage administrative accounts." },
      { status: 403 }
    );
  }

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend("/admin/users", "GET");
      const data = await res.json();
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "INTERNAL_ERROR", message: err instanceof Error ? err.message : "Failed to retrieve accounts." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = await getAdminAuth();
    const adminDb = await getAdminFirestore();

    // 1. Fetch users from Firebase Authentication (resilient try/catch)
    let authUsers: Array<{
      uid: string;
      email?: string;
      displayName?: string;
      disabled?: boolean;
      customClaims?: Record<string, unknown>;
      metadata?: { creationTime?: string; lastSignInTime?: string };
    }> = [];

    try {
      const listUsersResult = await adminAuth.listUsers(100);
      authUsers = listUsersResult.users as typeof authUsers;
    } catch (authErr) {
      console.warn("adminAuth.listUsers failed; proceeding with Firestore users fallback:", authErr);
    }

    // 2. Fetch users metadata from Firestore
    const firestoreDocs = await adminDb.collection("users").get();
    const firestoreMap = new Map<string, Record<string, unknown>>();
    firestoreDocs.forEach((doc: { id: string; data: () => Record<string, unknown> }) => {
      firestoreMap.set(doc.id, doc.data());
    });

    // 3. Fetch authoritative robot assignments from 'robots' collection
    const robotsDocs = await adminDb.collection("robots").get();
    const robotAssignmentsMap = new Map<string, string[]>();
    robotsDocs.forEach((doc: { id: string; data: () => Record<string, unknown> }) => {
      const data = doc.data();
      const adminId = typeof data.assignedAdminId === "string" ? data.assignedAdminId.trim() : null;
      if (adminId) {
        const existing = robotAssignmentsMap.get(adminId) || [];
        if (!existing.includes(doc.id)) {
          existing.push(doc.id);
        }
        robotAssignmentsMap.set(adminId, existing);
      }
    });

    const seenUids = new Set<string>();
    const accounts: Array<{
      uid: string;
      name: string;
      email: string;
      role: "superAdmin" | "admin";
      active: boolean;
      mustChangePassword: boolean;
      permissions: Record<string, boolean>;
      assignedRobots: string[];
      createdAt?: string;
      lastSignInTime?: string;
    }> = [];

    // Merge Auth users first
    for (const u of authUsers) {
      seenUids.add(u.uid);
      const claims = (u.customClaims as Record<string, unknown>) || {};
      const fsData = firestoreMap.get(u.uid) || {};

      const email = u.email || (fsData.email as string) || "";
      const isSuperEmail =
        email.toLowerCase() === "superadmin@gmail.com" ||
        email.toLowerCase() === "apurbobiswas32@gmail.com";
      const role = (claims.role || fsData.role || (isSuperEmail ? "superAdmin" : "admin")) as
        | "superAdmin"
        | "admin";
      const active = !u.disabled && fsData.active !== false;
      const permissions =
        role === "superAdmin"
          ? {
              dashboard: true,
              robots: true,
              assistant: true,
              people: true,
              knowledgeBase: true,
              telemetry: true,
              settings: true,
            }
          : (claims.permissions as Record<string, boolean>) ||
            (fsData.permissions as Record<string, boolean>) ||
            DEFAULT_ADMIN_PERMS;

      const docRobots = Array.isArray(fsData.assignedRobots) ? (fsData.assignedRobots as string[]) : [];
      const fromRobots = robotAssignmentsMap.get(u.uid) || [];
      const assignedRobots = Array.from(new Set([...docRobots, ...fromRobots]));

      accounts.push({
        uid: u.uid,
        name: u.displayName || (fsData.name as string) || email.split("@")[0] || "User",
        email,
        role,
        active,
        mustChangePassword: Boolean(claims.mustChangePassword || fsData.mustChangePassword),
        permissions,
        assignedRobots,
        createdAt: u.metadata?.creationTime,
        lastSignInTime: u.metadata?.lastSignInTime,
      });
    }

    // Merge any Firestore user records not present in authUsers
    for (const [uid, fsData] of firestoreMap.entries()) {
      if (seenUids.has(uid)) continue;
      seenUids.add(uid);

      const email = (fsData.email as string) || "";
      const isSuperEmail =
        email.toLowerCase() === "superadmin@gmail.com" ||
        email.toLowerCase() === "apurbobiswas32@gmail.com";
      const role = ((fsData.role as string) || (isSuperEmail ? "superAdmin" : "admin")) as
        | "superAdmin"
        | "admin";
      const active = fsData.active !== false;
      const permissions =
        role === "superAdmin"
          ? {
              dashboard: true,
              robots: true,
              assistant: true,
              people: true,
              knowledgeBase: true,
              telemetry: true,
              settings: true,
            }
          : (fsData.permissions as Record<string, boolean>) || DEFAULT_ADMIN_PERMS;

      const docRobots = Array.isArray(fsData.assignedRobots) ? (fsData.assignedRobots as string[]) : [];
      const fromRobots = robotAssignmentsMap.get(uid) || [];
      const assignedRobots = Array.from(new Set([...docRobots, ...fromRobots]));

      accounts.push({
        uid,
        name: (fsData.name as string) || email.split("@")[0] || "User",
        email,
        role,
        active,
        mustChangePassword: Boolean(fsData.mustChangePassword),
        permissions,
        assignedRobots,
        createdAt: typeof fsData.createdAt === "string" ? fsData.createdAt : undefined,
        lastSignInTime: typeof fsData.lastSignInTime === "string" ? fsData.lastSignInTime : undefined,
      });
    }

    return NextResponse.json({ users: accounts });
  } catch (err: unknown) {
    return NextResponse.json(
      {
        error: "INTERNAL_ERROR",
        message: err instanceof Error ? err.message : "Failed to retrieve accounts.",
      },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  const authHeader = request.headers.get("authorization");
  const authResult = await authenticateFirebaseToken(authHeader, { checkRevoked: true });

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
    );
  }

  // Enforce mandatory password change for bootstrap accounts
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
      { error: "FORBIDDEN", message: "Only Super Admin can create Admin accounts." },
      { status: 403 }
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "BAD_REQUEST", message: "Invalid JSON body." }, { status: 400 });
  }

  const name = typeof body.name === "string" ? body.name.trim() : "";
  const email = typeof body.email === "string" ? body.email.trim().toLowerCase() : "";
  const password = typeof body.password === "string" ? body.password : "";
  const permissionsInput = (body.permissions as Record<string, boolean>) || {};
  const assignedRobots = Array.isArray(body.assignedRobots)
    ? (body.assignedRobots as string[]).filter((r): r is string => typeof r === "string" && Boolean(r.trim()))
    : [];

  if (!email || !email.includes("@")) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "Valid email is required." }, { status: 400 });
  }
  if (!password || password.length < 6) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "Password must be at least 6 characters." }, { status: 400 });
  }
  if (!name) {
    return NextResponse.json({ error: "INVALID_ARGUMENT", message: "Name is required." }, { status: 400 });
  }

  // Merge permissions
  const permissions = {
    ...DEFAULT_ADMIN_PERMS,
    ...permissionsInput,
  };

  if (!hasServerAdminCredentials()) {
    try {
      const res = await forwardAdminRequestToBackend("/admin/users", "POST", {
        name,
        email,
        password,
        role: "admin",
        permissions,
        assignedRobots,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        const message =
          typeof data.detail === "string"
            ? data.detail
            : Array.isArray(data.detail)
              ? data.detail.map((d: { msg?: string }) => d.msg || "Validation error").join("; ")
              : (data.message || "Failed to create Admin account.");
        return NextResponse.json(
          { error: data.error || (res.status === 409 ? "ALREADY_EXISTS" : "CREATION_FAILED"), message },
          { status: res.status }
        );
      }
      return NextResponse.json(data, { status: res.status });
    } catch (err: unknown) {
      return NextResponse.json(
        { error: "CREATION_FAILED", message: err instanceof Error ? err.message : "Failed to create Admin account." },
        { status: 500 }
      );
    }
  }

  try {
    const adminAuth = await getAdminAuth();
    const adminDb = await getAdminFirestore();

    // 1. Create user in Firebase Authentication
    const newUser = await adminAuth.createUser({
      email,
      password,
      displayName: name,
    });

    // 2. Set custom claims for RBAC and mandatory first password change
    await adminAuth.setCustomUserClaims(newUser.uid, {
      role: "admin",
      permissions,
      mustChangePassword: true,
    });

    // 3. Persist in Firestore
    await adminDb.collection("users").doc(newUser.uid).set({
      name,
      email,
      role: "admin",
      active: true,
      permissions,
      assignedRobots,
      mustChangePassword: true,
      createdAt: new Date().toISOString(),
      createdBy: authResult.principal.uid,
    });

    // 4. Update assigned robots
    for (const robotId of assignedRobots) {
      try {
        await adminDb.collection("robots").doc(robotId).set(
          {
            assignedAdminId: newUser.uid,
            assignedAdminName: name,
          },
          { merge: true }
        );
      } catch (rErr) {
        console.warn(`[admin/users] Failed to associate robot ${robotId}:`, rErr);
      }
    }

    return NextResponse.json(
      {
        user: {
          uid: newUser.uid,
          name,
          email,
          role: "admin",
          active: true,
          permissions,
          assignedRobots,
          mustChangePassword: true,
        },
      },
      { status: 201 }
    );
  } catch (err: unknown) {
    const code = (err as { code?: string })?.code;
    if (code === "auth/email-already-exists") {
      return NextResponse.json(
        { error: "ALREADY_EXISTS", message: "An account with this email address already exists." },
        { status: 409 }
      );
    }
    return NextResponse.json(
      {
        error: "CREATION_FAILED",
        message: err instanceof Error ? err.message : "Failed to create Admin account.",
      },
      { status: 500 }
    );
  }
}
