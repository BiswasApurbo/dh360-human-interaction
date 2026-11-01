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
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
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

    // 1. Fetch users from Firebase Authentication
    const listUsersResult = await adminAuth.listUsers(100);

    // 2. Fetch users metadata from Firestore
    const firestoreDocs = await adminDb.collection("users").get();
    const firestoreMap = new Map<string, Record<string, unknown>>();
    firestoreDocs.forEach((doc) => {
      firestoreMap.set(doc.id, doc.data());
    });

    const accounts = listUsersResult.users.map((u) => {
      const claims = (u.customClaims as Record<string, unknown>) || {};
      const fsData = firestoreMap.get(u.uid) || {};

      const role = claims.role || fsData.role || (u.email === "superadmin@gmail.com" ? "superAdmin" : "admin");
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

      return {
        uid: u.uid,
        name: u.displayName || (fsData.name as string) || u.email?.split("@")[0] || "User",
        email: u.email || "",
        role: role as "superAdmin" | "admin",
        active,
        mustChangePassword: Boolean(claims.mustChangePassword || fsData.mustChangePassword),
        permissions,
        createdAt: u.metadata.creationTime,
        lastSignInTime: u.metadata.lastSignInTime,
      };
    });

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
  const authResult = await authenticateFirebaseToken(authHeader);

  if (!authResult.success) {
    return NextResponse.json(
      { error: authResult.code, message: authResult.message },
      { status: authResult.status }
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
      });
      const data = await res.json();
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

    // 2. Set custom claims for RBAC
    await adminAuth.setCustomUserClaims(newUser.uid, {
      role: "admin",
      permissions,
    });

    // 3. Persist in Firestore
    await adminDb.collection("users").doc(newUser.uid).set({
      name,
      email,
      role: "admin",
      active: true,
      permissions,
      createdAt: new Date().toISOString(),
      createdBy: authResult.principal.uid,
    });

    return NextResponse.json(
      {
        user: {
          uid: newUser.uid,
          name,
          email,
          role: "admin",
          active: true,
          permissions,
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
