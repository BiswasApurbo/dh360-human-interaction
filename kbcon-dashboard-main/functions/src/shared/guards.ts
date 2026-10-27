import { HttpsError, type CallableRequest } from "firebase-functions/https";

export function requireSuperAdmin(request: CallableRequest<unknown>): string {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "লগইন করতে হবে");
  }
  if (request.auth.token.role !== "superAdmin") {
    throw new HttpsError(
      "permission-denied",
      "শুধু Super Admin এই কাজ করতে পারবেন",
    );
  }
  return request.auth.uid;
}

export function requireAdminOrSuper(request: CallableRequest<unknown>): {
  uid: string;
  role: "admin" | "superAdmin";
} {
  if (!request.auth) {
    throw new HttpsError("unauthenticated", "লগইন করতে হবে");
  }
  const role = request.auth.token.role;
  if (role !== "admin" && role !== "superAdmin") {
    throw new HttpsError("permission-denied", "অনুমতি নেই");
  }
  return { uid: request.auth.uid, role };
}
