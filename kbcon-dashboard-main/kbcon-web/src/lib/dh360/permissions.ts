/**
 * Explicit Route Allowlist & RBAC Permissions for DH360 BFF.
 *
 * Implements strict deny-by-default access control:
 * 1. Checks that the requested path and HTTP method match an allowed route.
 * 2. Checks that the caller's role meets or exceeds the required role.
 * 3. Enforces payload-level restrictions (e.g. consent_status updates require superAdmin).
 * 4. Protects destructive operations (delete person, reset biometrics) for superAdmin only.
 */

import type { AuthPrincipal } from "./auth.ts";

export type HttpMethod = "GET" | "POST" | "PATCH" | "DELETE" | "PUT";

export interface RouteRule {
  name: string;
  pattern: RegExp;
  allowedMethods: HttpMethod[];
  minRole: "admin" | "superAdmin";
  scope: string;
  customCheck?: (
    body: Record<string, unknown> | null,
    principal: AuthPrincipal
  ) => { allowed: boolean; reason?: string };
}

export interface PermissionCheckResult {
  matched: boolean;
  allowed: boolean;
  httpStatus?: 403 | 404 | 405;
  reason?: string;
  rule?: RouteRule;
}

/**
 * Ordered list of allowed route patterns.
 */
export const ALLOWED_DH360_RULES: RouteRule[] = [
  // Health & Runtime State
  {
    name: "health",
    pattern: /^\/health\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "system:read",
  },
  {
    name: "runtime_state",
    pattern: /^\/runtime\/state\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "system:read",
  },
  {
    name: "runtime_metrics",
    pattern: /^\/runtime\/metrics\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "system:read",
  },

  // Live Interaction Context & Events
  {
    name: "interaction_context",
    pattern: /^\/interaction\/context\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
  {
    name: "interaction_summary_latest",
    pattern: /^\/interaction\/summary\/latest\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
  {
    name: "interaction_events",
    pattern: /^\/interaction\/events\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },

  // Person Profiles: Collection routes
  {
    name: "persons_collection",
    pattern: /^\/persons\/?$/,
    allowedMethods: ["GET", "POST"],
    minRole: "admin",
    scope: "persons:read", // and persons:create for POST
  },

  // Enrollment Session Routes
  {
    name: "enrollment_preview",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/enrollment\/sessions\/preview\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "enrollment:manage",
  },
  {
    name: "enrollment_session_step",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/enrollment\/sessions\/step\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "enrollment:manage",
  },
  {
    name: "enrollment_session_commit",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/enrollment\/sessions\/commit\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "enrollment:manage",
  },
  {
    name: "enrollment_session_cancel",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/enrollment\/sessions\/cancel\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "enrollment:manage",
  },
  {
    name: "enrollment_session_person",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/enrollment\/sessions\/?$/,
    allowedMethods: ["GET", "POST"],
    minRole: "admin",
    scope: "enrollment:manage",
  },
  {
    name: "enrollment_session_current_cancel",
    pattern: /^\/persons\/enrollment\/sessions\/current\/?$/,
    allowedMethods: ["DELETE"],
    minRole: "admin",
    scope: "enrollment:manage",
  },

  // Biometrics Reset: Destructive operation strictly limited to Super Admin
  {
    name: "person_biometrics_reset",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/biometrics\/?$/,
    allowedMethods: ["DELETE"],
    minRole: "superAdmin",
    scope: "biometrics:reset",
  },

  // Person Profile by ID: Read, Update, Delete
  {
    name: "person_entity",
    pattern: /^\/persons\/([A-Za-z0-9_-]+)\/?$/,
    allowedMethods: ["GET", "PATCH", "DELETE"],
    minRole: "admin",
    scope: "persons:read",
    customCheck: (body, principal) => {
      // 1. Person deletion is strictly limited to superAdmin
      // (Handled by method check or here)
      // 2. Consent modification strictly requires superAdmin
      if (body && Object.prototype.hasOwnProperty.call(body, "consent_status")) {
        if (principal.role !== "superAdmin") {
          return {
            allowed: false,
            reason: "Only superAdmin is permitted to modify person consent_status.",
          };
        }
      }
      return { allowed: true };
    },
  },

  // Authorization Sessions
  {
    name: "authorization_sessions",
    pattern: /^\/authorization\/sessions\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "authorization:manage",
  },
  {
    name: "authorization_session_current",
    pattern: /^\/authorization\/sessions\/current\/?$/,
    allowedMethods: ["GET", "DELETE"],
    minRole: "admin",
    scope: "authorization:manage",
  },
  {
    name: "authorization_session_current_cancel",
    pattern: /^\/authorization\/sessions\/current\/cancel\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "authorization:manage",
  },
  {
    name: "authorization_logout",
    pattern: /^\/authorization\/logout\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "authorization:manage",
  },
  {
    name: "authorization_session_context",
    pattern: /^\/authorization\/sessions\/([A-Za-z0-9_-]+)\/context\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "authorization:manage",
  },

  // Conversational Dialogue & Voice
  {
    name: "dialogue_turn",
    pattern: /^\/dialogue\/turn\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "dialogue:interact",
  },
  {
    name: "dialogue_turn_voice",
    pattern: /^\/dialogue\/turn\/voice\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "dialogue:interact",
  },
  {
    name: "dialogue_interrupt",
    pattern: /^\/dialogue\/interrupt\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "dialogue:interact",
  },

  // Knowledge Base Management (CRUD & Synchronization)
  {
    name: "knowledge_items_collection",
    pattern: /^\/knowledge\/items\/?$/,
    allowedMethods: ["GET", "POST"],
    minRole: "admin",
    scope: "knowledge:manage",
  },
  {
    name: "knowledge_item_entity",
    pattern: /^\/knowledge\/items\/([A-Za-z0-9_-]+)\/?$/,
    allowedMethods: ["GET", "PUT", "PATCH", "DELETE"],
    minRole: "admin",
    scope: "knowledge:manage",
  },
  {
    name: "knowledge_reload",
    pattern: /^\/knowledge\/reload\/?$/,
    allowedMethods: ["POST"],
    minRole: "admin",
    scope: "knowledge:manage",
  },

  // System Settings & Institution Configuration
  {
    name: "settings_institution",
    pattern: /^\/settings\/institution\/?$/,
    allowedMethods: ["GET", "PUT", "POST"],
    minRole: "admin",
    scope: "settings:manage",
  },

  // Session History & Analytics
  {
    name: "sessions_collection",
    pattern: /^\/sessions\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
  {
    name: "sessions_count",
    pattern: /^\/sessions\/count\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
  {
    name: "sessions_analytics",
    pattern: /^\/sessions\/analytics\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
  {
    name: "sessions_entity",
    pattern: /^\/sessions\/([A-Za-z0-9_-]+)\/?$/,
    allowedMethods: ["GET"],
    minRole: "admin",
    scope: "interaction:read",
  },
];

/**
 * Check if the caller has permission for the given path, method, and request body.
 */
export function checkPermission(
  pathname: string,
  method: string,
  principal: AuthPrincipal,
  body: Record<string, unknown> | null = null
): PermissionCheckResult {
  const normPath = "/" + pathname.replace(/^\/+|\/+$/g, "");
  const normMethod = method.toUpperCase() as HttpMethod;

  // Find matching rule
  let matchedRule: RouteRule | null = null;
  for (const rule of ALLOWED_DH360_RULES) {
    if (rule.pattern.test(normPath)) {
      matchedRule = rule;
      break;
    }
  }

  if (!matchedRule) {
    return {
      matched: false,
      allowed: false,
      httpStatus: 404,
      reason: `The path '${normPath}' is not an approved DH360 operation.`,
    };
  }

  if (!matchedRule.allowedMethods.includes(normMethod)) {
    return {
      matched: true,
      allowed: false,
      httpStatus: 405,
      reason: `Method '${normMethod}' is not permitted for '${normPath}'.`,
      rule: matchedRule,
    };
  }

  // Check destructive operations on person_entity
  if (matchedRule.name === "person_entity" && normMethod === "DELETE") {
    if (principal.role !== "superAdmin") {
      return {
        matched: true,
        allowed: false,
        httpStatus: 403,
        reason: "Person profile deletion is restricted to superAdmin.",
        rule: matchedRule,
      };
    }
  }

  // Check role hierarchy (superAdmin can do anything admin can do)
  if (matchedRule.minRole === "superAdmin" && principal.role !== "superAdmin") {
    return {
      matched: true,
      allowed: false,
      httpStatus: 403,
      reason: `Operation '${matchedRule.name}' requires superAdmin role.`,
      rule: matchedRule,
    };
  }

  // Custom checks (e.g. consent_status payload check)
  if (matchedRule.customCheck) {
    const customResult = matchedRule.customCheck(body, principal);
    if (!customResult.allowed) {
      return {
        matched: true,
        allowed: false,
        httpStatus: 403,
        reason: customResult.reason || "Forbidden by fine-grained security policy.",
        rule: matchedRule,
      };
    }
  }

  return {
    matched: true,
    allowed: true,
    rule: matchedRule,
  };
}
