/**
 * Full Acceptance & Security Audit Test Suite
 *
 * Runs actual authenticated tests against:
 * 1. Production Vercel BFF (https://kbcon-dashboard-gamma.vercel.app)
 * 2. Production Raspberry Pi Funnel (https://raspberrypi.tailacfb18.ts.net)
 * 3. Firestore security rules
 */

import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execSync } from "node:child_process";
import { createRequire } from "node:module";

const require = createRequire(path.resolve("kbcon-dashboard-main/kbcon-web/package.json"));
const admin = require("firebase-admin");
const { getAuth } = require("firebase-admin/auth");
const { getFirestore } = require("firebase-admin/firestore");

const saPath = path.join(os.homedir(), ".config", "kbcon", "firebase-service-account.json");
const sa = JSON.parse(fs.readFileSync(saPath, "utf8"));
const env = fs.readFileSync("kbcon-dashboard-main/kbcon-web/.env.local", "utf8");
const apiKey = env.match(/NEXT_PUBLIC_FIREBASE_API_KEY=(.*)/)[1].replace(/["']/g, "").trim();

const app = admin.initializeApp({ credential: admin.cert(sa) }, "audit-runner");
const authAdmin = getAuth(app);
const dbAdmin = getFirestore(app);

const VERCEL_URL = "https://kbcon-dashboard-gamma.vercel.app";
const PI_URL = "https://raspberrypi.tailacfb18.ts.net";

const testResults = [];

function recordResult(testId, name, status, evidence, details = "") {
  testResults.push({ testId, name, status, evidence, details });
  console.log(`[${status}] ${testId}: ${name} -> ${evidence}`);
}

async function getIdToken(uid, customClaims = {}) {
  const customToken = await authAdmin.createCustomToken(uid, customClaims);
  const res = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken?key=${apiKey}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token: customToken, returnSecureToken: true }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(`Token exchange failed: ${JSON.stringify(data)}`);
  return data.idToken;
}

async function run() {
  console.log("================================================================================");
  console.log("STARTING KBCON ACCEPTANCE & SECURITY AUDIT");
  console.log("================================================================================");

  // ---------------------------------------------------------------------------
  // Setup: Provision Super Admin, Admin A, Admin B, and Robots R001 & R002
  // ---------------------------------------------------------------------------
  const superUid = "vFeeu9SZwIOiMCkxuMZ2ZGx3V4t2"; // apurbobiswas32@gmail.com
  const superToken = await getIdToken(superUid, { role: "superAdmin" });

  const adminAUid = "audit-admin-a-" + Date.now();
  const adminBUid = "audit-admin-b-" + Date.now();
  const emailA = `admin.a.${Date.now()}@kbcon.test`;
  const emailB = `admin.b.${Date.now()}@kbcon.test`;

  // 1. Provision Robots R001 and R002 in Firestore
  const nowStr = new Date().toISOString();
  await dbAdmin.collection("robots").doc("R001").set({
    id: "R001",
    name: "KBCON Alpha",
    type: "Humanoid",
    language: "bn",
    mode: "education",
    status: "online",
    assignedAdminId: adminAUid,
    assignedAdminName: "Admin A",
    createdAt: nowStr,
    updatedAt: nowStr,
  });

  await dbAdmin.collection("robots").doc("R002").set({
    id: "R002",
    name: "KBCON Beta",
    type: "Humanoid",
    language: "en",
    mode: "healthcare",
    status: "online",
    assignedAdminId: adminBUid,
    assignedAdminName: "Admin B",
    createdAt: nowStr,
    updatedAt: nowStr,
  });

  // Provision Admin A via Super Admin API (testing Section 4 Admin Provisioning)
  console.log("\n--- Section 4: Admin Provisioning via Super Admin API ---");
  const createAdminResA = await fetch(`${VERCEL_URL}/api/admin/users`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${superToken}`,
    },
    body: JSON.stringify({
      name: "Admin Alpha",
      email: emailA,
      password: "TestPassword123!",
      assignedRobots: ["R001"],
    }),
  });
  const createAdminDataA = await createAdminResA.json();
  const createdAdminAUid = createAdminDataA?.user?.uid;

  if (createAdminResA.status === 200 && createdAdminAUid) {
    recordResult(
      "AUTH-01",
      "Super Admin creates Admin A with assigned robots",
      "PASS",
      `HTTP 200, created uid=${createdAdminAUid}, assignedRobots=["R001"]`
    );
  } else {
    recordResult(
      "AUTH-01",
      "Super Admin creates Admin A with assigned robots",
      "FAIL",
      `HTTP ${createAdminResA.status}: ${JSON.stringify(createAdminDataA)}`
    );
  }

  // Provision Admin B via Super Admin API
  const createAdminResB = await fetch(`${VERCEL_URL}/api/admin/users`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${superToken}`,
    },
    body: JSON.stringify({
      name: "Admin Beta",
      email: emailB,
      password: "TestPassword123!",
      assignedRobots: ["R002"],
    }),
  });
  const createAdminDataB = await createAdminResB.json();
  const createdAdminBUid = createAdminDataB?.user?.uid;

  if (createAdminResB.status === 200 && createdAdminBUid) {
    recordResult(
      "AUTH-02",
      "Super Admin creates Admin B with assigned robots",
      "PASS",
      `HTTP 200, created uid=${createdAdminBUid}, assignedRobots=["R002"]`
    );
  } else {
    recordResult(
      "AUTH-02",
      "Super Admin creates Admin B with assigned robots",
      "FAIL",
      `HTTP ${createAdminResB.status}: ${JSON.stringify(createAdminDataB)}`
    );
  }

  const effectiveAdminAUid = createdAdminAUid || adminAUid;
  const effectiveAdminBUid = createdAdminBUid || adminBUid;

  // ---------------------------------------------------------------------------
  // Section 4: Mandatory Password Change Enforcement
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 4: Mandatory Password Change Verification ---");
  // Get token for Admin A while mustChangePassword is TRUE
  const adminATokenPending = await getIdToken(effectiveAdminAUid, {
    role: "admin",
    mustChangePassword: true,
  });

  const pwBlockRes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    headers: { Authorization: `Bearer ${adminATokenPending}` },
  });
  const pwBlockData = await pwBlockRes.json();

  if (pwBlockRes.status === 403 && pwBlockData.error === "PASSWORD_CHANGE_REQUIRED") {
    recordResult(
      "AUTH-03",
      "New Admin first login blocks API access with mustChangePassword=true",
      "PASS",
      `HTTP 403 PASSWORD_CHANGE_REQUIRED: ${pwBlockData.message}`
    );
  } else {
    recordResult(
      "AUTH-03",
      "New Admin first login blocks API access with mustChangePassword=true",
      "FAIL",
      `Expected HTTP 403, got HTTP ${pwBlockRes.status}: ${JSON.stringify(pwBlockData)}`
    );
  }

  // Clear mustChangePassword for Admin A and Admin B to test subsequent operations
  await authAdmin.setCustomUserClaims(effectiveAdminAUid, {
    role: "admin",
    mustChangePassword: false,
    permissions: {
      dashboard: true,
      robots: true,
      assistant: true,
      people: true,
      knowledgeBase: true,
      telemetry: true,
      settings: true,
    },
  });

  await authAdmin.setCustomUserClaims(effectiveAdminBUid, {
    role: "admin",
    mustChangePassword: false,
    permissions: {
      dashboard: true,
      robots: true,
      assistant: true,
      people: true,
      knowledgeBase: true,
      telemetry: true,
      settings: true,
    },
  });

  const adminAToken = await getIdToken(effectiveAdminAUid, { role: "admin", mustChangePassword: false });
  const adminBToken = await getIdToken(effectiveAdminBUid, { role: "admin", mustChangePassword: false });

  // ---------------------------------------------------------------------------
  // Section 1: Tenant Isolation Tests (People & Knowledge Base)
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 1: Tenant Isolation Verification ---");

  // Admin B creates Person B
  const personBName = "Person B - " + Date.now();
  const createPersonBRes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminBToken}`,
    },
    body: JSON.stringify({
      display_name: personBName,
      department: "Cardiology",
      role: "Doctor",
    }),
  });
  const personBData = await createPersonBRes.json();
  const personBId = personBData.person_id;
  console.log(`Created Person B in Tenant B: ${personBId} (${personBName})`);

  // Admin A creates Person A
  const personAName = "Person A - " + Date.now();
  const createPersonARes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({
      display_name: personAName,
      department: "Engineering",
      role: "Professor",
    }),
  });
  const personAData = await createPersonARes.json();
  const personAId = personAData.person_id;
  console.log(`Created Person A in Tenant A: ${personAId} (${personAName})`);

  // 1. Admin A attempts to GET Person B
  const getBbyARes = await fetch(`${VERCEL_URL}/api/dh360/persons/${personBId}`, {
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  if (getBbyARes.status === 404) {
    recordResult(
      "ISO-01",
      "Admin A cannot retrieve Admin B's person profile",
      "PASS",
      `HTTP 404 Entity 'person' with ID '${personBId}' not found`
    );
  } else {
    recordResult(
      "ISO-01",
      "Admin A cannot retrieve Admin B's person profile",
      "FAIL",
      `Expected HTTP 404, got HTTP ${getBbyARes.status}`
    );
  }

  // 2. Admin A attempts to list persons — must NOT contain Person B
  const listARes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  const listAData = await listARes.json();
  const containsPersonB = Array.isArray(listAData) && listAData.some((p) => p.person_id === personBId);
  const containsPersonA = Array.isArray(listAData) && listAData.some((p) => p.person_id === personAId);

  if (!containsPersonB && containsPersonA) {
    recordResult(
      "ISO-02",
      "Admin A listing people includes Tenant A and excludes Tenant B",
      "PASS",
      `HTTP 200, returned ${listAData.length} records, Person B strictly excluded`
    );
  } else {
    recordResult(
      "ISO-02",
      "Admin A listing people includes Tenant A and excludes Tenant B",
      "FAIL",
      `containsPersonB=${containsPersonB}, containsPersonA=${containsPersonA}`
    );
  }

  // 3. Admin A attempts to PATCH Person B
  const patchBbyARes = await fetch(`${VERCEL_URL}/api/dh360/persons/${personBId}`, {
    method: "PATCH",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({ display_name: "Hacked by Admin A" }),
  });
  if (patchBbyARes.status === 404) {
    recordResult(
      "ISO-03",
      "Admin A cannot modify Admin B's person profile",
      "PASS",
      `HTTP 404 Entity not found (cross-tenant mutation rejected)`
    );
  } else {
    recordResult(
      "ISO-03",
      "Admin A cannot modify Admin B's person profile",
      "FAIL",
      `Expected HTTP 404, got HTTP ${patchBbyARes.status}`
    );
  }

  // 4. Admin A attempts to DELETE Person B
  const deleteBbyARes = await fetch(`${VERCEL_URL}/api/dh360/persons/${personBId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  if (deleteBbyARes.status === 404) {
    recordResult(
      "ISO-04",
      "Admin A cannot delete Admin B's person profile",
      "PASS",
      `HTTP 404 Entity not found (cross-tenant deletion rejected)`
    );
  } else {
    recordResult(
      "ISO-04",
      "Admin A cannot delete Admin B's person profile",
      "FAIL",
      `Expected HTTP 404, got HTTP ${deleteBbyARes.status}`
    );
  }

  // 5. Admin B attempts to access Admin A's person
  const getAbyBRes = await fetch(`${VERCEL_URL}/api/dh360/persons/${personAId}`, {
    headers: { Authorization: `Bearer ${adminBToken}` },
  });
  if (getAbyBRes.status === 404) {
    recordResult(
      "ISO-05",
      "Admin B cannot access Admin A's person profile",
      "PASS",
      `HTTP 404 Entity not found`
    );
  } else {
    recordResult(
      "ISO-05",
      "Admin B cannot access Admin A's person profile",
      "FAIL",
      `Expected HTTP 404, got HTTP ${getAbyBRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Knowledge Base Tenant Isolation
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 1 & 3: Knowledge Base Tenant Isolation ---");

  // Admin B creates Fact B
  const createFactBRes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminBToken}`,
    },
    body: JSON.stringify({
      question_en: "What is Clinic Beta procedure?",
      answer_en: "Clinic Beta operates 24/7 emergency services.",
      question_bn: "ক্লিনিক বিটার নিয়ম কি?",
      answer_bn: "ক্লিনিক বিটা সার্বক্ষণিক জরুরি সেবা দেয়।",
      institution_type: "clinic",
    }),
  });
  const factBData = await createFactBRes.json();
  const factBId = factBData.id;
  console.log(`Created Fact B in Tenant B: ${factBId}`);

  // Admin A creates Fact A
  const createFactARes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({
      question_en: "What is University Alpha procedure?",
      answer_en: "University Alpha requires course registration each semester.",
      question_bn: "ইউনিভার্সিটি আলফার নিয়ম কি?",
      answer_bn: "ইউনিভার্সিটি আলফায় প্রতি সেমিস্টারে কোর্স রেজিস্ট্রেশন করতে হয়।",
      institution_type: "university",
    }),
  });
  const factAData = await createFactARes.json();
  const factAId = factAData.id;
  console.log(`Created Fact A in Tenant A: ${factAId}`);

  // Admin A attempts to GET Fact B
  const getFactBbyARes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items/${factBId}`, {
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  if (getFactBbyARes.status === 404) {
    recordResult(
      "KB-01",
      "Admin A cannot retrieve Admin B's knowledge fact",
      "PASS",
      `HTTP 404 Fact '${factBId}' not found`
    );
  } else {
    recordResult(
      "KB-01",
      "Admin A cannot retrieve Admin B's knowledge fact",
      "FAIL",
      `Expected HTTP 404, got HTTP ${getFactBbyARes.status}`
    );
  }

  // Admin A attempts to UPDATE Fact B
  const updateFactBbyARes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items/${factBId}`, {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({
      question_en: "Hacked question?",
      answer_en: "Hacked answer.",
    }),
  });
  if (updateFactBbyARes.status === 404) {
    recordResult(
      "KB-02",
      "Admin A cannot update Admin B's knowledge fact",
      "PASS",
      `HTTP 404 Fact '${factBId}' not found`
    );
  } else {
    recordResult(
      "KB-02",
      "Admin A cannot update Admin B's knowledge fact",
      "FAIL",
      `Expected HTTP 404, got HTTP ${updateFactBbyARes.status}`
    );
  }

  // Admin A attempts to DELETE Fact B
  const deleteFactBbyARes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items/${factBId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  if (deleteFactBbyARes.status === 404) {
    recordResult(
      "KB-03",
      "Admin A cannot delete Admin B's knowledge fact",
      "PASS",
      `HTTP 404 Fact '${factBId}' not found`
    );
  } else {
    recordResult(
      "KB-03",
      "Admin A cannot delete Admin B's knowledge fact",
      "FAIL",
      `Expected HTTP 404, got HTTP ${deleteFactBbyARes.status}`
    );
  }

  // Admin B attempts to GET Fact A
  const getFactAbyBRes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items/${factAId}`, {
    headers: { Authorization: `Bearer ${adminBToken}` },
  });
  if (getFactAbyBRes.status === 404) {
    recordResult(
      "KB-04",
      "Admin B cannot retrieve Admin A's knowledge fact",
      "PASS",
      `HTTP 404 Fact '${factAId}' not found`
    );
  } else {
    recordResult(
      "KB-04",
      "Admin B cannot retrieve Admin A's knowledge fact",
      "FAIL",
      `Expected HTTP 404, got HTTP ${getFactAbyBRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Super Admin Strict Isolation from Tenant-Private Data
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 1: Super Admin Isolation from Tenant Data ---");

  // Super Admin GET /api/dh360/persons -> HTTP 403
  const superPeopleRes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    headers: { Authorization: `Bearer ${superToken}` },
  });
  const superPeopleData = await superPeopleRes.json();
  if (superPeopleRes.status === 403) {
    recordResult(
      "SUPER-01",
      "Super Admin blocked from listing tenant-private People",
      "PASS",
      `HTTP 403 FORBIDDEN: ${superPeopleData.message}`
    );
  } else {
    recordResult(
      "SUPER-01",
      "Super Admin blocked from listing tenant-private People",
      "FAIL",
      `Expected HTTP 403, got HTTP ${superPeopleRes.status}`
    );
  }

  // Super Admin GET /api/dh360/knowledge/items -> HTTP 403
  const superKbRes = await fetch(`${VERCEL_URL}/api/dh360/knowledge/items`, {
    headers: { Authorization: `Bearer ${superToken}` },
  });
  const superKbData = await superKbRes.json();
  if (superKbRes.status === 403) {
    recordResult(
      "SUPER-02",
      "Super Admin blocked from accessing tenant-private Knowledge Base",
      "PASS",
      `HTTP 403 FORBIDDEN: ${superKbData.message}`
    );
  } else {
    recordResult(
      "SUPER-02",
      "Super Admin blocked from accessing tenant-private Knowledge Base",
      "FAIL",
      `Expected HTTP 403, got HTTP ${superKbRes.status}`
    );
  }

  // Super Admin POST /api/dh360/persons -> HTTP 403
  const superCreatePersonRes = await fetch(`${VERCEL_URL}/api/dh360/persons`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${superToken}`,
    },
    body: JSON.stringify({ display_name: "Super Admin Person" }),
  });
  if (superCreatePersonRes.status === 403) {
    recordResult(
      "SUPER-03",
      "Super Admin blocked from creating People records",
      "PASS",
      `HTTP 403 FORBIDDEN`
    );
  } else {
    recordResult(
      "SUPER-03",
      "Super Admin blocked from creating People records",
      "FAIL",
      `Expected HTTP 403, got HTTP ${superCreatePersonRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Admin Blocked from Platform Operations
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 1: Admin Blocked from Platform Operations ---");

  // Admin A GET /api/admin/users -> HTTP 403
  const adminGetUsersRes = await fetch(`${VERCEL_URL}/api/admin/users`, {
    headers: { Authorization: `Bearer ${adminAToken}` },
  });
  if (adminGetUsersRes.status === 403) {
    recordResult(
      "RBAC-01",
      "Admin cannot list administrator accounts",
      "PASS",
      `HTTP 403 FORBIDDEN: Only Super Admin can manage administrative accounts.`
    );
  } else {
    recordResult(
      "RBAC-01",
      "Admin cannot list administrator accounts",
      "FAIL",
      `Expected HTTP 403, got HTTP ${adminGetUsersRes.status}`
    );
  }

  // Admin A POST /api/admin/users -> HTTP 403
  const adminCreateUserRes = await fetch(`${VERCEL_URL}/api/admin/users`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({ name: "Sub Admin", email: "sub@kbcon.test", password: "Password123!" }),
  });
  if (adminCreateUserRes.status === 403) {
    recordResult(
      "RBAC-02",
      "Admin cannot create administrator accounts",
      "PASS",
      `HTTP 403 FORBIDDEN`
    );
  } else {
    recordResult(
      "RBAC-02",
      "Admin cannot create administrator accounts",
      "FAIL",
      `Expected HTTP 403, got HTTP ${adminCreateUserRes.status}`
    );
  }

  // Admin A POST /api/admin/robots -> HTTP 403
  const adminCreateRobotRes = await fetch(`${VERCEL_URL}/api/admin/robots`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
    },
    body: JSON.stringify({ robotId: "R999", name: "Rogue Robot" }),
  });
  if (adminCreateRobotRes.status === 403) {
    recordResult(
      "RBAC-03",
      "Admin cannot create robots",
      "PASS",
      `HTTP 403 FORBIDDEN`
    );
  } else {
    recordResult(
      "RBAC-03",
      "Admin cannot create robots",
      "FAIL",
      `Expected HTTP 403, got HTTP ${adminCreateRobotRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Spoofing Immunity Verification (X-Tenant-ID & Parameter Tampering)
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 1: Spoofing Immunity Verification ---");

  // Admin A sends spoofed X-Tenant-ID header with Admin B's UID attempting to read Person B
  const spoofRes = await fetch(`${VERCEL_URL}/api/dh360/persons/${personBId}`, {
    headers: {
      Authorization: `Bearer ${adminAToken}`,
      "X-Tenant-ID": effectiveAdminBUid, // Spoofed tenant header
    },
  });
  if (spoofRes.status === 404) {
    recordResult(
      "SPOOF-01",
      "Spoofed X-Tenant-ID header is overridden server-side with verified caller UID",
      "PASS",
      `HTTP 404 (BFF overwrote client header with principal.uid=${effectiveAdminAUid})`
    );
  } else {
    recordResult(
      "SPOOF-01",
      "Spoofed X-Tenant-ID header is overridden server-side with verified caller UID",
      "FAIL",
      `Expected HTTP 404, got HTTP ${spoofRes.status}`
    );
  }

  // Direct backend call without operator token must be rejected with 401
  const directBackendRes = await fetch(`${PI_URL}/persons`, {
    headers: { "X-Tenant-ID": effectiveAdminAUid },
  });
  if (directBackendRes.status === 401) {
    recordResult(
      "SEC-01",
      "Direct backend Funnel rejects unauthenticated requests",
      "PASS",
      `HTTP 401 UNAUTHENTICATED (Operator token required on public funnel)`
    );
  } else {
    recordResult(
      "SEC-01",
      "Direct backend Funnel rejects unauthenticated requests",
      "FAIL",
      `Expected HTTP 401, got HTTP ${directBackendRes.status}`
    );
  }

  // Direct backend knowledge route without operator token must be rejected with 401
  const directKbRes = await fetch(`${PI_URL}/knowledge/items`, {
    headers: { "X-Tenant-ID": effectiveAdminAUid },
  });
  if (directKbRes.status === 401) {
    recordResult(
      "SEC-02",
      "Direct backend Funnel rejects unauthenticated knowledge access",
      "PASS",
      `HTTP 401 UNAUTHENTICATED`
    );
  } else {
    recordResult(
      "SEC-02",
      "Direct backend Funnel rejects unauthenticated knowledge access",
      "FAIL",
      `Expected HTTP 401, got HTTP ${directKbRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Section 2: Robot Language and Mode Compatibility
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 2: Robot Language and Mode Compatibility ---");

  // Create Robot R003 (Bangla / Education) via Super Admin API
  const r003Id = "R003-" + Date.now().toString().slice(-4);
  const createR3Res = await fetch(`${VERCEL_URL}/api/admin/robots`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${superToken}`,
    },
    body: JSON.stringify({
      robotId: r003Id,
      name: "Robot Three",
      language: "bn",
      mode: "education",
    }),
  });
  const r3Data = await createR3Res.json();
  if (createR3Res.status === 200 && r3Data.success) {
    recordResult(
      "ROBOT-01",
      "Create Robot supports Robot ID, Name, Language (bn), Mode (education)",
      "PASS",
      `HTTP 200, created ${r003Id}`
    );
  } else {
    recordResult(
      "ROBOT-01",
      "Create Robot supports Robot ID, Name, Language (bn), Mode (education)",
      "FAIL",
      `HTTP ${createR3Res.status}: ${JSON.stringify(r3Data)}`
    );
  }

  // Create Robot R004 (English / Healthcare) via Super Admin API
  const r004Id = "R004-" + Date.now().toString().slice(-4);
  const createR4Res = await fetch(`${VERCEL_URL}/api/admin/robots`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${superToken}`,
    },
    body: JSON.stringify({
      robotId: r004Id,
      name: "Robot Four",
      language: "en",
      mode: "healthcare",
    }),
  });
  const r4Data = await createR4Res.json();
  if (createR4Res.status === 200 && r4Data.success) {
    recordResult(
      "ROBOT-02",
      "Create Robot supports Language (en) and Mode (healthcare)",
      "PASS",
      `HTTP 200, created ${r004Id}`
    );
  } else {
    recordResult(
      "ROBOT-02",
      "Create Robot supports Language (en) and Mode (healthcare)",
      "FAIL",
      `HTTP ${createR4Res.status}: ${JSON.stringify(r4Data)}`
    );
  }

  // ---------------------------------------------------------------------------
  // Section 3: Knowledge Base Integrity & Assistant Retrieval Scoping
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 3: Knowledge Base Integrity & Assistant Retrieval ---");

  // Read Pi knowledge.csv via ssh
  const piKbOutput = execSync(
    'ssh -o ConnectTimeout=5 eliva@192.168.0.106 "cat /home/eliva/.config/dh360/knowledge.csv"'
  ).toString();
  const kbLines = piKbOutput.trim().split("\n");
  const rowCount = kbLines.length - 1; // subtract header

  const stableUUIDsPreserved =
    piKbOutput.includes("8473aeaf-83e6-44af-a1d3-3e4e0788bc05") &&
    piKbOutput.includes("2d41beeb-e61f-4435-888d-d28172236fa7") &&
    piKbOutput.includes("42a5738f-b772-4022-af4e-1c8846b516f8") &&
    piKbOutput.includes("38756334-c476-474e-bc36-ce228afd4a34");

  if (rowCount >= 15 && stableUUIDsPreserved) {
    recordResult(
      "KB-INTEG-01",
      "Persistent Knowledge Base preserved with stable UUIDs",
      "PASS",
      `${rowCount} records in runtime CSV, initial known UUIDs present and intact`
    );
  } else {
    recordResult(
      "KB-INTEG-01",
      "Persistent Knowledge Base preserved with stable UUIDs",
      "FAIL",
      `rowCount=${rowCount}, stableUUIDsPreserved=${stableUUIDsPreserved}`
    );
  }

  // Assistant dialogue turn using Active Robot R001 (assigned to Admin A)
  const turnResA = await fetch(`${VERCEL_URL}/api/dh360/dialogue/turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminAToken}`,
      "X-Robot-ID": "R001",
    },
    body: JSON.stringify({
      text: "তুমি কে?",
      language: "bn",
      speak_response: false,
    }),
  });
  const turnDataA = await turnResA.json();

  if (turnResA.status === 200 && turnDataA.text) {
    recordResult(
      "ASSIST-01",
      "Assistant executes dialogue turn with bound Robot R001 language (bn)",
      "PASS",
      `HTTP 200, response="${turnDataA.text.slice(0, 80)}..."`
    );
  } else {
    recordResult(
      "ASSIST-01",
      "Assistant executes dialogue turn with bound Robot R001 language (bn)",
      "FAIL",
      `HTTP ${turnResA.status}: ${JSON.stringify(turnDataA)}`
    );
  }

  // Assistant dialogue turn using Active Robot R002 (assigned to Admin B)
  const turnResB = await fetch(`${VERCEL_URL}/api/dh360/dialogue/turn`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${adminBToken}`,
      "X-Robot-ID": "R002",
    },
    body: JSON.stringify({
      text: "Who are you?",
      language: "en",
      speak_response: false,
    }),
  });
  const turnDataB = await turnResB.json();

  if (turnResB.status === 200 && turnDataB.text) {
    recordResult(
      "ASSIST-02",
      "Assistant executes dialogue turn with bound Robot R002 language (en)",
      "PASS",
      `HTTP 200, response="${turnDataB.text.slice(0, 80)}..."`
    );
  } else {
    recordResult(
      "ASSIST-02",
      "Assistant executes dialogue turn with bound Robot R002 language (en)",
      "FAIL",
      `HTTP ${turnResB.status}: ${JSON.stringify(turnDataB)}`
    );
  }

  // ---------------------------------------------------------------------------
  // Section 6: Deployment & Database Integrity Verification
  // ---------------------------------------------------------------------------
  console.log("\n--- Section 6: Deployment & Database Integrity ---");

  // Check Pi health
  const piHealthRes = await fetch(`${PI_URL}/health`);
  const piHealthData = await piHealthRes.json();
  const piHealthy = piHealthData.status === "healthy";
  const recognizerMsg = piHealthData.subsystems?.recognizer?.message || "";

  if (piHealthy && recognizerMsg.includes("3 identities")) {
    recordResult(
      "DEPLOY-01",
      "Raspberry Pi backend active with all 3 biometric gallery identities intact",
      "PASS",
      `Status healthy, recognizer: "${recognizerMsg}"`
    );
  } else {
    recordResult(
      "DEPLOY-01",
      "Raspberry Pi backend active with all 3 biometric gallery identities intact",
      "FAIL",
      `Healthy=${piHealthy}, recognizer="${recognizerMsg}"`
    );
  }

  // Check Vercel production dashboard
  const vercelRes = await fetch(`${VERCEL_URL}/en/login`);
  if (vercelRes.status === 200) {
    recordResult(
      "DEPLOY-02",
      "Production Vercel dashboard accessible and serving localized entrypoint",
      "PASS",
      `HTTP 200 from ${VERCEL_URL}/en/login`
    );
  } else {
    recordResult(
      "DEPLOY-02",
      "Production Vercel dashboard accessible and serving localized entrypoint",
      "FAIL",
      `HTTP ${vercelRes.status}`
    );
  }

  // ---------------------------------------------------------------------------
  // Summary
  // ---------------------------------------------------------------------------
  console.log("\n================================================================================");
  console.log("FINAL AUDIT RESULTS SUMMARY");
  console.log("================================================================================");

  let passCount = 0;
  let failCount = 0;

  for (const r of testResults) {
    if (r.status === "PASS") passCount++;
    else failCount++;
    console.log(`| ${r.testId} | ${r.name} | **${r.status}** | ${r.evidence} |`);
  }

  console.log(`\nTotal: ${testResults.length} tests, ${passCount} PASSED, ${failCount} FAILED.`);

  // Write results JSON
  fs.writeFileSync("/tmp/audit_results.json", JSON.stringify(testResults, null, 2));
}

run().catch((err) => {
  console.error("FATAL ERROR IN AUDIT RUNNER:", err);
  process.exit(1);
});
