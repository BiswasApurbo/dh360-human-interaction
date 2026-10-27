import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

console.log("Running People Feature Verification Suite...");

// 1. Verify i18n messages
const enMessages = JSON.parse(
  fs.readFileSync(path.join(ROOT, "messages/en.json"), "utf8")
);
const bnMessages = JSON.parse(
  fs.readFileSync(path.join(ROOT, "messages/bn.json"), "utf8")
);

// Check Nav item
assert.equal(enMessages.Nav.people, "People", "en.json missing Nav.people");
assert.equal(bnMessages.Nav.people, "ব্যক্তিবর্গ", "bn.json missing Nav.people");

// Check People block
assert.ok(enMessages.People, "en.json missing People block");
assert.ok(bnMessages.People, "bn.json missing People block");

const requiredKeys = [
  "title",
  "subtitle",
  "registerPerson",
  "searchPlaceholder",
  "noPeople",
  "name",
  "preferredName",
  "role",
  "department",
  "officeRoom",
  "workingHours",
  "profileDescription",
  "consentStatus",
  "status",
  "active",
  "suspended",
  "granted",
  "revoked",
  "pending",
  "actions",
  "viewProfile",
  "resetBiometrics",
  "delete",
  "qaPairs",
  "question",
  "answer",
  "addQa",
  "cancel",
  "save",
  "deleteConfirmTitle",
  "resetConfirmTitle",
  "enrollBiometrics",
  "enrollmentTitle",
  "testAuthorization",
  "authTitle",
];

for (const key of requiredKeys) {
  assert.ok(
    enMessages.People[key],
    `en.json People block missing required key: ${key}`
  );
  assert.ok(
    bnMessages.People[key],
    `bn.json People block missing required key: ${key}`
  );
}
console.log("✓ 1. i18n Localization keys verified in both English and Bengali");

// 2. Verify files exist
const expectedFiles = [
  "src/app/[locale]/(dashboard)/people/page.tsx",
  "src/features/people/types.ts",
  "src/features/people/use-people.ts",
  "src/features/people/people-table.tsx",
  "src/features/people/create-person-dialog.tsx",
  "src/features/people/delete-person-dialog.tsx",
  "src/features/people/reset-biometrics-dialog.tsx",
  "src/features/people/person-detail-sheet.tsx",
  "src/features/people/enrollment-dialog.tsx",
  "src/features/people/authorization-dialog.tsx",
  "src/features/people/people-page-content.tsx",
];

for (const f of expectedFiles) {
  assert.ok(
    fs.existsSync(path.join(ROOT, f)),
    `Expected component file missing: ${f}`
  );
}
console.log("✓ 2. All 9 People feature modules and pages exist");

// 3. Verify AppSidebar includes people navigation
const sidebarCode = fs.readFileSync(
  path.join(ROOT, "src/components/layout/app-sidebar.tsx"),
  "utf8"
);
assert.ok(
  sidebarCode.includes('href: "/people"'),
  "app-sidebar.tsx does not link to /people"
);
console.log("✓ 3. AppSidebar successfully integrates /people route");

console.log("========================================");
console.log("People Feature Suite: All checks passed!");
console.log("========================================");
