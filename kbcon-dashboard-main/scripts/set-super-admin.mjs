// ব্যবহার: node scripts/set-super-admin.mjs <ইমেইল>
// এটি শুধু Emulator-এর জন্য। আসল Firebase-এ চালানো হবে না।

import { createRequire } from "node:module";

const require = createRequire(
  new URL("../functions/package.json", import.meta.url),
);
const admin = require("firebase-admin");

const email = process.argv[2];

if (!email) {
  console.error(
    "ইমেইল দিন। উদাহরণ: node scripts/set-super-admin.mjs admin@test.com",
  );
  process.exit(1);
}

// Emulator ছাড়া চালালে থামিয়ে দেয়, যাতে ভুল করে আসল প্রজেক্টে না লাগে
if (!process.env.FIREBASE_AUTH_EMULATOR_HOST) {
  console.error(
    "FIREBASE_AUTH_EMULATOR_HOST সেট করা নেই। এই স্ক্রিপ্ট এখন শুধু Emulator-এর জন্য।",
  );
  process.exit(1);
}

admin.initializeApp({ projectId: "kbcon-dashboard" });

async function main() {
  const user = await admin.auth().getUserByEmail(email);

  await admin.auth().setCustomUserClaims(user.uid, { role: "superAdmin" });

  await admin
    .firestore()
    .collection("users")
    .doc(user.uid)
    .set(
      {
        name: user.displayName || "",
        email: user.email,
        role: "superAdmin",
        active: true,
        createdAt: new Date().toISOString(),
        createdBy: "bootstrap-script",
      },
      { merge: true },
    );

  console.log(`সফল: ${email} (${user.uid}) এখন superAdmin।`);
}

main().catch((err) => {
  console.error("ব্যর্থ:", err.message);
  process.exit(1);
});
