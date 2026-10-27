import { initializeApp, getApps, getApp } from "firebase/app";
import { getAuth, connectAuthEmulator } from "firebase/auth";
import { getFirestore, connectFirestoreEmulator } from "firebase/firestore";
import { getFunctions, connectFunctionsEmulator } from "firebase/functions";

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

// Validate presence of critical client environment variables without exposing secret values
if (typeof window !== "undefined") {
  const missingKeys: string[] = [];
  if (!firebaseConfig.apiKey) missingKeys.push("NEXT_PUBLIC_FIREBASE_API_KEY");
  if (!firebaseConfig.projectId) missingKeys.push("NEXT_PUBLIC_FIREBASE_PROJECT_ID");
  if (!firebaseConfig.authDomain) missingKeys.push("NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN");
  if (missingKeys.length > 0) {
    console.warn(
      `[Firebase] Missing required client configuration variables: ${missingKeys.join(", ")}. Please verify your environment setup.`
    );
  }
}

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

export const auth = getAuth(app);
export const db = getFirestore(app);
// Cloud Functions asia-south1 অঞ্চলে ডিপ্লয় হয়
export const functions = getFunctions(app, "asia-south1");

const useEmulator = process.env.NEXT_PUBLIC_USE_EMULATOR === "true";

// একবারই সংযোগ হবে, hot reload-এ বারবার নয়
const globalForEmu = globalThis as unknown as { __emuConnected?: boolean };

if (
  useEmulator &&
  typeof window !== "undefined" &&
  !globalForEmu.__emuConnected
) {
  connectAuthEmulator(auth, "http://127.0.0.1:9099", { disableWarnings: true });
  connectFirestoreEmulator(db, "127.0.0.1", 8080);
  connectFunctionsEmulator(functions, "127.0.0.1", 5001);
  globalForEmu.__emuConnected = true;
}
