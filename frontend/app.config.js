// Expo config as JS rather than app.json, so the cleartext-HTTP escape hatches
// can be switched off for real builds.
//
// Development talks to the backend over plain HTTP — the Android emulator uses
// http://10.0.2.2:8000 and iOS uses http://<LAN-IP>:8000 (see src/lib/apiUrl.ts).
// Both platforms block that by default, so dev builds need the exceptions below.
// Shipping them, though, means an app carrying bank transactions and Supabase
// JWTs will happily talk plaintext to anything.
//
// So: SECURE BY DEFAULT, opt in for development. The npm dev scripts set
// EXPO_ALLOW_CLEARTEXT=1; `eas build` and `expo export` do not go through those
// scripts, so a release build gets HTTPS-only whether or not anyone remembers.
const allowCleartext = process.env.EXPO_ALLOW_CLEARTEXT === "1";

const locationPermission =
  "R2R records which shop a price was seen in, because the same item costs " +
  "different amounts at different shops and in different cities. Your location " +
  "is turned into a place name on this device and only that name is stored — " +
  "never your coordinates.";

module.exports = {
  expo: {
    name: "R2R",
    slug: "r2r",
    version: "1.0.0",
    orientation: "portrait",
    icon: "./assets/icon.png",
    userInterfaceStyle: "light",
    newArchEnabled: true,
    scheme: "r2r",
    splash: {
      image: "./assets/splash-icon.png",
      resizeMode: "contain",
      backgroundColor: "#6366f1",
    },
    ios: {
      supportsTablet: true,
      bundleIdentifier: "com.r2r.app",
      infoPlist: {
        // Declares that this app uses no non-exempt encryption. The app itself
        // only speaks HTTPS; the Fernet encryption protecting stored API keys
        // runs on the SERVER and is not in this binary. Without this key App
        // Store Connect asks the export-compliance question on every single
        // upload, which is the kind of friction that eventually gets answered
        // carelessly. Confirm it against your own build before submitting —
        // this is a legal declaration, not a config flag.
        ITSAppUsesNonExemptEncryption: false,
        // No ATS exception in a release build — that is the secure default, and
        // an absent key is stricter than any value we could write here.
        ...(allowCleartext ? { NSAppTransportSecurity: { NSAllowsArbitraryLoads: true } } : {}),
      },
    },
    android: {
      adaptiveIcon: {
        foregroundImage: "./assets/adaptive-icon.png",
        backgroundColor: "#6366f1",
      },
      softwareKeyboardLayoutMode: "pan",
      package: "com.r2r.app",
      // Explicitly false rather than omitted: Android's default flipped to false
      // in API 28, but being explicit means a manifest merge from some library
      // cannot quietly turn it back on.
      usesCleartextTraffic: allowCleartext,
    },
    web: {
      favicon: "./assets/favicon.png",
      bundler: "metro",
    },
    plugins: [
      "expo-router",
      [
        "expo-camera",
        {
          cameraPermission:
            "Allow R2R to access your camera to capture receipts and documents.",
        },
      ],
      [
        "expo-location",
        {
          locationAlwaysAndWhenInUsePermission: locationPermission,
          locationWhenInUsePermission: locationPermission,
          isIosBackgroundLocationEnabled: false,
          isAndroidBackgroundLocationEnabled: false,
        },
      ],
    ],
  },
};
