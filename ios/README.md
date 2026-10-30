# Calorie Lens iOS: install loop (Windows)

CI builds an **unsigned** `.ipa`; you sign and install it with Sideloadly.

## One-time setup
1. Install **iTunes** and **iCloud** from apple.com (the direct installers, NOT the Microsoft Store versions; Sideloadly cannot use those).
2. Install **Sideloadly** (sideloadly.io).
3. Use a free Apple ID. A secondary Apple ID just for sideloading is recommended.

## Every build
1. Push a change under `ios/` (or run the workflow manually: Actions > "iOS build" > Run workflow).
2. Open the finished run, download the artifact **CalorieLens-ipa**, and unzip it to get `CalorieLens.ipa`.
3. Connect the iPhone by USB and tap Trust. Open Sideloadly, drag in the `.ipa`, enter the Apple ID, click Start.
4. First install only: on the iPhone go to Settings > Privacy & Security > **Developer Mode**. The toggle appears only after the first install attempt. Turn it on and restart. Then go to Settings > General > VPN & Device Management > your Apple ID > **Trust**.
5. Launch Calorie Lens and allow camera access.

## Expiry
Free-account apps expire after **7 days**. Re-run Sideloadly with the same IPA, or enable its auto-refresh (keep Sideloadly running, with the phone on the same Wi-Fi or USB). Max 3 sideloaded apps.

## Logs and crashes
- In app: tap **Logs** (top right), then the share button to export `logs.txt`.
- Files live in the app's Documents folder (file sharing is enabled, so they appear in the Files app under "On My iPhone > Calorie Lens"): `logs.txt` and `last_crash.txt`.
- If the app crashed, `last_crash.txt` is read on the next launch, shown in an alert, appended to the log and deleted.
- dSYMs are uploaded as the **CalorieLens-dSYM** artifact.
