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

## Models
Core ML models are **not in git**; CI fetches them with `ios/scripts/fetch_models.sh` into `Sources/CalorieLens/Resources/Models/` (cached by script hash), and Xcode compiles them into the app.
- **Depth:** Apple's `DepthAnythingV2SmallF16.mlpackage` (about 50 MB) from Hugging Face `apple/coreml-depth-anything-v2-small`, always fetched.
- **Detector:** run Actions > "Core ML export" > Run workflow. Leave `weights_url` empty for the stock COCO `yolo11n-seg.pt`, or paste a URL to your trained `best.pt`. The artifact **CalorieLensDetector** holds `CalorieLensDetector.mlpackage`, `model_spec.txt` (inputs/outputs, tells you whether masks are decoded), `labels.json` and `validation.txt`.
- To bundle the detector, zip the `.mlpackage`, host it (e.g. a GitHub Release asset), and set the repository variable `DETECTOR_URL` to that URL (Settings > Secrets and variables > Actions > Variables). Locally: `DETECTOR_URL=... bash ios/scripts/fetch_models.sh`.

## Scanning (v0.1, no trained detector yet)
1. Put the food on a table with some texture (wood grain, a placemat). A plain white or glossy surface gives ARKit
   too few feature points to find the plane.
2. Tap **Scan**, point at the table until the guidance changes, then slowly circle the food for about 5 s, staying
   about 30-50 cm away and looking down about 45 degrees.
3. Tap each label to name the dish. Pick a bowl container to re-measure the food inside the bowl.

Known limits of the no-detector mode (HeightSegment finds anything taller than 6 mm above the table):
- Everything on **one plate or in one bento box merges into one item**, and the plate or box itself adds volume.
  Until the detector is trained (ml/), scan foods in separate bowls or directly on a plate-free surface for
  measured volumes; otherwise use the container chips (default portions).
- A bowl measures as food + bowl material until you choose the bowl container, which re-measures it with the
  bowl-cavity model.
- Drinks in clear cups are never measured from depth (cup size x sugar level instead). Use **+ Drink**.

Each sweep is saved as a zip under Files > On My iPhone > Calorie Lens > sweeps and can be shared from the bar.
It holds depth maps, masks and poses only, no photos. Replay on the PC: `py tools/replay/run_replay.py --dump sweep.zip`.
Release-mode CI timing of the whole pipeline (518x392, 12 frames) is about 0.45 s on the runner, so about 1 s on a phone.
