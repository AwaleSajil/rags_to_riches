import { Image, Platform } from "react-native";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
import * as FileSystem from "expo-file-system/legacy";
import { createLogger } from "./logger";

const log = createLogger("CompressImage");

/**
 * The longest edge a photo is stored at.
 *
 * Chosen to match what the extractor can actually use rather than to hit a file
 * size. The vision call sends the photo to OpenAI without a `detail` parameter,
 * so it defaults to high detail — which rescales anything larger to fit inside
 * 2048x2048 before the model sees a single pixel. Uploading a 12MP camera frame
 * therefore spends bandwidth and a storage quota on detail that is discarded
 * server-side, every time.
 *
 * 2048 is deliberately conservative: it is the point where the model's own
 * resize stops removing anything, so extraction sees the same image it saw
 * before. Going lower would save more — the high-detail path scales the SHORT
 * edge to 768px after that — but the stored photo is also what the receipt
 * viewer zooms into to check a line the OCR got wrong, and that wants pixels.
 */
const MAX_EDGE = 2048;

/**
 * JPEG quality. Receipts are high-contrast text on white, which is the easiest
 * case for JPEG; artifacts at this level are invisible at the sizes above.
 */
const QUALITY = 0.7;

/**
 * Below this, the photo is uploaded exactly as it came off the camera roll.
 *
 * Re-encoding used to be unconditional, so an image that needed no shrinking
 * was still decoded and written back out as JPEG — a second lossy pass that
 * bought nothing. Generation loss lands hardest on small high-contrast glyphs,
 * which is precisely the part of a receipt the extractor has to read: the
 * total. A photo already under a megabyte is not what makes an upload slow.
 */
const MAX_BYTES = 1024 * 1024;

/**
 * Formats that survive a pass-through.
 *
 * Anything else is converted even when it is small, because the SERVER decides
 * from the file extension alone — `is_image` in file_service.py routes on
 * .png/.jpg/.jpeg, and money_rag.py picks the vision call's mime type the same
 * way. A HEIC straight from an iPhone camera roll matches neither: it would be
 * filed as a CSV and then handed to the model labelled image/png. Normalising
 * here was accidentally covering for that, and pass-through removes the cover.
 *
 * `image/jpg` is not a registered type, but Android content providers hand it
 * back often enough that spelling it out is cheaper than the alternative: a
 * small photo failing this test is re-encoded for nothing, which is the exact
 * silent quality loss this pass-through exists to stop.
 */
const PASS_THROUGH = /^image\/(jpe?g|png)$/;

export type UploadFile = { uri: string; name: string; type: string };

const jpegName = (name: string) =>
  /\.jpe?g$/i.test(name) ? name : `${name.replace(/\.[^.]+$/, "")}.jpg`;

const measure = (uri: string) =>
  new Promise<{ width: number; height: number }>((resolve, reject) =>
    Image.getSize(uri, (width, height) => resolve({ width, height }), reject)
  );

/**
 * Size on disk, or null when the platform will not say.
 *
 * Null means compress: an unknown size is treated as "possibly huge", which
 * keeps the old behaviour for anything this cannot measure rather than letting
 * a 12MP frame through on a guess.
 */
async function byteSize(uri: string): Promise<number | null> {
  try {
    if (Platform.OS === "web") {
      // getInfoAsync has no web implementation; a blob URL can be read directly.
      return (await (await fetch(uri)).blob()).size;
    }
    const info = await FileSystem.getInfoAsync(uri);
    return info.exists ? info.size ?? null : null;
  } catch (e) {
    log.warn("Could not measure file size", { uri, error: String(e) });
    return null;
  }
}

/**
 * Shrink a photo before it is uploaded. Non-images are returned untouched.
 *
 * Only when it is worth doing: a JPEG or PNG under MAX_BYTES is passed straight
 * through, untouched. Everything else is resized to MAX_EDGE if it is larger
 * than that, and written out as JPEG.
 *
 * Applied in the upload services rather than at each camera and picker, so a
 * new way to attach a photo cannot quietly bypass it.
 *
 * Never throws: a phone that cannot re-encode the image should still be able to
 * file its receipt, so any failure falls back to uploading the original.
 */
export async function compressImage(file: UploadFile): Promise<UploadFile> {
  if (!file.type?.startsWith("image/")) return file;

  // Measured for every image, not only the ones eligible to skip: the same
  // number is what makes the log below say what the re-encode actually saved,
  // which is the only way to tell whether QUALITY is set sensibly.
  const bytes = await byteSize(file.uri);

  // Small and already in a format the server understands: leave it completely
  // alone. A resize would be a no-op at this size anyway, so all a round trip
  // through the encoder can do is cost quality.
  if (PASS_THROUGH.test(file.type) && bytes !== null && bytes <= MAX_BYTES) {
    log.info("Photo left as-is", { name: file.name, bytes });
    return file;
  }

  try {
    // Measured up front rather than by rendering once to read the size: that
    // would decode the full frame twice for no reason.
    const source = await measure(file.uri);
    const context = ImageManipulator.manipulate(file.uri);

    const longest = Math.max(source.width, source.height);
    // A big file at modest dimensions — a PNG screenshot, a HEIC — still gets
    // re-encoded, because the saving there comes from the format, not the size.
    if (longest > MAX_EDGE) {
      const scale = MAX_EDGE / longest;
      context.resize({
        width: Math.round(source.width * scale),
        height: Math.round(source.height * scale),
      });
    }

    const rendered = await context.renderAsync();
    const result = await rendered.saveAsync({ compress: QUALITY, format: SaveFormat.JPEG });

    log.info("Photo compressed", {
      name: file.name,
      from: `${source.width}x${source.height}`,
      to: `${result.width}x${result.height}`,
      // Both sizes, so one real upload answers what QUALITY costs on an actual
      // receipt — it depends far more on sensor noise and what the photo was
      // taken against than on the quality number.
      bytes: `${bytes ?? "?"} → ${(await byteSize(result.uri)) ?? "?"}`,
      quality: QUALITY,
    });

    return { uri: result.uri, name: jpegName(file.name), type: "image/jpeg" };
  } catch (e) {
    log.warn("Could not compress, uploading original", { name: file.name, error: String(e) });
    return file;
  }
}
