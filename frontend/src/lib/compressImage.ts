import { Image } from "react-native";
import { ImageManipulator, SaveFormat } from "expo-image-manipulator";
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

export type UploadFile = { uri: string; name: string; type: string };

const jpegName = (name: string) =>
  /\.jpe?g$/i.test(name) ? name : `${name.replace(/\.[^.]+$/, "")}.jpg`;

const measure = (uri: string) =>
  new Promise<{ width: number; height: number }>((resolve, reject) =>
    Image.getSize(uri, (width, height) => resolve({ width, height }), reject)
  );

/**
 * Shrink a photo before it is uploaded. Non-images are returned untouched.
 *
 * Applied in the upload services rather than at each camera and picker, so a
 * new way to attach a photo cannot quietly bypass it.
 *
 * Never throws: a phone that cannot re-encode the image should still be able to
 * file its receipt, so any failure falls back to uploading the original.
 */
export async function compressImage(file: UploadFile): Promise<UploadFile> {
  if (!file.type?.startsWith("image/")) return file;

  try {
    // Measured up front rather than by rendering once to read the size: that
    // would decode the full frame twice for no reason.
    const source = await measure(file.uri);
    const context = ImageManipulator.manipulate(file.uri);

    const longest = Math.max(source.width, source.height);
    // Still re-encoded when it is already small enough: a PNG screenshot of a
    // receipt can be several megabytes at modest dimensions, and the saving
    // there comes from the format, not the size.
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
    });

    return { uri: result.uri, name: jpegName(file.name), type: "image/jpeg" };
  } catch (e) {
    log.warn("Could not compress, uploading original", { name: file.name, error: String(e) });
    return file;
  }
}
