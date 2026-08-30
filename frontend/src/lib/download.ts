/**
 * Getting a file out of the app and onto the device.
 *
 * Both photo entry points need this — a receipt in the Files tab and the same
 * receipt attached to a chat reply — so it lives here rather than in whichever
 * component happened to need it first.
 */

import { Linking, Platform } from "react-native";

/**
 * Hand a URL to the platform so it lands in the user's downloads.
 *
 * Native has no share sheet available (expo-sharing is not a dependency), so
 * the OS browser does the saving — which puts the file in the device's normal
 * Downloads folder, where someone looking for it will go.
 */
export async function saveToDevice(url: string, filename: string): Promise<void> {
  if (Platform.OS === "web") {
    // An anchor rather than window.open: a popup blocker will silently eat the
    // second, and Content-Disposition means no tab is needed anyway.
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.rel = "noopener";
    document.body.appendChild(link);
    link.click();
    link.remove();
    return;
  }
  await Linking.openURL(url);
}

/**
 * Ask storage to send a file as an attachment rather than render it.
 *
 * `download` is a plain query parameter the storage server reads; it is not
 * part of what the URL was signed over, so it can be appended to a link that
 * has already been signed. That matters for chat attachments, where the signed
 * URL arrives from the agent and cannot be re-minted here.
 */
export function asAttachment(url: string, filename: string): string {
  const separator = url.includes("?") ? "&" : "?";
  return `${url}${separator}download=${encodeURIComponent(filename)}`;
}

/**
 * A displayable name for a photo known only by its URL.
 *
 * Chat attachments carry no metadata — just a signed link — so the last path
 * segment is the only name available. Query string and percent-encoding are
 * stripped so it reads as the filename it is rather than as a URL fragment.
 */
export function photoFilename(url: string, fallback = "Photo"): string {
  try {
    const path = url.split("?")[0];
    const last = path.substring(path.lastIndexOf("/") + 1);
    const name = decodeURIComponent(last).trim();
    return name || fallback;
  } catch {
    // A malformed percent-escape is not worth failing a title over.
    return fallback;
  }
}
