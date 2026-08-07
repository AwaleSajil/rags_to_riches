import * as Location from "expo-location";
import { Platform } from "react-native";
import { createLogger } from "./logger";

const log = createLogger("PlaceName");

/**
 * Where a photo was taken, resolved to a NAME on this device.
 *
 * Prices are local. The same jug of milk is a different price at a different
 * shop, and very different in a different city — a gallon bought in Huntsville
 * is not the going rate in Norwalk. A shelf tag carries no address, so unless
 * the phone says where it was, an observation has nothing to anchor it but a
 * merchant name the user may have several branches of.
 *
 * The coordinates never leave the device. The fix is reverse-geocoded here and
 * only the resulting label — "Main St, Norwalk" — is sent, so the server and the
 * database hold a place, not a position. That is a deliberate line: knowing
 * which shop a price came from is the point; knowing where somebody is is not.
 *
 * Every step is allowed to fail. A refused permission, a denied fix, an indoor
 * signal that never resolves — all of them return null and everything
 * downstream works without it.
 */
export async function currentPlaceName(): Promise<string | null> {
  // No usable equivalent in a browser without a second permission prompt for
  // little gain; the shop case is a phone case.
  if (Platform.OS === "web") return null;

  try {
    const { status } = await Location.getForegroundPermissionsAsync();
    if (status !== Location.PermissionStatus.GRANTED) {
      log.debug("Location permission not granted — capturing without a place");
      return null;
    }

    // Balanced, not Highest: naming a shop needs tens of metres, not one, and
    // the high-accuracy fix takes far longer indoors — where every shop is.
    const fix = await Location.getCurrentPositionAsync({
      accuracy: Location.Accuracy.Balanced,
    });

    const [place] = await Location.reverseGeocodeAsync({
      latitude: fix.coords.latitude,
      longitude: fix.coords.longitude,
    });
    if (!place) return null;

    // Street and city are enough to tell two branches apart while stopping well
    // short of a doorstep. House number is deliberately omitted.
    const label = [place.name && place.name !== place.street ? null : place.street, place.city]
      .filter(Boolean)
      .join(", ");
    const resolved = label || place.city || place.region || null;
    log.info("Resolved place name", { resolved });
    return resolved;
  } catch (e) {
    log.warn("Could not resolve a place name", e);
    return null;
  }
}

/**
 * Ask for location, explaining why first.
 *
 * Returns whether it is now granted. Declining is a normal answer: it costs the
 * shop name on a price, and nothing else.
 */
export async function requestPlacePermission(): Promise<boolean> {
  if (Platform.OS === "web") return false;
  try {
    const { status } = await Location.requestForegroundPermissionsAsync();
    return status === Location.PermissionStatus.GRANTED;
  } catch (e) {
    log.warn("Location permission request failed", e);
    return false;
  }
}

export async function hasPlacePermission(): Promise<boolean> {
  if (Platform.OS === "web") return false;
  try {
    const { status } = await Location.getForegroundPermissionsAsync();
    return status === Location.PermissionStatus.GRANTED;
  } catch {
    return false;
  }
}
