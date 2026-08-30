import React, { useCallback, useEffect, useRef, useState } from "react";
import {
  StyleSheet,
  View,
  Modal,
  TouchableOpacity,
  Platform,
} from "react-native";
import { Button, IconButton, Text } from "react-native-paper";
import { CameraView, useCameraPermissions } from "expo-camera";
import { colors, spacing, typography } from "../styles/theme";
import {
  currentPlaceName,
  hasPlacePermission,
  requestPlacePermission,
} from "../lib/placeName";

interface CameraCaptureProps {
  visible: boolean;
  onClose: () => void;
  /** `place` is a resolved name like "Main St, Norwalk" — never coordinates,
   *  and null whenever location is unavailable or declined. */
  onCapture: (
    photo: { uri: string; name: string; type: string },
    place: string | null
  ) => void;
}

export function CameraCapture({
  visible,
  onClose,
  onCapture,
}: CameraCaptureProps) {
  const cameraRef = useRef<any>(null);
  const [facing, setFacing] = useState<"back" | "front">("back");
  const [flash, setFlash] = useState<"off" | "on">("off");
  const [isTaking, setIsTaking] = useState(false);
  // Resolved while the viewfinder is open rather than on the shutter: a fix
  // indoors takes a few seconds and nobody should wait for it holding up a
  // phone at a shelf.
  const [place, setPlace] = useState<string | null>(null);
  const [locationAllowed, setLocationAllowed] = useState<boolean | null>(null);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    (async () => {
      const allowed = await hasPlacePermission();
      if (cancelled) return;
      setLocationAllowed(allowed);
      if (!allowed) return;
      const resolved = await currentPlaceName();
      if (!cancelled) setPlace(resolved);
    })();
    return () => {
      cancelled = true;
    };
  }, [visible]);

  const enableLocation = useCallback(async () => {
    const allowed = await requestPlacePermission();
    setLocationAllowed(allowed);
    if (allowed) setPlace(await currentPlaceName());
  }, []);

  const handleTakePicture = async () => {
    if (!cameraRef.current || isTaking) return;
    setIsTaking(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        // Captured lossless-ish, not at 0.8. compressImage re-encodes this
        // frame before it is uploaded — a 12MP capture is always over
        // MAX_BYTES — so encoding once here and again there threw away detail
        // twice to reach the same stored file. What survives is one encode.
        quality: 1,
      });
      const timestamp = Date.now();
      onCapture(
        {
          uri: photo.uri,
          name: `receipt_${timestamp}.jpg`,
          type: "image/jpeg",
        },
        place
      );
      onClose();
    } catch (e) {
      console.error("Failed to take picture:", e);
    } finally {
      setIsTaking(false);
    }
  };

  const toggleFacing = () => {
    setFacing((prev) => (prev === "back" ? "front" : "back"));
  };

  const toggleFlash = () => {
    setFlash((prev) => (prev === "off" ? "on" : "off"));
  };

  if (Platform.OS === "web") {
    return null;
  }

  return (
    <Modal
      visible={visible}
      animationType="slide"
      presentationStyle="fullScreen"
      onRequestClose={onClose}
    >
      <View style={styles.container}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          flash={flash}
        />

        {/* Overlay controls on top of camera */}
        <View style={styles.overlay}>
          {/* Top controls */}
          <View style={styles.topBar}>
            <IconButton
              icon="close"
              iconColor="#fff"
              size={28}
              onPress={onClose}
              style={styles.topButton}
            />
            <IconButton
              icon={flash === "off" ? "flash-off" : "flash"}
              iconColor="#fff"
              size={28}
              onPress={toggleFlash}
              style={styles.topButton}
            />
          </View>

          {/* Spacer between top and bottom controls */}
          <View style={styles.spacer} />

          {/* Why location matters, said before it is asked for rather than in a
              bare system prompt: the same item costs different amounts at
              different shops, so a price with no place attached is much weaker
              evidence later. Declining is a normal answer. */}
          {locationAllowed === false && (
            <View style={styles.locationPrompt}>
              <Text style={styles.locationTitle}>Record the shop?</Text>
              <Text style={styles.locationBody}>
                Prices depend on where you are — the same item costs different amounts at
                different shops and in different cities. Adding the place keeps your price
                notes worth comparing later.
              </Text>
              <Text style={styles.locationBody}>
                Your location becomes a place name on this phone. Only the name is saved,
                never your coordinates.
              </Text>
              <View style={styles.locationActions}>
                <Button mode="contained" compact onPress={enableLocation}>
                  Add the shop
                </Button>
                <Button
                  mode="text"
                  compact
                  textColor="#fff"
                  onPress={() => setLocationAllowed(null)}
                >
                  Not now
                </Button>
              </View>
            </View>
          )}
          {place && (
            <View style={styles.placeChip}>
              <Text style={styles.placeText} numberOfLines={1}>
                📍 {place}
              </Text>
            </View>
          )}

          {/* Bottom controls */}
          <View style={styles.bottomBar}>
            <View style={styles.bottomSpacer} />

            <TouchableOpacity
              style={[
                styles.captureButton,
                isTaking && styles.captureButtonDisabled,
              ]}
              onPress={handleTakePicture}
              disabled={isTaking}
              activeOpacity={0.7}
            >
              <View style={styles.captureInner} />
            </TouchableOpacity>

            <View style={styles.bottomSpacer}>
              <IconButton
                icon="camera-flip"
                iconColor="#fff"
                size={30}
                onPress={toggleFacing}
                style={styles.flipButton}
              />
            </View>
          </View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  locationPrompt: {
    backgroundColor: "rgba(0,0,0,0.78)",
    marginHorizontal: spacing.md,
    marginBottom: spacing.sm,
    padding: spacing.md,
    borderRadius: 12,
    gap: spacing.xs,
  },
  locationTitle: { ...typography.body2, color: "#fff", fontWeight: "700" },
  locationBody: { ...typography.caption, color: "rgba(255,255,255,0.85)" },
  locationActions: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.xs,
    marginTop: spacing.xs,
  },
  placeChip: {
    alignSelf: "center",
    backgroundColor: "rgba(0,0,0,0.6)",
    paddingHorizontal: spacing.sm,
    paddingVertical: 4,
    borderRadius: 999,
    marginBottom: spacing.sm,
    maxWidth: "80%",
  },
  placeText: { ...typography.caption, color: "#fff" },
  container: {
    flex: 1,
    backgroundColor: "#000",
  },
  camera: {
    ...StyleSheet.absoluteFillObject,
  },
  overlay: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 1,
  },
  topBar: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingTop: 50,
    paddingHorizontal: 8,
  },
  topButton: {
    backgroundColor: "rgba(0,0,0,0.4)",
    borderRadius: 20,
  },
  spacer: {
    flex: 1,
  },
  bottomBar: {
    flexDirection: "row",
    justifyContent: "center",
    alignItems: "center",
    paddingBottom: 40,
    paddingHorizontal: 20,
  },
  bottomSpacer: {
    flex: 1,
    alignItems: "center",
  },
  captureButton: {
    width: 76,
    height: 76,
    borderRadius: 38,
    backgroundColor: "rgba(255,255,255,0.3)",
    justifyContent: "center",
    alignItems: "center",
    borderWidth: 3,
    borderColor: "#fff",
  },
  captureButtonDisabled: {
    opacity: 0.5,
  },
  captureInner: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: "#fff",
  },
  flipButton: {
    backgroundColor: "rgba(0,0,0,0.4)",
    borderRadius: 20,
  },
});
