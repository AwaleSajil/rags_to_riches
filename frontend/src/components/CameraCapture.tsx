import React, { useRef, useState } from "react";
import {
  StyleSheet,
  View,
  Modal,
  TouchableOpacity,
  Platform,
} from "react-native";
import { IconButton } from "react-native-paper";
import { CameraView, useCameraPermissions } from "expo-camera";
import { colors } from "../styles/theme";

interface CameraCaptureProps {
  visible: boolean;
  onClose: () => void;
  onCapture: (photo: { uri: string; name: string; type: string }) => void;
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

  const handleTakePicture = async () => {
    if (!cameraRef.current || isTaking) return;
    setIsTaking(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.8,
      });
      const timestamp = Date.now();
      onCapture({
        uri: photo.uri,
        name: `receipt_${timestamp}.jpg`,
        type: "image/jpeg",
      });
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
