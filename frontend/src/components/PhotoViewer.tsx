/**
 * Looking at a photo — the same way, wherever you tapped it.
 *
 * There were two of these. Tapping a receipt in Files opened a sheet with the
 * filename, a rotate control, a download and previous/next; tapping the same
 * receipt in a chat reply opened a bare black screen with an X. Same picture,
 * two answers to "what can I do with this", and only one of them was useful.
 *
 * This owns what is specific to a photo — the rotate control and the zoomable
 * image — and gets the sheet, header and paging from PreviewSheet, which the
 * CSV preview shares. What differs between callers is data, not layout: a Files
 * preview knows a filename and an upload date, a chat attachment knows neither,
 * and either may or may not be downloadable.
 *
 * Deliberately NOT here: signing URLs, listing siblings, deciding what a file
 * is. Those belong to whoever owns the data.
 */

import React from "react";
import { StyleSheet, View } from "react-native";
import { ActivityIndicator, IconButton, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import { colors, spacing, typography } from "../styles/theme";
import type { SourceVisual } from "../lib/sourceVisual";
import { PreviewSheet } from "./PreviewSheet";
import { ZoomableImage } from "./ZoomableImage";

export interface PhotoViewerProps {
  /**
   * Open or shut. Separate from `uri` because the sheet must stay up while the
   * next photo's URL is being signed — tying the two together would close and
   * reopen it on every page, which is the animation this was fixing.
   */
  visible: boolean;
  /** The image to show, once there is one. */
  uri: string | null;
  onClose: () => void;
  title: string;
  subtitle?: string;
  icon?: SourceVisual;
  /**
   * Save the original. Omitted when there is nothing to save — a photo just
   * taken on this device is already in the camera roll, and offering to
   * download it is a button that cannot mean anything.
   */
  onDownload?: () => void;
  downloading?: boolean;
  notice?: string | null;
  onPrevious?: () => void;
  onNext?: () => void;
  position?: { index: number; total: number };
  /** Actions below the image, e.g. "Review / edit extracted data". */
  footer?: React.ReactNode;
  /**
   * Shown in place of the image while its URL is being signed, and if that
   * fails.
   *
   * Handled HERE rather than by the caller swapping in a different component,
   * because swapping means a different Modal at the same position: React
   * unmounts one and mounts the other, and both play their slide animation. On
   * every tap of Next the sheet dropped off the bottom of the screen and came
   * back. The states live inside one Modal so paging just changes the picture.
   */
  loading?: boolean;
  error?: string | null;
  /**
   * Where this photo starts, in quarter turns clockwise. Remembered per file
   * by the caller; 0 for anything with nowhere to remember it.
   */
  initialRotation?: number;
  /**
   * Called when the user settles on a new angle — debounced, so spinning all
   * the way round is one call rather than four.
   */
  onRotationChange?: (degrees: number) => void;
  /**
   * Rendered inside the sheet, alongside the image. For overlays that must sit
   * above it — a confirmation dialog, say, which has to be inside this Modal to
   * appear over it at all.
   */
  children?: React.ReactNode;
}

export function PhotoViewer({
  visible,
  uri,
  onClose,
  title,
  subtitle,
  icon,
  onDownload,
  downloading = false,
  notice,
  onPrevious,
  onNext,
  position,
  footer,
  loading = false,
  error,
  initialRotation = 0,
  onRotationChange,
  children,
}: PhotoViewerProps) {
  // Quarter turns. The stored image is never rewritten — the angle is what is
  // remembered, so the original bytes stay exactly as they were photographed.
  const [rotation, setRotation] = React.useState(initialRotation);

  // Each photo opens at ITS OWN saved angle. Carrying the last one's turn
  // across would rotate a picture that was already fine.
  React.useEffect(() => {
    setRotation(initialRotation);
  }, [uri, initialRotation]);

  // Saved a beat after the last tap, so turning 90° four times back to upright
  // writes once — and writes 0, not a pointless 360.
  const settled = React.useRef(initialRotation);
  React.useEffect(() => {
    if (!onRotationChange || rotation === settled.current) return;
    const timer = setTimeout(() => {
      settled.current = rotation;
      onRotationChange(rotation);
    }, 600);
    return () => clearTimeout(timer);
  }, [rotation, onRotationChange]);

  // A different photo resets the baseline, or the next one's first turn would
  // be compared against the previous photo's angle.
  React.useEffect(() => {
    settled.current = initialRotation;
  }, [uri, initialRotation]);

  return (
    <PreviewSheet
      visible={visible}
      onClose={onClose}
      title={title}
      subtitle={subtitle}
      icon={icon}
      onPrevious={onPrevious}
      onNext={onNext}
      position={position}
      notice={notice}
      footer={footer}
      actions={
        <>
          {uri ? (
            <IconButton
              icon="rotate-right"
              size={22}
              iconColor={colors.textSecondary}
              onPress={() => setRotation((current) => (current + 90) % 360)}
              accessibilityLabel="Rotate the photo 90 degrees"
            />
          ) : null}
          {onDownload && uri ? (
            <IconButton
              icon="tray-arrow-down"
              size={22}
              iconColor={colors.textSecondary}
              onPress={onDownload}
              disabled={downloading}
              accessibilityLabel="Download the original photo"
            />
          ) : null}
        </>
      }
    >
      {loading ? (
        <View style={styles.state}>
          <ActivityIndicator color={colors.primary} />
          <Text style={styles.stateText}>Loading photo…</Text>
        </View>
      ) : error ? (
        <View style={styles.state}>
          <MaterialCommunityIcons
            name="image-off-outline"
            size={40}
            color={colors.textTertiary}
          />
          <Text style={styles.errorText}>{error}</Text>
        </View>
      ) : uri ? (
        /* Fills whatever the sheet has left rather than a fixed fraction of the
           screen — a receipt is tall and detailed, and every pixel of it is the
           reason someone opened this.
           Keyed on the uri so paging to the next photo starts unzoomed rather
           than inheriting the last one's pan and scale. */
        <ZoomableImage key={uri} uri={uri} rotation={rotation} style={styles.image} />
      ) : null}
      {children}
    </PreviewSheet>
  );
}

const styles = StyleSheet.create({
  image: {
    flex: 1,
    width: "100%",
  },
  state: {
    alignItems: "center",
    justifyContent: "center",
    gap: spacing.sm,
    padding: spacing.xl,
  },
  stateText: {
    ...typography.caption,
    color: colors.textTertiary,
  },
  errorText: {
    ...typography.body2,
    color: colors.textSecondary,
    textAlign: "center",
  },
});
