/**
 * The bottom sheet everything is previewed in.
 *
 * Only the chrome: the sheet, the header (icon, title, subtitle, actions,
 * close), the floating paging arrows, a notice line and a footer slot. What
 * goes inside is the caller's — a zoomable photo, a CSV table — because that is
 * the only part that genuinely differs between them.
 *
 * Split out from PhotoViewer once the CSV preview needed the same header: two
 * copies of a header is exactly how the chat viewer and the file viewer drifted
 * into looking like unrelated features in the first place.
 */

import React from "react";
import { Modal, StyleSheet, View } from "react-native";
import { IconButton, Portal, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";

import { colors, spacing, typography } from "../styles/theme";
import type { SourceVisual } from "../lib/sourceVisual";

export interface PreviewSheetProps {
  visible: boolean;
  onClose: () => void;
  title: string;
  subtitle?: string;
  /** Tinted circle beside the title. Omit when the kind is unknown. */
  icon?: SourceVisual;
  /** Icon buttons between the subtitle and the close button. */
  actions?: React.ReactNode;
  /** Paging. `position` is what turns the arrows on. */
  onPrevious?: () => void;
  onNext?: () => void;
  position?: { index: number; total: number };
  /** One line under the content: the result of an action, or a warning. */
  notice?: string | null;
  /** Buttons below the content. */
  footer?: React.ReactNode;
  children: React.ReactNode;
}

export function PreviewSheet({
  visible,
  onClose,
  title,
  subtitle,
  icon,
  actions,
  onPrevious,
  onNext,
  position,
  notice,
  footer,
  children,
}: PreviewSheetProps) {
  const canPage = Boolean(position && position.total > 1);
  // The counter rides in the header rather than in a strip of its own — the
  // paging controls now float over the image, so there is nowhere else for it
  // and no reason to spend a row of height on it.
  const headerSubtitle = [
    subtitle,
    canPage ? `${position!.index + 1} of ${position!.total}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      {/* Its own Portal.Host, so a Dialog rendered by anything inside this
          sheet lands INSIDE this Modal. Paper's Portal otherwise mounts at the
          app root, which is behind a native Modal — a delete confirmation put
          there is invisible under the very sheet that raised it. */}
      <Portal.Host>
          <View style={styles.backdrop}>
            <View style={styles.sheet}>
            {/* Says "this can be dismissed" before anyone hunts for the X. */}
            <View style={styles.grabber} />

            <View style={styles.headerRow}>
              {icon ? (
                <View style={[styles.headerIcon, { backgroundColor: icon.background }]}>
                  <MaterialCommunityIcons name={icon.icon} size={20} color={icon.color} />
                </View>
              ) : null}
              <View style={styles.headerText}>
                <Text style={styles.title} numberOfLines={1}>
                  {title}
                </Text>
                {headerSubtitle ? (
                  <Text style={styles.subtitle} numberOfLines={1}>
                    {headerSubtitle}
                  </Text>
                ) : null}
              </View>
              {actions}
              <IconButton
                icon="close"
                size={22}
                iconColor={colors.textSecondary}
                onPress={onClose}
                accessibilityLabel="Close preview"
              />
            </View>

            {/* The content takes every pixel the header and footer do not. */}
            <View style={styles.content}>
              {children}

              {/* Paging floats over the content, centred on its edges, so it
                  costs no height. Each side is rendered only when there is
                  somewhere to go — a permanently greyed-out arrow sitting on top
                  of a photo is worse than no arrow. */}
              {canPage && onPrevious ? (
                <IconButton
                  icon="chevron-left"
                  size={28}
                  iconColor="#ffffff"
                  containerColor={PAGE_BUTTON_TINT}
                  onPress={onPrevious}
                  style={[styles.pageButton, styles.pageLeft]}
                  accessibilityLabel="Previous"
                />
              ) : null}
              {canPage && onNext ? (
                <IconButton
                  icon="chevron-right"
                  size={28}
                  iconColor="#ffffff"
                  containerColor={PAGE_BUTTON_TINT}
                  onPress={onNext}
                  style={[styles.pageButton, styles.pageRight]}
                  accessibilityLabel="Next"
                />
              ) : null}
            </View>

            {notice ? <Text style={styles.notice}>{notice}</Text> : null}
            {footer ? <View style={styles.footer}>{footer}</View> : null}
          </View>
        </View>
      </Portal.Host>
    </Modal>
  );
}

// Dark enough to stay legible over a white receipt, translucent enough that
// what is underneath is still visible.
const PAGE_BUTTON_TINT = "rgba(0,0,0,0.45)";

const styles = StyleSheet.create({
  backdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.5)",
    justifyContent: "flex-end",
    // A sliver of backdrop, so the sheet still reads as a sheet that can be
    // dismissed rather than as a new screen.
    paddingTop: 28,
  },
  sheet: {
    // Fills everything below that sliver. A fixed maxHeight left a receipt
    // squeezed into two-thirds of the screen with empty sheet beneath it —
    // the picture is the whole point of this view, so it gets the space.
    flex: 1,
    backgroundColor: colors.background,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
    paddingBottom: 24,
    overflow: "hidden",
  },
  content: {
    flex: 1,
    justifyContent: "center",
  },
  pageButton: {
    position: "absolute",
    // Half the 48pt hit area, to sit true to centre.
    top: "50%",
    marginTop: -24,
    margin: 0,
  },
  pageLeft: {
    left: spacing.sm,
  },
  pageRight: {
    right: spacing.sm,
  },
  grabber: {
    alignSelf: "center",
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.border,
    marginTop: spacing.sm,
    marginBottom: spacing.xs,
  },
  headerRow: {
    flexDirection: "row",
    alignItems: "center",
    paddingLeft: spacing.lg,
    paddingRight: spacing.xs,
    paddingVertical: spacing.xs,
    borderBottomWidth: 1,
    borderBottomColor: colors.surfaceBorder,
    gap: spacing.sm,
  },
  headerIcon: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
  },
  headerText: {
    flex: 1,
    minWidth: 0,
  },
  title: {
    ...typography.subtitle2,
    color: colors.text,
  },
  subtitle: {
    ...typography.caption,
    color: colors.textSecondary,
    marginTop: 1,
  },
  notice: {
    ...typography.caption,
    color: colors.textSecondary,
    textAlign: "center",
    paddingTop: spacing.xs,
  },
  footer: {
    paddingTop: spacing.xs,
    paddingHorizontal: spacing.md,
    // Deliberately NOT alignItems:"center" — that sizes the footer's child to
    // its content, so a button asking to flex across the row could not.
  },
});
