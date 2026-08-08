import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Text, IconButton } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import { fileVisual } from "../lib/sourceVisual";
import type { FileItem } from "../lib/types";

interface FileListItemProps {
  file: FileItem;
  onDelete: (file: FileItem) => void;
  onPress?: (file: FileItem) => void;
  onToggleVisibility?: (file: FileItem) => void;
  isTogglingVisibility?: boolean;
}

function FileListItemComponent({ file, onDelete, onPress, onToggleVisibility, isTogglingVisibility }: FileListItemProps) {
  // Shared with the Transactions tab, so one receipt looks like the same thing
  // in both places. This also tells a price tag apart from a receipt, which the
  // generic "file-image" never did even though `kind` has always said which.
  const visual = fileVisual(file);

  return (
    <View style={styles.container}>
      <Pressable
        style={({ pressed }) => [styles.pressArea, pressed && styles.pressed]}
        onPress={() => onPress?.(file)}
      >
        <View style={[styles.iconCircle, { backgroundColor: visual.background }]}>
          <MaterialCommunityIcons name={visual.icon} size={20} color={visual.color} />
        </View>
        <View style={styles.textContainer}>
          <Text style={styles.filename} numberOfLines={1}>
            {file.filename}
          </Text>
          <Text style={styles.date}>
            {file.upload_date?.slice(0, 10) ?? "N/A"} · tap to preview
          </Text>
          {/* Only ever shown for receipts — is_verified is absent on CSVs and
              price tags, and `false` is the state that needs saying. A verified
              receipt gets a quiet tick rather than a matching badge, so the
              list reads as "these two need you" instead of a wall of labels. */}
          {file.is_verified === false && (
            <View style={styles.needsReviewBadge}>
              <MaterialCommunityIcons
                name="alert-circle-outline"
                size={12}
                color={colors.warning}
              />
              <Text style={styles.needsReviewText}>Not counted — needs review</Text>
            </View>
          )}
          {file.is_verified === true && (
            <View style={styles.needsReviewBadge}>
              <MaterialCommunityIcons name="check-circle" size={12} color={colors.success} />
              <Text style={styles.verifiedText}>Verified</Text>
            </View>
          )}
          {/* The bank recorded this purchase too. Worth saying here because the
              transactions list shows the pair as ONE row: without it, a receipt
              sitting beside a statement covering the same shop reads as a
              duplicate someone is about to delete. */}
          {file.is_linked && (
            <View style={styles.needsReviewBadge}>
              <MaterialCommunityIcons
                name="link-variant"
                size={12}
                color={colors.textSecondary}
              />
              <Text style={styles.linkedText}>Also on your statement</Text>
            </View>
          )}
        </View>
      </Pressable>
      <IconButton
        icon={file.is_hidden ? "eye-off-outline" : "eye-outline"}
        iconColor={colors.textSecondary}
        size={20}
        onPress={() => onToggleVisibility?.(file)}
        disabled={isTogglingVisibility}
        accessibilityLabel={file.is_hidden ? "Show transactions from this file" : "Hide transactions from this file"}
      />
      <IconButton
        icon="trash-can-outline"
        iconColor={colors.error}
        size={20}
        onPress={() => onDelete(file)}
      />
    </View>
  );
}

// The list re-renders on every keystroke in the search box and on every upload
// progress tick. Without this, each of those re-renders every mounted row even
// though nothing about the row changed.
export const FileListItem = React.memo(FileListItemComponent);

const styles = StyleSheet.create({
  needsReviewBadge: {
    flexDirection: "row",
    alignItems: "center",
    gap: 3,
    marginTop: 2,
  },
  needsReviewText: {
    ...typography.caption,
    color: colors.warning,
    fontWeight: "600",
  },
  verifiedText: {
    ...typography.caption,
    color: colors.textTertiary,
  },
  linkedText: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  container: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
    borderRadius: 12,
    paddingLeft: spacing.md,
    paddingRight: spacing.xs,
    paddingVertical: spacing.xs,
    marginBottom: spacing.sm,
  },
  pressArea: {
    flex: 1,
    flexDirection: "row",
    alignItems: "center",
    paddingVertical: spacing.sm,
  },
  pressed: {
    opacity: 0.6,
  },
  iconCircle: {
    width: 36,
    height: 36,
    borderRadius: 18,
    alignItems: "center",
    justifyContent: "center",
    marginRight: spacing.md,
  },
  textContainer: {
    flex: 1,
  },
  filename: {
    ...typography.subtitle2,
    color: colors.text,
  },
  date: {
    ...typography.caption,
    color: colors.textSecondary,
  },
});
