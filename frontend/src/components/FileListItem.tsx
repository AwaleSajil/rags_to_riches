import React from "react";
import { Pressable, StyleSheet, View } from "react-native";
import { Text, IconButton } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import type { FileItem } from "../lib/types";

interface FileListItemProps {
  file: FileItem;
  onDelete: (file: FileItem) => void;
  onPress?: (file: FileItem) => void;
  onToggleVisibility?: (file: FileItem) => void;
  isTogglingVisibility?: boolean;
}

function FileListItemComponent({ file, onDelete, onPress, onToggleVisibility, isTogglingVisibility }: FileListItemProps) {
  const isCSV = file.type === "csv";
  const iconName = isCSV ? "file-delimited" : "file-image";
  const iconColor = isCSV ? colors.success : colors.warning;

  return (
    <View style={styles.container}>
      <Pressable
        style={({ pressed }) => [styles.pressArea, pressed && styles.pressed]}
        onPress={() => onPress?.(file)}
      >
        <View style={[styles.iconCircle, { backgroundColor: isCSV ? "#dcfce7" : "#fef3c7" }]}>
          <MaterialCommunityIcons name={iconName} size={20} color={iconColor} />
        </View>
        <View style={styles.textContainer}>
          <Text style={styles.filename} numberOfLines={1}>
            {file.filename}
          </Text>
          <Text style={styles.date}>
            {file.upload_date?.slice(0, 10) ?? "N/A"} · tap to preview
          </Text>
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
