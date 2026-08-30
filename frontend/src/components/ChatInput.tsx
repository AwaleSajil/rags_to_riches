import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import { TextInput, IconButton, Portal, Modal, List, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, spacing, typography } from "../styles/theme";

export type ChatAttachAction = "camera" | "library" | "document";

interface ChatInputProps {
  onSend: (message: string) => void;
  onAttach?: (action: ChatAttachAction) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, onAttach, disabled }: ChatInputProps) {
  const [text, setText] = useState("");
  const [sheetOpen, setSheetOpen] = useState(false);

  const handleSend = () => {
    if (text.trim() && !disabled) {
      onSend(text.trim());
      setText("");
    }
  };

  const choose = (action: ChatAttachAction) => {
    setSheetOpen(false);
    onAttach?.(action);
  };

  return (
    <View style={styles.container}>
      {onAttach && (
        <>
          {/* A bare camera icon reads as "receipts only" and leaves CSV upload
              stranded in another tab. The familiar "+" sheet covers every way
              data gets into the app from one place. */}
          <IconButton
            icon="plus"
            mode="contained-tonal"
            iconColor={colors.primary}
            containerColor={colors.primaryFaded}
            size={22}
            onPress={() => setSheetOpen(true)}
            disabled={disabled}
            style={styles.attachButton}
            accessibilityLabel="Add a photo or file"
          />
          <Portal>
            <Modal
              visible={sheetOpen}
              onDismiss={() => setSheetOpen(false)}
              contentContainerStyle={styles.sheet}
            >
              <View style={styles.sheetHandle} />
              <Text style={styles.sheetTitle}>Add to this chat</Text>
              <List.Item
                title="Take a photo"
                description="A price tag or a receipt — we'll work out which"
                left={() => <SheetIcon name="camera-outline" />}
                onPress={() => choose("camera")}
              />
              <List.Item
                title="Choose from library"
                description="Pick an existing photo"
                left={() => <SheetIcon name="image-outline" />}
                onPress={() => choose("library")}
              />
              <List.Item
                title="Upload a statement"
                description="A CSV export from your bank"
                left={() => <SheetIcon name="file-delimited-outline" />}
                onPress={() => choose("document")}
              />
            </Modal>
          </Portal>
        </>
      )}
      <TextInput
        mode="outlined"
        placeholder="Ask about your spending..."
        value={text}
        onChangeText={setText}
        onSubmitEditing={handleSend}
        disabled={disabled}
        multiline
        style={styles.input}
        outlineStyle={styles.outline}
        contentStyle={styles.inputContent}
        textColor={colors.text}
        dense
      />
      <IconButton
        icon="send"
        mode="contained"
        iconColor="#fff"
        containerColor={colors.primary}
        size={24}
        onPress={handleSend}
        disabled={disabled || !text.trim()}
        style={styles.sendButton}
      />
    </View>
  );
}

function SheetIcon({ name }: { name: React.ComponentProps<typeof MaterialCommunityIcons>["name"] }) {
  return (
    <View style={styles.sheetIcon}>
      <MaterialCommunityIcons name={name} size={22} color={colors.primary} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: "row",
    alignItems: "flex-end",
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.surfaceBorder,
  },
  attachButton: {
    margin: 0,
    marginRight: spacing.xs,
    marginBottom: 2,
  },
  sheet: {
    backgroundColor: colors.surface,
    marginTop: "auto",
    marginHorizontal: 0,
    paddingBottom: spacing.xl,
    paddingTop: spacing.sm,
    borderTopLeftRadius: 20,
    borderTopRightRadius: 20,
  },
  sheetHandle: {
    alignSelf: "center",
    width: 36,
    height: 4,
    borderRadius: 2,
    backgroundColor: colors.surfaceBorder,
    marginBottom: spacing.sm,
  },
  sheetTitle: {
    ...typography.caption,
    color: colors.textSecondary,
    paddingHorizontal: spacing.lg,
    paddingBottom: spacing.xs,
  },
  sheetIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.primaryFaded,
    alignItems: "center",
    justifyContent: "center",
    marginLeft: spacing.md,
  },
  input: {
    flex: 1,
    backgroundColor: colors.surfaceSubtle,
    marginRight: spacing.sm,
    maxHeight: 120,
  },
  outline: {
    borderRadius: 12,
    borderColor: colors.border,
  },
  inputContent: {
    paddingVertical: spacing.sm,
  },
  sendButton: {
    margin: 0,
    marginBottom: 2,
  },
});
