import React from "react";
import { Button, Dialog, Portal, Text } from "react-native-paper";
import { colors } from "../styles/theme";

interface DeleteConfirmModalProps {
  visible: boolean;
  filename: string;
  onConfirm: () => void;
  onDismiss: () => void;
}

export function DeleteConfirmModal({
  visible,
  filename,
  onConfirm,
  onDismiss,
}: DeleteConfirmModalProps) {
  return (
    <Portal>
      <Dialog visible={visible} onDismiss={onDismiss} style={{ borderRadius: 16 }}>
        <Dialog.Title>Delete File</Dialog.Title>
        <Dialog.Content>
          <Text>
            Are you sure you want to delete <Text style={{ fontWeight: "700" }}>{filename}</Text>?
            This permanently removes it from Cloud Storage, the SQL Database, and the Vector Index.
          </Text>
        </Dialog.Content>
        <Dialog.Actions>
          <Button onPress={onDismiss} textColor={colors.textSecondary}>
            Cancel
          </Button>
          <Button
            onPress={onConfirm}
            textColor="#ffffff"
            mode="contained"
            buttonColor={colors.error}
            style={{ borderRadius: 8 }}
          >
            Delete
          </Button>
        </Dialog.Actions>
      </Dialog>
    </Portal>
  );
}
