import React, { useState } from "react";
import { StyleSheet, View } from "react-native";
import { Text, Button, Divider } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";
import { confirmTransaction } from "../services/transactionService";
import type { PendingTransaction } from "../lib/types";
import { createLogger } from "../lib/logger";
import { formatDate, money } from "../lib/format";

const log = createLogger("TransactionConfirmCard");

interface Props {
  transaction: PendingTransaction;
}

type CardStatus = "pending" | "confirming" | "confirmed" | "cancelled" | "error";

export function TransactionConfirmCard({ transaction }: Props) {
  const [status, setStatus] = useState<CardStatus>("pending");
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleConfirm = async () => {
    setStatus("confirming");
    try {
      await confirmTransaction(transaction);
      setStatus("confirmed");
      log.info("Transaction confirmed", { description: transaction.description });
    } catch (e: any) {
      log.error("Transaction confirmation failed", e);
      setStatus("error");
      setErrorMsg(e.message || "Failed to save");
    }
  };

  const handleCancel = () => {
    setStatus("cancelled");
    log.info("Transaction cancelled by user", { description: transaction.description });
  };

  const iconName =
    status === "confirmed"
      ? "check-circle"
      : status === "cancelled"
        ? "close-circle"
        : "cash-plus";

  const iconColor =
    status === "confirmed"
      ? colors.success
      : status === "cancelled"
        ? colors.textTertiary
        : colors.primary;

  const headerText =
    status === "confirmed"
      ? "Transaction Saved"
      : status === "cancelled"
        ? "Transaction Cancelled"
        : "Confirm Transaction";

  return (
    <View style={styles.card}>
      <View style={styles.header}>
        <MaterialCommunityIcons name={iconName} size={20} color={iconColor} />
        <Text style={styles.headerText}>{headerText}</Text>
      </View>

      <Divider style={styles.divider} />

      <View style={styles.details}>
        <DetailRow label="Description" value={transaction.description} />
        <DetailRow label="Amount" value={money(transaction.amount)} highlight />
        <DetailRow label="Date" value={formatDate(transaction.trans_date)} />
        <DetailRow label="Category" value={transaction.category} />
        {transaction.merchant_name !== transaction.description && (
          <DetailRow label="Merchant" value={transaction.merchant_name} />
        )}
      </View>

      {status === "pending" && (
        <View style={styles.actions}>
          <Button
            mode="outlined"
            onPress={handleCancel}
            style={styles.cancelBtn}
            textColor={colors.textSecondary}
          >
            Cancel
          </Button>
          <Button
            mode="contained"
            onPress={handleConfirm}
            style={styles.confirmBtn}
            buttonColor={colors.success}
          >
            Confirm
          </Button>
        </View>
      )}

      {status === "confirming" && (
        <View style={styles.actions}>
          <Button mode="contained" loading disabled style={styles.confirmBtn} buttonColor={colors.success}>
            Saving...
          </Button>
        </View>
      )}

      {status === "error" && (
        <View>
          <Text style={styles.errorText}>{errorMsg}</Text>
          <View style={styles.actions}>
            <Button mode="contained" onPress={handleConfirm} style={styles.confirmBtn} buttonColor={colors.primary}>
              Retry
            </Button>
          </View>
        </View>
      )}
    </View>
  );
}

function DetailRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <View style={styles.detailRow}>
      <Text style={styles.detailLabel}>{label}</Text>
      <Text style={[styles.detailValue, highlight && styles.highlightValue]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  card: {
    backgroundColor: colors.primaryLight,
    borderRadius: 12,
    padding: spacing.lg,
    marginTop: spacing.sm,
    borderWidth: 1,
    borderColor: colors.primaryBorder,
  },
  header: {
    flexDirection: "row",
    alignItems: "center",
    gap: spacing.sm,
  },
  headerText: {
    ...typography.subtitle2,
    color: colors.primary,
  },
  divider: {
    marginVertical: spacing.sm,
    backgroundColor: colors.primaryBorder,
  },
  details: {
    gap: spacing.xs,
  },
  detailRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    paddingVertical: 2,
  },
  detailLabel: {
    ...typography.caption,
    color: colors.textSecondary,
  },
  detailValue: {
    ...typography.body2,
    color: colors.text,
    fontWeight: "600",
  },
  highlightValue: {
    color: colors.primaryDark,
    fontSize: 16,
    fontWeight: "700",
  },
  actions: {
    flexDirection: "row",
    justifyContent: "flex-end",
    gap: spacing.sm,
    marginTop: spacing.md,
  },
  cancelBtn: {
    borderColor: colors.border,
  },
  confirmBtn: {
    minWidth: 100,
  },
  errorText: {
    ...typography.caption,
    color: colors.error,
    marginTop: spacing.xs,
  },
});
