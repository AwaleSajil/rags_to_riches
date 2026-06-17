import React, { useState } from "react";
import { Image, Modal, Platform, Pressable, ScrollView, StyleSheet, useWindowDimensions, View } from "react-native";
import { IconButton, Text } from "react-native-paper";
import Markdown from "react-native-markdown-display";
import { PlotlyChart } from "./PlotlyChart";
import { TransactionConfirmCard } from "./TransactionConfirmCard";
import { ToolTrace } from "./ToolTrace";
import { colors, typography, spacing } from "../styles/theme";
import type { ChatMessage as ChatMessageType } from "../lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const { width: screenWidth, height: screenHeight } = useWindowDimensions();
  const hasCharts = message.charts && message.charts.length > 0;
  const [expandedImage, setExpandedImage] = useState<string | null>(null);

  // Scale receipt images based on screen width
  const imageWidth = Math.min(Math.floor(screenWidth * 0.45), 200);
  const imageHeight = Math.round(imageWidth * 1.4);

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <Text style={[styles.roleLabel, isUser ? styles.userLabel : styles.assistantLabel]}>
        {isUser ? "You" : "R2R"}
      </Text>
      <View style={[
        styles.bubble,
        isUser ? styles.userBubble : styles.assistantBubble,
        !isUser && hasCharts && styles.wideBubble,
      ]}>
        {message.content?.trim() ? (
          <Markdown
            style={isUser ? markdownStylesUser : markdownStylesAssistant}
          >
            {message.content}
          </Markdown>
        ) : hasCharts ? (
          <Text style={{ color: colors.textSecondary, fontSize: 14, marginBottom: 4 }}>
            Here's what I found:
          </Text>
        ) : null}
        {hasCharts && (
          <View>
            {message.charts!.map((chartJson, i) => (
              <PlotlyChart key={i} chartJson={chartJson} />
            ))}
            {Platform.OS !== "web" && (
              <Text style={styles.chartHint}>Tap a data point for details</Text>
            )}
          </View>
        )}
        {message.images && message.images.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.imageRow}>
            {message.images.map((url, i) => (
              <Pressable key={i} onPress={() => setExpandedImage(url)}>
                <Image
                  source={{ uri: url }}
                  style={[styles.receiptImage, { width: imageWidth, height: imageHeight }]}
                  resizeMode="contain"
                />
              </Pressable>
            ))}
          </ScrollView>
        )}
        {expandedImage && (
          <Modal visible transparent animationType="fade" onRequestClose={() => setExpandedImage(null)}>
            <Pressable style={styles.modalBackdrop} onPress={() => setExpandedImage(null)}>
              <View style={styles.modalHeader}>
                <IconButton icon="close" iconColor="#fff" size={28} onPress={() => setExpandedImage(null)} />
              </View>
              <Image
                source={{ uri: expandedImage }}
                style={{ width: screenWidth * 0.95, height: screenHeight * 0.8 }}
                resizeMode="contain"
              />
            </Pressable>
          </Modal>
        )}
        {message.pendingTransactions && message.pendingTransactions.length > 0 && (
          <View>
            {message.pendingTransactions.map((tx, i) => (
              <TransactionConfirmCard key={i} transaction={tx} />
            ))}
          </View>
        )}
        {!isUser && message.toolTraces && message.toolTraces.length > 0 && (
          <ToolTrace traces={message.toolTraces} />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
  },
  userContainer: {
    alignItems: "flex-end",
  },
  assistantContainer: {
    alignItems: "flex-start",
  },
  roleLabel: {
    ...typography.caption,
    marginBottom: 2,
    marginHorizontal: spacing.xs,
  },
  userLabel: {
    color: colors.textTertiary,
  },
  assistantLabel: {
    color: colors.primary,
  },
  bubble: {
    maxWidth: "85%",
    borderRadius: 16,
    padding: 16,
  },
  userBubble: {
    backgroundColor: colors.userBubble,
    borderBottomRightRadius: 6,
  },
  assistantBubble: {
    backgroundColor: colors.assistantBubble,
    borderBottomLeftRadius: 6,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  wideBubble: {
    maxWidth: "98%",
    paddingHorizontal: 8,
  },
  imageRow: {
    marginTop: spacing.sm,
  },
  receiptImage: {
    borderRadius: 8,
    marginRight: 8,
    backgroundColor: colors.surfaceBorder,
  },
  chartHint: {
    fontSize: 11,
    color: colors.textTertiary,
    textAlign: "center",
    marginTop: 4,
    marginBottom: 2,
  },
  modalBackdrop: {
    flex: 1,
    backgroundColor: "rgba(0,0,0,0.92)",
    justifyContent: "center",
    alignItems: "center",
  },
  modalHeader: {
    position: "absolute",
    top: 40,
    right: 8,
    zIndex: 1,
  },
});

const markdownStylesUser = {
  body: { color: "#fff", fontSize: 15 },
  paragraph: { marginBottom: 6, marginTop: 0 },
  link: { color: "#c7d2fe" },
  code_inline: { backgroundColor: "rgba(255,255,255,0.15)", color: "#fff", borderRadius: 4, paddingHorizontal: 4 },
};

const markdownStylesAssistant = {
  body: { color: colors.text, fontSize: 15 },
  paragraph: { marginBottom: 6, marginTop: 0 },
  link: { color: colors.primary },
  code_inline: { backgroundColor: colors.primaryLight, color: colors.primaryDark, borderRadius: 4, paddingHorizontal: 4 },
  code_block: { backgroundColor: "#f1f5f9", borderRadius: 8, padding: 12 },
};
