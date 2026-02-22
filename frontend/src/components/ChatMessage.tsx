import React from "react";
import { Image, Platform, ScrollView, StyleSheet, useWindowDimensions, View } from "react-native";
import { Text } from "react-native-paper";
import Markdown from "react-native-markdown-display";
import { PlotlyChart } from "./PlotlyChart";
import { colors, typography, spacing } from "../styles/theme";
import type { ChatMessage as ChatMessageType } from "../lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";
  const { width: screenWidth } = useWindowDimensions();
  const hasCharts = message.charts && message.charts.length > 0;

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
              <Image
                key={i}
                source={{ uri: url }}
                style={[styles.receiptImage, { width: imageWidth, height: imageHeight }]}
                resizeMode="contain"
              />
            ))}
          </ScrollView>
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
