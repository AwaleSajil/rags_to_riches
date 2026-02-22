import React from "react";
import { Image, ScrollView, StyleSheet, View } from "react-native";
import { Text } from "react-native-paper";
import Markdown from "react-native-markdown-display";
import { ToolTrace } from "./ToolTrace";
import { PlotlyChart } from "./PlotlyChart";
import { colors } from "../styles/theme";
import type { ChatMessage as ChatMessageType } from "../lib/types";

interface ChatMessageProps {
  message: ChatMessageType;
}

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === "user";

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <View style={[styles.bubble, isUser ? styles.userBubble : styles.assistantBubble]}>
        {message.toolTraces && message.toolTraces.length > 0 && (
          <ToolTrace traces={message.toolTraces} />
        )}
        <Markdown
          style={isUser ? markdownStylesUser : markdownStylesAssistant}
        >
          {message.content}
        </Markdown>
        {message.charts?.map((chartJson, i) => (
          <PlotlyChart key={i} chartJson={chartJson} />
        ))}
        {message.images && message.images.length > 0 && (
          <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.imageRow}>
            {message.images.map((url, i) => (
              <Image
                key={i}
                source={{ uri: url }}
                style={styles.receiptImage}
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
    paddingHorizontal: 12,
    paddingVertical: 4,
  },
  userContainer: {
    alignItems: "flex-end",
  },
  assistantContainer: {
    alignItems: "flex-start",
  },
  bubble: {
    maxWidth: "85%",
    borderRadius: 16,
    padding: 14,
  },
  userBubble: {
    backgroundColor: colors.userBubble,
    borderBottomRightRadius: 4,
  },
  assistantBubble: {
    backgroundColor: colors.assistantBubble,
    borderBottomLeftRadius: 4,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  imageRow: {
    marginTop: 8,
  },
  receiptImage: {
    width: 200,
    height: 280,
    borderRadius: 8,
    marginRight: 8,
    backgroundColor: colors.surfaceBorder,
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
