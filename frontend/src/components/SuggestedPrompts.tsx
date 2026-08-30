import React from "react";
import { StyleSheet, View } from "react-native";
import { Text, Chip } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { colors, typography, spacing } from "../styles/theme";

// The same set the attach sheet offers, so the empty state can reach every way
// in. It used to be missing "library" entirely, which meant the one screen a
// new user actually reads could not offer an existing photo.
export type PromptAction = "camera" | "library" | "document";

interface SuggestedPromptsProps {
  onSelectPrompt: (prompt: string) => void;
  onAction?: (action: PromptAction) => void;
}

// Two is enough to teach that free-form questions work. A longer list reads as
// a menu of the only things you may ask, and crowds out the actions above it.
const PROMPTS = [
  { text: "What are my top spending categories?", icon: "chart-pie" as const },
  { text: "Show spending trends this month", icon: "trending-up" as const },
];

type Action = {
  text: string;
  icon: "camera-outline" | "image-outline" | "file-delimited-outline";
  action: PromptAction;
};

// Things you *do*, as opposed to things you ask. Kept visually distinct and
// listed first: nobody discovers "you can photograph a shelf tag" by reading a
// list of questions, and this screen is the only place it can be taught.
//
// That teaching lives in the subtitle rather than in the buttons. There were
// once two — "Check a price" and "Snap a receipt" — but both ran the same
// action, opening the same camera and letting the classifier decide. Offering
// one action twice only made people worry about choosing wrong.
const CAPTURE: Action = { text: "Snap a photo", icon: "camera-outline", action: "camera" };

const UPLOADS: Action[] = [
  { text: "Upload image", icon: "image-outline", action: "library" },
  // Named for what it takes. The upload allowlist is CSV and images only, so a
  // broader word like "document" would promise PDFs the API rejects.
  { text: "Upload statement", icon: "file-delimited-outline", action: "document" },
];

const renderAction = (a: Action, onAction: (action: PromptAction) => void) => (
  <Chip
    key={a.text}
    // Rendered explicitly rather than passed as a name: on a filled chip Paper
    // tints the icon from the theme, which puts the primary colour on a primary
    // background and hides it. textStyle only reaches the label, so it cannot
    // fix this.
    icon={({ size }) => <MaterialCommunityIcons name={a.icon} size={size} color="#fff" />}
    onPress={() => onAction(a.action)}
    style={styles.actionChip}
    textStyle={styles.actionChipText}
  >
    {a.text}
  </Chip>
);

export function SuggestedPrompts({ onSelectPrompt, onAction }: SuggestedPromptsProps) {
  return (
    <View style={styles.container}>
      <MaterialCommunityIcons
        name="chat-question-outline"
        size={48}
        color={colors.primaryMedium}
      />
      <Text style={styles.title}>Ask about your finances</Text>
      {/* Says what a photo can be, which is the one thing the buttons below
          cannot say for themselves. It stops short of restating them: they
          already name themselves, and echoing them just doubles the text. */}
      <Text style={styles.subtitle}>Snap a price tag or receipt, or just ask</Text>
      {onAction && (
        <>
          <View style={styles.grid}>
            {renderAction(CAPTURE, onAction)}
          </View>
          <View style={[styles.grid, styles.uploadGrid]}>
            {UPLOADS.map((a) => renderAction(a, onAction))}
          </View>
        </>
      )}
      <View style={[styles.grid, onAction && styles.askGrid]}>
        {PROMPTS.map((prompt) => (
          <Chip
            key={prompt.text}
            mode="outlined"
            icon={prompt.icon}
            onPress={() => onSelectPrompt(prompt.text)}
            style={styles.chip}
            textStyle={styles.chipText}
          >
            {prompt.text}
          </Chip>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: spacing.xxl,
  },
  title: {
    ...typography.h3,
    color: colors.text,
    marginTop: spacing.lg,
    marginBottom: spacing.sm,
  },
  subtitle: {
    ...typography.body2,
    color: colors.textSecondary,
    textAlign: "center",
    marginBottom: spacing.xxl,
    maxWidth: 300,
  },
  grid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "center",
    gap: spacing.sm,
    maxWidth: 360,
  },
  askGrid: {
    marginTop: spacing.lg,
  },
  uploadGrid: {
    marginTop: spacing.sm,
  },
  actionChip: {
    backgroundColor: colors.primary,
  },
  actionChipText: {
    ...typography.caption,
    color: "#fff",
    fontWeight: "600",
  },
  chip: {
    backgroundColor: colors.primaryFaded,
    borderColor: colors.primaryBorder,
  },
  chipText: {
    ...typography.caption,
    color: colors.text,
  },
});
