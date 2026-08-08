import React, { useState } from "react";
import { Image, Platform, Pressable, StyleSheet, useWindowDimensions, View } from "react-native";
import { IconButton, Text } from "react-native-paper";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import Markdown from "react-native-markdown-display";
import { PlotlyChart } from "./PlotlyChart";
import { PhotoViewer } from "./PhotoViewer";
import { asAttachment, photoFilename, saveToDevice } from "../lib/download";
import { sourceVisual } from "../lib/sourceVisual";
import { TransactionConfirmCard } from "./TransactionConfirmCard";
import { CorrectionConfirmCard } from "./CorrectionConfirmCard";
import { KindPromptCard } from "./KindPromptCard";
import { PriceTagConfirmCard } from "./PriceTagConfirmCard";
import { ToolTrace } from "./ToolTrace";
import { colors, typography, spacing } from "../styles/theme";
import type { ChatMessage as ChatMessageType } from "../lib/types";
import type { CaptureResult, PriceTagDraft } from "../services/captureService";
import type { PriceComparison } from "../services/priceService";

interface ChatMessageProps {
  message: ChatMessageType;
  onKindResolved?: (fileId: string, result: CaptureResult) => void;
  onConfirmPriceTag?: (
    fileId: string,
    draft: PriceTagDraft & { tag_index: number }
  ) => Promise<PriceComparison | null>;
  onDiscardCapture?: (fileId: string) => Promise<void>;
  onAskAboutPrices?: (
    drafts: (PriceTagDraft & { tag_index: number })[],
    fileId?: string
  ) => void;
  onReviewReceipt?: (fileId: string) => void;
  /** Send this prompt again. Offered on your own messages so a failed or
   *  unsatisfying answer can be retried without retyping. */
  onRerun?: (text: string) => void;
  /** Suppresses rerun while a turn is in flight — sendMessage would ignore it
   *  anyway, so an enabled button that silently does nothing is worse. */
  rerunDisabled?: boolean;
}

/** Routes a captured photo to the card that matches what it turned out to be. */
function CaptureCard({
  capture,
  onKindResolved,
  onConfirmPriceTag,
  onDiscardCapture,
  onAskAboutPrices,
  onReviewReceipt,
}: {
  capture: CaptureResult;
  onKindResolved?: (fileId: string, result: CaptureResult) => void;
  onConfirmPriceTag?: (
    fileId: string,
    draft: PriceTagDraft & { tag_index: number }
  ) => Promise<PriceComparison | null>;
  onDiscardCapture?: (fileId: string) => Promise<void>;
  onAskAboutPrices?: (
    drafts: (PriceTagDraft & { tag_index: number })[],
    fileId?: string
  ) => void;
  onReviewReceipt?: (fileId: string) => void;
}) {
  if (capture.kind === "price_tag") {
    return (
      <PriceTagConfirmCard
        fileId={capture.file_id}
        // One photo can show a tag per product, so the draft carries a list.
        // Older rows stored a single flat tag; wrap those rather than dropping them.
        tags={capture.draft?.tags ?? (capture.draft?.item_description ? [capture.draft] : [])}
        onConfirm={(draft) => onConfirmPriceTag!(capture.file_id, draft)}
        onDiscard={() => onDiscardCapture!(capture.file_id)}
        onAsk={onAskAboutPrices}
        // Where the photo was taken, resolved on the device. Recorded with the
        // price because the same item costs different amounts at different shops.
        place={capture.location ?? null}
      />
    );
  }
  if (capture.kind === "receipt") {
    return (
      <View style={styles.receiptPrompt}>
        <Text style={styles.receiptPromptText}>
          {capture.draft?.merchant_name || "Receipt"}
          {capture.draft?.total_amount != null ? ` · $${capture.draft.total_amount}` : ""}
        </Text>
        <IconButton
          icon="arrow-right"
          mode="contained"
          size={18}
          iconColor="#fff"
          containerColor={colors.primary}
          onPress={() => onReviewReceipt?.(capture.file_id)}
          accessibilityLabel="Review this receipt"
        />
      </View>
    );
  }
  // Undecided — ask rather than guess.
  return (
    <KindPromptCard
      fileId={capture.file_id}
      onResolved={(result) => onKindResolved?.(capture.file_id, result)}
    />
  );
}

export function ChatMessage({
  message,
  onKindResolved,
  onConfirmPriceTag,
  onDiscardCapture,
  onAskAboutPrices,
  onReviewReceipt,
  onRerun,
  rerunDisabled,
}: ChatMessageProps) {
  const isUser = message.role === "user";
  const { width: screenWidth } = useWindowDimensions();
  const hasCharts = message.charts && message.charts.length > 0;
  const [expandedImage, setExpandedImage] = useState<string | null>(null);
  const [downloading, setDownloading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);

  // Scale receipt images based on screen width
  const imageWidth = Math.min(Math.floor(screenWidth * 0.45), 200);
  const imageHeight = Math.round(imageWidth * 1.4);

  // Every photo in this message, in the order they are shown, so the viewer can
  // page between them. Attached receipts first, then anything just captured.
  const photos = [...(message.images ?? []), ...(message.localImages ?? [])];
  const photoIndex = expandedImage ? photos.indexOf(expandedImage) : -1;

  // A photo taken moments ago is already on this device; only a receipt the
  // agent attached from storage is worth offering to save.
  const isRemote = expandedImage ? !expandedImage.startsWith("file:") : false;

  const handleDownload = async () => {
    if (!expandedImage || downloading) return;
    setDownloading(true);
    setNotice(null);
    try {
      const name = photoFilename(expandedImage);
      await saveToDevice(asAttachment(expandedImage, name), name);
      setNotice("Saved to your downloads.");
    } catch (e: any) {
      setNotice(e?.message || "Could not download that photo.");
    } finally {
      setDownloading(false);
    }
  };

  return (
    <View style={[styles.container, isUser ? styles.userContainer : styles.assistantContainer]}>
      <Text style={[styles.roleLabel, isUser ? styles.userLabel : styles.assistantLabel]}>
        {isUser ? "You" : "R2R"}
      </Text>
      {/* The rerun button sits OUTSIDE the bubble, to its left. Inside it would
          either steal width from the text or sit on the indigo fill where a
          plain icon has poor contrast. */}
      <View style={isUser ? styles.userRow : undefined}>
      {isUser && onRerun && !!message.content?.trim() && (
        <IconButton
          icon="refresh"
          size={16}
          iconColor={colors.textTertiary}
          disabled={rerunDisabled}
          onPress={() => onRerun(message.content)}
          style={styles.rerunButton}
          accessibilityLabel="Ask this again"
        />
      )}
      <View style={[
        styles.bubble,
        isUser ? styles.userBubble : styles.assistantBubble,
        !isUser && hasCharts && styles.wideBubble,
        message.isError && styles.errorBubble,
      ]}>
        {message.isError ? (
          <View style={styles.errorRow}>
            <MaterialCommunityIcons name="alert-circle-outline" size={18} color="#b45309" />
            <Text style={styles.errorText}>{message.content}</Text>
          </View>
        ) : message.content?.trim() ? (
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
          <View style={styles.imageRow}>
            {message.images.map((url, i) => (
              <Pressable key={i} onPress={() => setExpandedImage(url)}>
                <Image
                  source={{ uri: url }}
                  style={[styles.receiptImage, { width: imageWidth, height: imageHeight }]}
                  resizeMode="contain"
                />
              </Pressable>
            ))}
          </View>
        )}
        {/* The same viewer the Files tab and the transaction screen use, so a
            receipt tapped here offers the same things: rotate, download, and
            paging when a reply attached more than one. */}
        <PhotoViewer
          visible={!!expandedImage}
          uri={expandedImage}
          onClose={() => {
            setExpandedImage(null);
            setNotice(null);
          }}
          title={expandedImage ? photoFilename(expandedImage) : ""}
          subtitle={isRemote ? "Receipt" : "Just captured"}
          icon={isRemote ? sourceVisual("receipt") : undefined}
          onDownload={isRemote ? handleDownload : undefined}
          downloading={downloading}
          notice={notice}
          onPrevious={
            photoIndex > 0 ? () => setExpandedImage(photos[photoIndex - 1]) : undefined
          }
          onNext={
            photoIndex >= 0 && photoIndex < photos.length - 1
              ? () => setExpandedImage(photos[photoIndex + 1])
              : undefined
          }
          position={photoIndex >= 0 ? { index: photoIndex, total: photos.length } : undefined}
        />
        {message.localImages && message.localImages.length > 0 && (
          <View style={styles.imageRow}>
            {message.localImages.map((uri, i) => (
              <Pressable key={i} onPress={() => setExpandedImage(uri)}>
                <Image
                  source={{ uri }}
                  style={[styles.receiptImage, { width: imageWidth, height: imageHeight }]}
                  resizeMode="cover"
                />
              </Pressable>
            ))}
          </View>
        )}
        {message.capture && (
          <CaptureCard
            capture={message.capture}
            onKindResolved={onKindResolved}
            onConfirmPriceTag={onConfirmPriceTag}
            onDiscardCapture={onDiscardCapture}
            onAskAboutPrices={onAskAboutPrices}
            onReviewReceipt={onReviewReceipt}
          />
        )}
        {message.pendingCorrections && message.pendingCorrections.length > 0 && (
          <View>
            {message.pendingCorrections.map((fix, i) => (
              <CorrectionConfirmCard key={`${fix.row_id}-${i}`} correction={fix} />
            ))}
          </View>
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
  userRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "flex-end",
    // The bubble's own maxWidth is a percentage of this row, so the row has to
    // be allowed to shrink or a long prompt pushes the button off-screen.
    flexShrink: 1,
  },
  rerunButton: {
    margin: 0,
    marginRight: 2,
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
  errorBubble: {
    backgroundColor: "#FEF3C7",
    borderColor: "#FCD34D",
    borderWidth: 1,
  },
  errorRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 8,
  },
  errorText: {
    flex: 1,
    color: "#92400e",
    fontSize: 14,
    lineHeight: 20,
  },
  // A wrapping row, not a horizontal ScrollView. A ScrollView grows to fill the
  // space it is given in both directions, so one photo turned the whole bubble
  // into a full-height block of colour with the picture stranded at the top.
  // Wrapping also reads better on a phone than sideways scrolling.
  imageRow: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 8,
    marginTop: spacing.sm,
  },
  receiptImage: {
    borderRadius: 8,
    backgroundColor: colors.surfaceBorder,
  },
  chartHint: {
    fontSize: 11,
    color: colors.textTertiary,
    textAlign: "center",
    marginTop: 4,
    marginBottom: 2,
  },
  receiptPrompt: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    gap: spacing.sm,
    marginTop: spacing.sm,
    paddingLeft: spacing.md,
    paddingRight: spacing.xs,
    paddingVertical: spacing.xs,
    backgroundColor: colors.surfaceSubtle,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.surfaceBorder,
  },
  receiptPromptText: {
    ...typography.body2,
    color: colors.text,
    flexShrink: 1,
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
