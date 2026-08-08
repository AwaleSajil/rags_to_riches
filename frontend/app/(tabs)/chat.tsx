import React, { useRef, useEffect, useCallback, useState } from "react";
import { StyleSheet, View, FlatList, KeyboardAvoidingView, Platform, useWindowDimensions } from "react-native";
import { Banner, Snackbar, Text } from "react-native-paper";
import { useRouter, useFocusEffect, useLocalSearchParams } from "expo-router";
import { useCameraPermissions } from "expo-camera";
import * as DocumentPicker from "expo-document-picker";
import { ChatMessage } from "../../src/components/ChatMessage";
import { ChatInput } from "../../src/components/ChatInput";
import type { ChatAttachAction } from "../../src/components/ChatInput";
import { CameraCapture } from "../../src/components/CameraCapture";
import { SuggestedPrompts } from "../../src/components/SuggestedPrompts";
import { savePriceObservation } from "../../src/services/priceService";
import { discardCapture } from "../../src/services/captureService";
import type { CaptureResult, PriceTagDraft } from "../../src/services/captureService";
import { TypingIndicator } from "../../src/components/TypingIndicator";
import { ConversationDrawer } from "../../src/components/ConversationDrawer";
import { useChat } from "../../src/hooks/useChat";
import { useConversations } from "../../src/hooks/useConversations";
import { useFiles } from "../../src/hooks/useFiles";
import { useChatSession } from "../../src/providers/ChatSessionProvider";
import { colors, spacing } from "../../src/styles/theme";
import { createLogger } from "../../src/lib/logger";
import { openReview } from "../../src/lib/reviewRouting";
import type { ChatMessage as ChatMessageType } from "../../src/lib/types";

const log = createLogger("ChatScreen");

// Memoized row renderer — prevents re-rendering every message when a new one arrives
const MemoizedChatMessage = React.memo(ChatMessage);

const MAX_CHAT_WIDTH = 720;

export default function ChatScreen() {
  log.debug("ChatScreen rendered");
  const {
    messages,
    isStreaming,
    isReceivingText,
    currentToolTraces,
    sendMessage,
    sendPhoto,
    showCapture,
    updateCapture,
    conversationId,
    loadConversation,
    newConversation,
  } = useChat();
  const { files, hasLoaded: filesLoaded, loadFiles, uploadFiles } = useFiles();
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraPermission, requestCameraPermission] = useCameraPermissions();
  const [snack, setSnack] = useState<string | null>(null);
  const {
    conversations,
    isLoading: convLoading,
    refresh: refreshConversations,
    remove: removeConversation,
  } = useConversations();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const flatListRef = useRef<FlatList>(null);

  // Files + conversations are loaded per-screen, so re-check on focus —
  // otherwise the "no data" banner and chat list stay stale after changes elsewhere.
  useFocusEffect(
    useCallback(() => {
      loadFiles();
      refreshConversations();
    }, [loadFiles, refreshConversations])
  );

  // After a message finishes streaming, refresh the list (new chat / title / order).
  useEffect(() => {
    if (!isStreaming) refreshConversations();
  }, [isStreaming]);

  // Wire the native header's ☰ / ＋ buttons to this screen's actions.
  const { setHandlers } = useChatSession();
  useEffect(() => {
    setHandlers({
      onMenu: () => setDrawerOpen(true),
      onNewChat: () => newConversation(),
    });
  }, [setHandlers, newConversation]);

  const router = useRouter();
  const { width: screenWidth } = useWindowDimensions();

  // A photo read by the batch upload path arrives here as a file id: only a
  // receipt has its own review screen, so price tags and unclassified photos are
  // routed to this tab, where their card is the review step. Without this the
  // param was written and never read, and a price tag uploaded from the Files
  // tab landed on an empty chat window with no way to confirm it.
  const { captureFileId } = useLocalSearchParams<{ captureFileId?: string }>();
  useEffect(() => {
    if (!captureFileId) return;
    void showCapture(captureFileId);
    // Cleared so coming back to this tab later does not re-open the same photo.
    router.setParams({ captureFileId: undefined });
  }, [captureFileId, showCapture, router]);

  const isWide = Platform.OS === "web" && screenWidth > MAX_CHAT_WIDTH;
  const fileCount = files.length;

  // Auto-scroll to bottom on new messages (longer delay for chart WebViews to mount)
  useEffect(() => {
    log.debug("Messages/streaming state changed", {
      messageCount: messages.length,
      isStreaming,
      fileCount: files.length,
    });
    if (messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      const hasChart = lastMsg?.charts && lastMsg.charts.length > 0;
      // Charts need more time to mount their WebView before we can scroll accurately
      const delay = hasChart ? 300 : 100;
      setTimeout(() => flatListRef.current?.scrollToEnd({ animated: true }), delay);
    }
  }, [messages.length, isStreaming]);

  // A streaming answer grows without changing the message COUNT, so the effect
  // above never fires for it and the text runs off the bottom of the screen.
  // Unanimated on purpose: this lands every ~50ms, and animating each one
  // fights the next.
  const streamingLength = isReceivingText
    ? messages[messages.length - 1]?.content?.length ?? 0
    : 0;
  useEffect(() => {
    if (!streamingLength) return;
    flatListRef.current?.scrollToEnd({ animated: false });
  }, [streamingLength]);

  // --- capture entry points -------------------------------------------------

  /**
   * Capture a photo, then hand a receipt straight to the review form.
   *
   * A receipt is not finished when the photo is read — it still has to be
   * checked and confirmed before it becomes a transaction. The Files tab has
   * always opened that form itself; chat used to stop at a card and wait to be
   * tapped, so the identical receipt ended up either reviewed or forgotten
   * depending on which screen it was uploaded from.
   *
   * Price tags and unclassified photos stay put: their card *is* the review
   * step, and it lives here.
   */
  const captureAndRoute = useCallback(
    async (
      photo: { uri: string; name: string; type: string },
      place?: string | null
    ) => {
      // A resolved place name, never coordinates. Only a live capture has one:
      // a photo picked from the library was taken somewhere else, and attaching
      // where the user is NOW would be worse than attaching nothing.
      const result = await sendPhoto(photo, place || undefined);
      if (result?.kind === "receipt") {
        openReview(router, [{ file_id: result.file_id, kind: result.kind }]);
      }
    },
    [sendPhoto, router]
  );

  const openCamera = useCallback(async () => {
    if (!cameraPermission?.granted) {
      const result = await requestCameraPermission();
      if (!result.granted) {
        log.warn("Camera permission denied");
        return;
      }
    }
    setCameraOpen(true);
  }, [cameraPermission?.granted, requestCameraPermission]);

  const pickImage = useCallback(async () => {
    const result = await DocumentPicker.getDocumentAsync({
      type: ["image/png", "image/jpeg"],
      multiple: false,
    });
    if (result.canceled || !result.assets?.length) return;
    const asset = result.assets[0];
    await captureAndRoute({
      uri: asset.uri,
      name: asset.name,
      type: asset.mimeType || "image/jpeg",
    });
  }, [captureAndRoute]);

  const pickDocument = useCallback(async () => {
    // CSVs still go through the batch pipeline — it handles dedup, enrichment
    // and vector indexing that a single-photo capture has no need for.
    const result = await DocumentPicker.getDocumentAsync({
      type: ["text/csv", "text/comma-separated-values", "application/vnd.ms-excel"],
      multiple: true,
    });
    if (result.canceled || !result.assets?.length) return;
    const picked = result.assets.map((a) => ({
      uri: a.uri,
      name: a.name,
      type: a.mimeType || "text/csv",
    }));
    const ok = await uploadFiles(picked);
    setSnack(
      ok
        ? "Statement uploaded — processing in the background."
        : "That upload didn't work. Try the Files tab?"
    );
  }, [uploadFiles]);

  const handleAttach = useCallback(
    (action: ChatAttachAction) => {
      if (action === "camera") return void openCamera();
      if (action === "library") return void pickImage();
      return void pickDocument();
    },
    [openCamera, pickImage, pickDocument]
  );

  // A shelf photo is not stored until it is confirmed, so the id the card holds
  // is a CAPTURE handle and the id the photo ends up with is a different one.
  // Remembering the mapping is what lets the question carry the real id — the
  // capture id was being stored with the message, matched nothing on reload, and
  // the picture silently did not come back.
  const storedFileIds = useRef<Record<string, string>>({});

  const handleConfirmPriceTag = useCallback(
    async (fileId: string, draft: PriceTagDraft & { tag_index: number }) => {
      const saved = await savePriceObservation(fileId, draft);
      if (saved.bill_file_id) storedFileIds.current[fileId] = saved.bill_file_id;
      return saved.comparison ?? null;
    },
    []
  );

  /**
   * Ask the agent about a whole photo's worth of confirmed prices, in ONE turn.
   *
   * This used to fire per tag. Two tags meant two chat turns started back to
   * back — and because no re-render happens between them, the second one's
   * `isStreaming` guard still read false, so both ran. They raced on the same
   * conversation and the second answer inherited the first's context, which is
   * why confirming potatoes and yams produced two replies both leading with
   * potatoes.
   */
  const handleAskAboutPrices = useCallback(
    (drafts: (PriceTagDraft & { tag_index: number })[], fileId?: string) => {
      const describe = (d: PriceTagDraft) => {
        const size = d.size_value
          ? ` (${d.size_value}${d.size_unit ? " " + d.size_unit : ""})`
          : "";
        const store = d.merchant_name ? ` at ${d.merchant_name}` : "";
        const tag = d.item_qualitative_description
          ? ` [tag says: "${d.item_qualitative_description}"]`
          : "";
        return `${d.item_description}${size} — $${d.item_subtotal_price}${store}${tag}`;
      };
      const many = drafts.length > 1;
      // The photo id travels with the question, so reopening this conversation
      // shows the picture again. Only a CONFIRMED capture has one — an
      // unconfirmed shelf photo is never stored, so there is nothing to replay.
      void sendMessage(
        `${drafts.map(describe).join("; ")}. Already recorded, don't save again. ` +
          (many ? "Good prices? Answer each one separately and briefly." : "Good price?"),
        // The stored id, not the capture handle it started as.
        fileId && storedFileIds.current[fileId]
          ? [storedFileIds.current[fileId]]
          : undefined
      );
    },
    [sendMessage]
  );

  /**
   * Throw a captured photo away for real.
   *
   * Discard was local UI state only: the card said "Discarded" while the photo
   * and its BillFile row stayed exactly where they were, so a tag the user had
   * explicitly dismissed still turned up in the Files tab. Deleting the file
   * takes its observations and vectors with it via ON DELETE CASCADE.
   */
  const handleDiscardCapture = useCallback(
    async (fileId: string) => {
      // /captures, not /files: an unconfirmed capture has no BillFile row to
      // delete, because a shelf photo is not stored until you confirm it.
      await discardCapture(fileId);
      // Only matters once the photo was confirmed and does have a row.
      await loadFiles();
    },
    [loadFiles]
  );

  // Answering "which is this?" with "receipt" reaches the review form too —
  // otherwise resolving a photo by hand would be the one way to end up with an
  // un-reviewed receipt.
  const handleKindResolved = useCallback(
    (fileId: string, result: CaptureResult) => {
      // Matched on the OLD id, because that is what the message on screen holds.
      updateCapture(fileId, result);
      if (result.kind === "receipt") {
        // Opened with the NEW one. Answering "receipt" stores the photo, which
        // mints a BillFile id and returns it — `fileId` here is still the
        // capture handle, and it names no row, so routing with it landed on
        // "Receipt review is not available".
        openReview(router, [{ file_id: result.file_id, kind: result.kind }]);
      }
    },
    [updateCapture, router]
  );

  const handleRerun = useCallback((text: string) => void sendMessage(text), [sendMessage]);

  const renderItem = useCallback(
    ({ item }: { item: ChatMessageType }) => (
      <MemoizedChatMessage
        message={item}
        onRerun={handleRerun}
        rerunDisabled={isStreaming}
        onKindResolved={handleKindResolved}
        onConfirmPriceTag={handleConfirmPriceTag}
        onDiscardCapture={handleDiscardCapture}
        onAskAboutPrices={handleAskAboutPrices}
        // Still reachable: the card stays in the thread after review, so this
        // is how the user gets back to a receipt they left or want to edit.
        onReviewReceipt={(fileId) =>
          openReview(router, [{ file_id: fileId, kind: "receipt" }])
        }
      />
    ),
    [handleKindResolved, handleConfirmPriceTag, handleDiscardCapture, handleAskAboutPrices, router, handleRerun, isStreaming]
  );

  const keyExtractor = useCallback((_: ChatMessageType, i: number) => String(i), []);

  const wrapperProps = Platform.OS === "ios"
    ? { style: styles.container, behavior: "padding" as const, keyboardVerticalOffset: 90 }
    : { style: styles.container, behavior: "padding" as const, keyboardVerticalOffset: 80 };

  const responsiveStyle = isWide
    ? { maxWidth: MAX_CHAT_WIDTH, width: "100%" as const, alignSelf: "center" as const }
    : undefined;

  return (
    <KeyboardAvoidingView {...wrapperProps}>
      <ConversationDrawer
        visible={drawerOpen}
        conversations={conversations}
        activeId={conversationId}
        isLoading={convLoading}
        onClose={() => setDrawerOpen(false)}
        onSelect={(id) => loadConversation(id)}
        onNew={() => newConversation()}
        onDelete={async (id) => {
          await removeConversation(id);
          if (id === conversationId) newConversation();
        }}
      />

      {/* File status banner. Waits for the first load to finish: `files` is
          empty before the response arrives too, so checking only the count
          flashed "no data loaded yet" on every launch and then withdrew it. */}
      {filesLoaded && fileCount === 0 && (
        <Banner
          visible
          style={styles.warningBanner}
          icon="alert-circle-outline"
          actions={[
            {
              label: "Upload Files",
              onPress: () => router.push("/(tabs)/ingest"),
            },
          ]}
        >
          No data loaded yet. Upload a CSV or receipt to start chatting.
        </Banner>
      )}

      {/* Messages */}
      <FlatList
        ref={flatListRef}
        data={messages}
        keyExtractor={keyExtractor}
        renderItem={renderItem}
        contentContainerStyle={[styles.messagesList, responsiveStyle]}
        keyboardShouldPersistTaps="handled"
        // Keep chart WebViews alive when scrolled off-screen to avoid re-loading CDN
        windowSize={7}
        maxToRenderPerBatch={5}
        removeClippedSubviews={false}
        ListEmptyComponent={
          <SuggestedPrompts
            onSelectPrompt={sendMessage}
            onAction={handleAttach}
          />
        }
      />

      {/* Streaming indicator with live tool status. Hidden once the answer
          itself starts arriving — "Thinking..." underneath text that is
          visibly being written is just noise. */}
      {isStreaming && !isReceivingText && (
        <View style={[styles.streamingIndicator, responsiveStyle]}>
          {currentToolTraces.length > 0 ? (
            <View style={styles.toolStatus}>
              <Text style={styles.toolStatusText}>
                {(() => {
                  const lastStart = [...currentToolTraces].reverse().find(t => t.type === "tool_start");
                  const lastEnd = [...currentToolTraces].reverse().find(t => t.type === "tool_end");
                  const lastStartIdx = lastStart ? currentToolTraces.lastIndexOf(lastStart) : -1;
                  const lastEndIdx = lastEnd ? currentToolTraces.lastIndexOf(lastEnd) : -1;
                  if (lastStartIdx > lastEndIdx && lastStart) {
                    const name = lastStart.name.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
                    return `🔍 ${name}...`;
                  }
                  return "💭 Thinking...";
                })()}
              </Text>
            </View>
          ) : (
            <TypingIndicator />
          )}
        </View>
      )}

      {/* Chat input */}
      <View style={isWide ? styles.inputWrapper : undefined}>
        <View style={responsiveStyle}>
          <ChatInput onSend={sendMessage} onAttach={handleAttach} disabled={isStreaming} />
        </View>
      </View>

      <CameraCapture
        visible={cameraOpen}
        onCapture={(photo, place) => {
          setCameraOpen(false);
          void captureAndRoute(photo, place);
        }}
        onClose={() => setCameraOpen(false)}
      />

      <Snackbar visible={!!snack} onDismiss={() => setSnack(null)} duration={4000}>
        {snack || ""}
      </Snackbar>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  warningBanner: {
    backgroundColor: "#FEF3C7",
  },
  messagesList: {
    paddingVertical: spacing.md,
    flexGrow: 1,
  },
  streamingIndicator: {
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
  },
  inputWrapper: {
    alignItems: "center" as const,
  },
  toolStatus: {
    flexDirection: "row" as const,
    alignItems: "center" as const,
    paddingVertical: 4,
  },
  toolStatusText: {
    fontSize: 13,
    color: colors.primary,
    fontWeight: "500" as const,
  },
});
