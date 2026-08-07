import type { Router } from "expo-router";
import type { ReviewItem } from "../services/fileService";

/**
 * Where a captured photo goes once we know what it is.
 *
 * Both entry points land here — the Files tab after the batch worker finishes,
 * and the Chat tab after an inline capture — so a receipt behaves identically
 * whichever way it arrived. Keeping the decision in one place is the point:
 * when the two screens each owned a copy, only one of them opened the review
 * form, so the same photo either got reviewed or sat waiting on a tap depending
 * on where you happened to upload it.
 */
export function openReview(router: Router, items: ReviewItem[]): void {
  if (!items.length) return;
  const [next, ...rest] = items;

  // Only receipts are reviewable from the receipt form. A price tag is
  // confirmed on its own card, and an unclassified photo needs the "which is
  // this?" prompt — both of those live in Chat, so send them there rather than
  // opening a receipt form on something that is not a receipt.
  if (next.kind !== "receipt") {
    // navigate, not push: this target is a TAB. router.push onto a sibling tab
    // stacks a second copy of the screen instead of switching to it, so the
    // photo's card was rendered somewhere the user never saw and the Files tab
    // simply appeared to do nothing.
    router.navigate({ pathname: "/(tabs)/chat", params: { captureFileId: next.file_id } });
    return;
  }

  router.push({
    pathname: "/receipt-review/[fileId]",
    params: {
      fileId: next.file_id,
      // The review screen pops these one at a time, so non-receipts are
      // dropped here rather than queued into a form that cannot show them.
      remaining: rest
        .filter((item) => item.kind === "receipt")
        .map((item) => item.file_id)
        .join(","),
    },
  });
}
