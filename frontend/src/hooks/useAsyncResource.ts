/**
 * Load something from the API, and say what happened while it loads.
 *
 * Every list hook wrote this out: `[data, isLoading, error]`, a `refresh` with
 * try/catch/finally, and a log line. They had already drifted — useTransactions
 * surfaced the error to the UI, useConversations only logged it, so a failed
 * conversation load left the drawer silently empty with no explanation and no
 * way to retry.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { createLogger } from "../lib/logger";

export interface AsyncResource<T> {
  data: T;
  /** For optimistic updates — removing a row without a full refetch. */
  setData: React.Dispatch<React.SetStateAction<T>>;
  isLoading: boolean;
  /** Null when the last load succeeded. Always surfaced, never only logged. */
  error: string | null;
  /**
   * True once a load has finished, successfully or not.
   *
   * The thing empty states actually need. "No data" and "no data YET" look
   * identical if you only check `data.length === 0`, so every list flashed its
   * empty message on launch and then replaced it with rows a moment later.
   * Gate on this instead: an empty list is only empty once we have looked.
   */
  hasLoaded: boolean;
  refresh: () => Promise<void>;
}

export function useAsyncResource<T>(
  fetcher: () => Promise<T>,
  initial: T,
  options: {
    /** Names the hook in logs, e.g. "transactions". */
    label: string;
    /** Load on mount. False when a screen refreshes on focus instead. */
    immediate?: boolean;
  }
): AsyncResource<T> {
  const { label, immediate = true } = options;
  const [data, setData] = useState<T>(initial);
  // Starts true when a load is already on its way, so the very first frame
  // reads as "loading" rather than as "loaded and empty". Without this there
  // is always at least one render showing an empty list that is not empty.
  const [isLoading, setIsLoading] = useState(immediate);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Held in a ref so `refresh` is stable even when the caller passes a fresh
  // arrow function each render — otherwise the effect below re-runs forever.
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  // A slow response that arrives after the component is gone must not call
  // setState, and two overlapping refreshes must not let the older one win.
  const requestId = useRef(0);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refresh = useCallback(async () => {
    const id = ++requestId.current;
    setIsLoading(true);
    setError(null);
    try {
      const result = await fetcherRef.current();
      if (!mounted.current || id !== requestId.current) return;
      setData(result);
    } catch (e: any) {
      if (!mounted.current || id !== requestId.current) return;
      createLogger("useAsyncResource").error(`Failed to load ${label}`, e);
      setError(e?.message || `Could not load ${label}`);
    } finally {
      if (mounted.current && id === requestId.current) {
        setIsLoading(false);
        // Set even on failure: we looked, and this is what we know. A list that
        // could not load should say so, not sit on a spinner forever.
        setHasLoaded(true);
      }
    }
  }, [label]);

  useEffect(() => {
    if (immediate) refresh();
  }, [immediate, refresh]);

  return { data, setData, isLoading, hasLoaded, error, refresh };
}
