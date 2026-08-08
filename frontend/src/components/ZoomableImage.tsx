import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Animated,
  GestureResponderEvent,
  Image,
  ImageResizeMode,
  LayoutChangeEvent,
  PanResponder,
  StyleProp,
  StyleSheet,
  View,
  ViewStyle,
} from "react-native";

interface Props {
  uri: string;
  style?: StyleProp<ViewStyle>;
  resizeMode?: ImageResizeMode;
  /** How far in a pinch or a double-tap may go. */
  maxScale?: number;
  /**
   * Quarter turns clockwise, in degrees (0 / 90 / 180 / 270).
   *
   * Display only — the stored photo is never rewritten. A receipt photographed
   * sideways needs turning to be read, not correcting: re-encoding and
   * re-uploading it would replace the original evidence to fix a viewing
   * problem, and the OCR has already run against the file as it is.
   */
  rotation?: number;
}

const DOUBLE_TAP_MS = 280;
const DOUBLE_TAP_SCALE = 2.5;
/** A touch that travels further than this was a drag, not a tap. */
const TAP_SLOP = 12;
/** Pinching out past natural size has this much give before it springs back. */
const MIN_PINCH_SCALE = 0.8;

type Point = { x: number; y: number };

type Touches = GestureResponderEvent["nativeEvent"]["touches"];

/** Midpoint of the active touches, in page coordinates. */
function centroid(event: GestureResponderEvent): Point {
  const touches = event.nativeEvent.touches;
  if (!touches || touches.length === 0) {
    // react-native-web synthesises mouse events with no touch list.
    return { x: event.nativeEvent.pageX, y: event.nativeEvent.pageY };
  }
  let x = 0;
  let y = 0;
  for (const touch of touches) {
    x += touch.pageX;
    y += touch.pageY;
  }
  return { x: x / touches.length, y: y / touches.length };
}

function spread(touches: Touches): number {
  const [a, b] = touches;
  // Floored: two fingers landing on the same pixel would otherwise divide the
  // scale by zero.
  return Math.max(1, Math.hypot(a.pageX - b.pageX, a.pageY - b.pageY));
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * An image you can pinch, drag and double-tap.
 *
 * Receipts are the reason this exists: the whole point of keeping the photo is
 * being able to check a line the OCR got wrong, and a receipt scaled to fit a
 * phone screen is unreadable exactly where it matters — the small print.
 *
 * Built on Animated and PanResponder rather than a gesture library. Neither
 * react-native-gesture-handler nor reanimated is in this project, and adding
 * them for one interaction would mean a native rebuild; these are in React
 * Native itself and work on web as well, where a double-click zooms since a
 * trackpad pinch never reaches the app.
 *
 * The transform is `translate` then `scale`, and that order is load-bearing:
 * scaling first would put the translation into the scaled coordinate space, so
 * a drag would move the image by its distance times the zoom factor.
 */
export function ZoomableImage({
  uri,
  style,
  resizeMode = "contain",
  maxScale = 4,
  rotation = 0,
}: Props) {
  // A quarter turn swaps which way the image is wide, and that changes both how
  // `contain` fits it and how far it may be panned. Normalised first so a
  // negative or accumulated angle (-90, 450) is classified the same as 270/90.
  const turn = (((rotation % 360) + 360) % 360);
  const quarterTurned = turn % 180 !== 0;
  const scale = useRef(new Animated.Value(1)).current;
  /**
   * Rotation is an Animated.Value rather than a plain "90deg" string, and it
   * has to be.
   *
   * The three values above are animated with `useNativeDriver: true`, which
   * hands the whole `transform` prop to the native side. A static entry sitting
   * in the same array is captured when that native node is built and never read
   * again — re-rendering with a new angle updated the JS style and changed
   * nothing on screen. Making the angle a fourth animated value puts it in the
   * same node, so it travels by the same route as the others.
   */
  const spin = useRef(new Animated.Value(0)).current;
  const translateX = useRef(new Animated.Value(0)).current;
  const translateY = useRef(new Animated.Value(0)).current;

  const [container, setContainer] = useState<{ width: number; height: number } | null>(null);
  const [aspect, setAspect] = useState<number | null>(null);

  // What is on screen right now. Read back synchronously by every gesture, so
  // it is kept here rather than pulled out of the Animated.Values — those are
  // native-driven during the settle animations and their JS-side value goes
  // stale the moment a spring starts.
  const live = useRef({ scale: 1, x: 0, y: 0 });

  // Where the gesture in progress started. Re-taken whenever the number of
  // fingers changes, so that lifting one finger out of a pinch continues as a
  // pan from where the pinch left off instead of jumping.
  const start = useRef<{
    touches: number;
    scale: number;
    x: number;
    y: number;
    focal: Point;
    distance: number;
  } | null>(null);

  const tap = useRef<{ origin: Point; moved: boolean } | null>(null);
  const lastTapAt = useRef(0);

  useEffect(() => {
    let alive = true;
    setAspect(null);
    Image.getSize(
      uri,
      (width, height) => {
        if (alive && height > 0) setAspect(width / height);
      },
      () => {
        /* Fall back to the container's own bounds. */
      }
    );
    return () => {
      alive = false;
    };
  }, [uri]);

  /**
   * The image's footprint on screen at scale 1, after any rotation.
   *
   * When quarter-turned the image is laid out in a box with the container's
   * dimensions swapped (see the render below), so `contain` fits it to the
   * orientation it will be seen in. What ends up on screen is that fitted box
   * turned on its side, hence the final swap.
   */
  const drawnSize = (): { width: number; height: number } => {
    if (!container) return { width: 0, height: 0 };
    const boxWidth = quarterTurned ? container.height : container.width;
    const boxHeight = quarterTurned ? container.width : container.height;
    let width = boxWidth;
    let height = boxHeight;
    if (aspect && resizeMode === "contain") {
      const fitted = Math.min(boxWidth, boxHeight * aspect);
      width = fitted;
      height = fitted / aspect;
    }
    return quarterTurned ? { width: height, height: width } : { width, height };
  };

  /**
   * How far the image may be dragged on each axis: half of whatever overflows
   * the container once scaled. Uses the size the image is actually drawn at,
   * not the container's, or panning would wander into the letterboxing that
   * `contain` leaves around a tall receipt.
   */
  const boundsFor = (at: number): Point => {
    if (!container) return { x: 0, y: 0 };
    const drawn = drawnSize();
    return {
      x: Math.max(0, (drawn.width * at - container.width) / 2),
      y: Math.max(0, (drawn.height * at - container.height) / 2),
    };
  };

  const apply = (next: { scale: number; x: number; y: number }) => {
    const bounds = boundsFor(next.scale);
    live.current = {
      scale: next.scale,
      x: clamp(next.x, -bounds.x, bounds.x),
      y: clamp(next.y, -bounds.y, bounds.y),
    };
    scale.setValue(live.current.scale);
    translateX.setValue(live.current.x);
    translateY.setValue(live.current.y);
  };

  // A quarter turn swaps the axes, so wherever the image was panned to no
  // longer means anything — a receipt zoomed into its bottom-left corner would
  // turn to show empty space. Reset to centred and unzoomed. Skipped on the
  // first render so mounting at a non-zero rotation is not treated as a turn.
  const settledRotation = useRef(turn);
  useEffect(() => {
    spin.setValue(turn);
    if (settledRotation.current === turn) return;
    settledRotation.current = turn;
    apply({ scale: 1, x: 0, y: 0 });
  }, [turn, spin]);

  const settle = (to: { scale: number; x: number; y: number }) => {
    const bounds = boundsFor(to.scale);
    live.current = {
      scale: to.scale,
      x: clamp(to.x, -bounds.x, bounds.x),
      y: clamp(to.y, -bounds.y, bounds.y),
    };
    Animated.parallel([
      Animated.spring(scale, { toValue: live.current.scale, useNativeDriver: true, friction: 7 }),
      Animated.spring(translateX, { toValue: live.current.x, useNativeDriver: true, friction: 7 }),
      Animated.spring(translateY, { toValue: live.current.y, useNativeDriver: true, friction: 7 }),
    ]).start();
  };

  // The container's centre in page coordinates. Touches arrive in page space
  // while the translation is measured from the centre, so anchoring a pinch
  // needs the offset between the two.
  const view = useRef<View>(null);
  const centre = useRef<Point>({ x: 0, y: 0 });

  const measure = () =>
    view.current?.measureInWindow((x, y, width, height) => {
      centre.current = { x: x + width / 2, y: y + height / 2 };
    });

  const onLayout = (event: LayoutChangeEvent) => {
    const { width, height } = event.nativeEvent.layout;
    setContainer({ width, height });
    measure();
  };

  /**
   * Keep the content under `focal` pinned there while the scale changes from
   * `from.scale` to `to`. Without this a pinch grows the image about the
   * container's centre, pushing whatever the fingers were over off-screen —
   * which on a receipt means the line you zoomed in to read.
   *
   * With `d = t + s·p` for a content point `p` measured from the centre,
   * holding `p` still across the change gives `t₁ = F₁ − (s₁/s₀)·(F₀ − t₀)`.
   */
  const anchor = (
    from: { scale: number; x: number; y: number },
    to: number,
    focalStart: Point,
    focalNow: Point
  ) => {
    const startX = focalStart.x - centre.current.x;
    const startY = focalStart.y - centre.current.y;
    const nowX = focalNow.x - centre.current.x;
    const nowY = focalNow.y - centre.current.y;
    return {
      scale: to,
      x: nowX - (to / from.scale) * (startX - from.x),
      y: nowY - (to / from.scale) * (startY - from.y),
    };
  };

  const responder = useMemo(
    () =>
      PanResponder.create({
        onStartShouldSetPanResponder: () => true,
        onMoveShouldSetPanResponder: (event, gesture) =>
          event.nativeEvent.touches.length >= 2 ||
          live.current.scale > 1 ||
          Math.abs(gesture.dx) > 2 ||
          Math.abs(gesture.dy) > 2,
        // A pinch inside a Modal or a scrollable parent must not be handed back
        // half way through.
        onPanResponderTerminationRequest: () => false,

        onPanResponderGrant: (event) => {
          start.current = null;
          // Re-measure: the viewer sits in a Modal that animates in, so the
          // position at layout time is not where it ends up.
          measure();
          const origin = centroid(event);
          tap.current = { origin, moved: false };
        },

        onPanResponderMove: (event, gesture) => {
          const touches = event.nativeEvent.touches;
          // Web synthesises mouse events with an empty touch list; treat that
          // as the one pointer it is.
          const count = Math.max(1, touches?.length ?? 1);
          const focal = centroid(event);

          if (tap.current && Math.hypot(gesture.dx, gesture.dy) > TAP_SLOP) {
            tap.current.moved = true;
          }

          // First move, or a finger went down or came up: rebase on where the
          // image sits now.
          if (!start.current || start.current.touches !== count) {
            start.current = {
              touches: count,
              scale: live.current.scale,
              x: live.current.x,
              y: live.current.y,
              focal,
              distance: count >= 2 ? spread(touches) : 0,
            };
            return;
          }

          const from = start.current;

          if (count >= 2) {
            const next = clamp(
              (spread(touches) / from.distance) * from.scale,
              MIN_PINCH_SCALE,
              maxScale
            );
            // The focal point is re-read every frame, so two fingers moving
            // together pan as well as pinch.
            apply(anchor(from, next, from.focal, focal));
            return;
          }

          if (live.current.scale > 1) {
            apply({
              scale: live.current.scale,
              x: from.x + (focal.x - from.focal.x),
              y: from.y + (focal.y - from.focal.y),
            });
          }
        },

        onPanResponderRelease: (event) => {
          start.current = null;

          const tapped = tap.current;
          tap.current = null;
          const now = Date.now();

          // Only a genuine tap — one finger, barely moved — can be half of a
          // double tap. Stamping every touch down, drag included, made a drag
          // followed by a tap zoom unexpectedly.
          if (tapped && !tapped.moved && (event.nativeEvent.touches?.length ?? 0) === 0) {
            if (now - lastTapAt.current < DOUBLE_TAP_MS) {
              lastTapAt.current = 0;
              const zoomed = live.current.scale > 1.05;
              if (zoomed) {
                settle({ scale: 1, x: 0, y: 0 });
              } else {
                const focal = tapped.origin;
                settle(anchor(live.current, DOUBLE_TAP_SCALE, focal, focal));
              }
              return;
            }
            lastTapAt.current = now;
          }

          if (live.current.scale <= 1.05) {
            // Anything at or below natural size returns to centre, so the image
            // can never be left parked off-screen.
            settle({ scale: 1, x: 0, y: 0 });
            return;
          }
          // Bring any overhang back inside the bounds with a spring.
          settle(live.current);
        },

        onPanResponderTerminate: () => {
          start.current = null;
          tap.current = null;
          settle(live.current.scale <= 1.05 ? { scale: 1, x: 0, y: 0 } : live.current);
        },
      }),
    [maxScale, scale, translateX, translateY, container, aspect, resizeMode, quarterTurned]
  );

  // Turned on its side, the image is laid out in a box with the container's
  // dimensions swapped and re-centred, so that `contain` fits it to the shape
  // it will actually be seen in. Without this a portrait receipt rotated 90°
  // is fitted to the portrait container first and then spun, leaving most of
  // it clipped off both sides.
  const box =
    quarterTurned && container
      ? {
          position: "absolute" as const,
          width: container.height,
          height: container.width,
          left: (container.width - container.height) / 2,
          top: (container.height - container.width) / 2,
        }
      : StyleSheet.absoluteFill;

  return (
    <View ref={view} style={[styles.container, style]} onLayout={onLayout} {...responder.panHandlers}>
      <Animated.Image
        source={{ uri }}
        resizeMode={resizeMode}
        style={[
          box,
          {
            // Rotate is LAST, so it happens innermost: the image turns about
            // its own centre and the pan and zoom then act on the turned image
            // as a unit. Placed first, a drag would move along the original
            // axes and feel sideways once rotated.
            transform: [
              { translateX },
              { translateY },
              { scale },
              {
                rotate: spin.interpolate({
                  inputRange: [0, 360],
                  outputRange: ["0deg", "360deg"],
                }),
              },
            ],
          },
        ]}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: { overflow: "hidden" },
});
