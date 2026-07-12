import React from "react";
import { Animated, StyleSheet, View, Platform, useWindowDimensions } from "react-native";
import { Text } from "react-native-paper";
import { colors } from "../styles/theme";

interface PlotlyChartProps {
  chartJson: string;
}

export function PlotlyChart({ chartJson }: PlotlyChartProps) {
  if (Platform.OS === "web") {
    return <PlotlyChartWeb chartJson={chartJson} />;
  }

  return <PlotlyChartNative chartJson={chartJson} />;
}

// ---- Loading skeleton shown while Plotly CDN loads ----

function ChartSkeleton({ width, height }: { width: number; height: number }) {
  const pulseAnim = React.useRef(new Animated.Value(0.3)).current;

  React.useEffect(() => {
    const pulse = Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 0.7, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 0.3, duration: 800, useNativeDriver: true }),
      ])
    );
    pulse.start();
    return () => pulse.stop();
  }, [pulseAnim]);

  return (
    <View style={[styles.skeletonContainer, { width, height }]}>
      {/* Title bar */}
      <Animated.View style={[styles.skeletonBlock, { opacity: pulseAnim, width: "55%", height: 12, marginBottom: 16 }]} />
      {/* Chart area */}
      <Animated.View style={[styles.skeletonBlock, { opacity: pulseAnim, width: "100%", height: height * 0.55, borderRadius: 6 }]} />
      {/* Legend dots */}
      <View style={{ flexDirection: "row", justifyContent: "center", marginTop: 12, gap: 16 }}>
        <Animated.View style={[styles.skeletonBlock, { opacity: pulseAnim, width: 50, height: 8 }]} />
        <Animated.View style={[styles.skeletonBlock, { opacity: pulseAnim, width: 50, height: 8 }]} />
        <Animated.View style={[styles.skeletonBlock, { opacity: pulseAnim, width: 50, height: 8 }]} />
      </View>
    </View>
  );
}

// ---- Native implementation using WebView + Plotly.js ----

function PlotlyChartNative({ chartJson }: { chartJson: string }) {
  const { WebView } = require("react-native-webview");
  const { width: screenWidth } = useWindowDimensions();
  const [webViewHeight, setWebViewHeight] = React.useState(0);
  const [isLoading, setIsLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Parse chart metadata once for layout decisions
  const chartMeta = React.useMemo(() => {
    try {
      const parsed = JSON.parse(chartJson);
      const firstTrace = parsed.data?.[0];
      const type = firstTrace?.type || "bar";
      return { type };
    } catch {
      return { type: "bar" as const };
    }
  }, [chartJson]);

  const isPie = chartMeta.type === "pie";

  // Measure the ACTUAL available width (the chat bubble) via onLayout, so the
  // chart fills its container instead of overflowing a screenWidth-based guess.
  const [boxWidth, setBoxWidth] = React.useState(0);
  const measured = boxWidth > 0;
  // chartContainer has 4px padding each side
  const chartWidth = measured
    ? Math.max(Math.floor(boxWidth) - 8, 220)
    : Math.floor(screenWidth * 0.9) - 8;
  // Taller, more readable aspect ratios on mobile
  const chartHeight = isPie
    ? Math.round(chartWidth * 0.85)
    : Math.round(chartWidth * 0.68);
  const fontSize = Math.max(11, Math.min(15, Math.round(chartWidth / 24)));

  // Base64-encode the chart JSON so it survives template literal injection safely.
  const chartJsonB64 = React.useMemo(() => {
    try {
      const parsed = JSON.parse(chartJson);
      const clean = JSON.stringify(parsed);
      return btoa(unescape(encodeURIComponent(clean)));
    } catch {
      return "";
    }
  }, [chartJson]);

  const html = React.useMemo(() => {
    if (!chartJsonB64) return "";

    return `
<!DOCTYPE html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { background: transparent; overflow: hidden; font-family: -apple-system, sans-serif; -webkit-tap-highlight-color: transparent; }
    #chart { width: ${chartWidth}px; }
    #status { color: #888; text-align: center; padding: 20px; font-size: 13px; display: none; }
    /* Hide modebar on mobile */
    .modebar { display: none !important; }
    /* Prevent text selection on chart */
    .plot-container { -webkit-user-select: none; user-select: none; }
    /* Improve touch targets */
    .point, .slice { cursor: pointer; }
  </style>
</head>
<body>
  <div id="status"></div>
  <div id="chart"></div>
  <script>
    var CHART_B64 = "${chartJsonB64}";

    function msg(obj) { window.ReactNativeWebView.postMessage(JSON.stringify(obj)); }

    function reportHeight() {
      setTimeout(function() {
        var el = document.getElementById('chart');
        var h = el ? el.offsetHeight : 0;
        if (h > 0) msg({ type: 'height', height: h });
      }, 150);
    }

    function truncateLabel(text, maxLen) {
      if (!text || typeof text !== 'string') return text;
      return text.length > maxLen ? text.substring(0, maxLen - 1) + '\\u2026' : text;
    }

    function renderChart() {
      try {
        var raw = decodeURIComponent(escape(atob(CHART_B64)));
        var chartData = JSON.parse(raw);
        if (!chartData || !chartData.data || !Array.isArray(chartData.data)) {
          throw new Error('Invalid chart data structure');
        }
        if (!chartData.layout) chartData.layout = {};
        var traceType = chartData.data[0] && chartData.data[0].type;
        var isPie = traceType === 'pie';
        var isLine = traceType === 'scatter' || traceType === 'line';

        // Bar/line/scatter trace optimizations
        if (!isPie && chartData.data) {
          chartData.data.forEach(function(trace) {
            if (trace.x && Array.isArray(trace.x)) {
              trace.x = trace.x.map(function(v) { return truncateLabel(String(v), 16); });
            }
            trace.hoverinfo = trace.hoverinfo || 'x+y+text';

            // Line charts: thicker lines + visible markers for touch
            if (isLine) {
              trace.line = Object.assign({}, trace.line || {}, { width: 3 });
              trace.marker = Object.assign({}, trace.marker || {}, { size: 8 });
              trace.mode = trace.mode || 'lines+markers';
            }

            // Bar charts: subtle separator between bars
            if (traceType === 'bar') {
              trace.marker = Object.assign({}, trace.marker || {}, {
                line: { width: 0.5, color: 'rgba(255,255,255,0.3)' }
              });
            }
          });
        }

        // Pie chart mobile optimizations
        if (isPie && chartData.data) {
          chartData.data.forEach(function(trace) {
            trace.textposition = 'outside';
            trace.textinfo = 'label+percent';
            trace.textfont = { size: ${fontSize} };
            trace.outsidetextfont = { size: ${Math.max(8, fontSize - 1)} };
            trace.insidetextorientation = 'horizontal';
            if (trace.labels && Array.isArray(trace.labels)) {
              trace.labels = trace.labels.map(function(v) { return truncateLabel(String(v), 14); });
            }
            trace.hoverinfo = 'label+value+percent';
            trace.pull = 0.02;
            trace.hole = 0.3;
          });
        }

        var origXaxis = (chartData.layout && chartData.layout.xaxis) || {};
        var origYaxis = (chartData.layout && chartData.layout.yaxis) || {};
        var xaxis = Object.assign({}, origXaxis, {
          tickangle: -45,
          automargin: true,
          fixedrange: true,
          tickfont: { size: ${Math.max(8, fontSize - 1)} }
        });
        var yaxis = Object.assign({}, origYaxis, {
          automargin: true,
          fixedrange: true,
          tickfont: { size: ${Math.max(8, fontSize - 1)} },
          gridcolor: 'rgba(0,0,0,0.06)',
          zerolinecolor: 'rgba(0,0,0,0.1)'
        });

        var baseLayout = {
          width: ${chartWidth},
          height: ${chartHeight},
          autosize: false,
          margin: isPie
            ? { l: 10, r: 10, t: 36, b: 10, pad: 0 }
            : { l: 4, r: 4, t: 36, b: 60, pad: 0, autoexpand: true },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { size: ${fontSize}, color: '#374151' },
          title: chartData.layout.title ? {
            text: truncateLabel(
              typeof chartData.layout.title === 'string'
                ? chartData.layout.title
                : ((chartData.layout.title && chartData.layout.title.text) || ''),
              40
            ),
            font: { size: ${fontSize + 2}, color: '#1e1e3a' },
            x: 0.5,
            xanchor: 'center',
            y: 0.98,
            yanchor: 'top'
          } : undefined,
          legend: {
            orientation: 'h',
            yanchor: 'top',
            y: -0.15,
            xanchor: 'center',
            x: 0.5,
            font: { size: ${Math.max(8, fontSize - 1)} },
            tracegroupgap: 4
          },
          dragmode: false,
          hoverlabel: {
            bgcolor: '#1e1e3a',
            bordercolor: '#6366f1',
            font: { size: ${fontSize + 1}, color: '#fff', family: '-apple-system, sans-serif' },
            namelength: -1
          },
          bargap: 0.2,
          bargroupgap: 0.1
        };

        // Merge with original layout, then add axis config
        // For pie charts, explicitly delete xaxis/yaxis to avoid Plotly reading .anchor on undefined
        var layout = Object.assign({}, chartData.layout || {}, baseLayout);
        if (isPie) {
          delete layout.xaxis;
          delete layout.yaxis;
        } else {
          layout.xaxis = xaxis;
          layout.yaxis = yaxis;
        }

        Plotly.newPlot('chart', chartData.data, layout, {
          responsive: false,
          displayModeBar: false,
          displaylogo: false,
          scrollZoom: false,
          staticPlot: false
        }).then(function() {
          msg({ type: 'loaded' });
          reportHeight();
          var gd = document.getElementById('chart');
          gd.on('plotly_afterplot', function() { setTimeout(reportHeight, 100); });
        });
      } catch(e) {
        document.getElementById('status').style.display = 'block';
        document.getElementById('status').innerText = 'Chart error: ' + e.message;
        msg({ type: 'error', height: 60, error: e.message });
      }
    }

    // Load full plotly bundle (plotly-basic misses anchor resolution for some layouts)
    var retries = 0;
    function loadPlotly() {
      var script = document.createElement('script');
      script.src = 'https://cdn.plot.ly/plotly-2.35.0.min.js';
      script.onload = function() { setTimeout(renderChart, 50); };
      script.onerror = function() {
        retries++;
        if (retries <= 2) {
          document.getElementById('status').style.display = 'block';
          document.getElementById('status').innerText = 'Retrying chart load...';
          setTimeout(loadPlotly, 1000 * retries);
        } else {
          document.getElementById('status').style.display = 'block';
          document.getElementById('status').innerText = 'Failed to load chart library.';
          msg({ type: 'error', height: 60, error: 'CDN load failed' });
        }
      };
      document.head.appendChild(script);
    }
    loadPlotly();
  </script>
</body>
</html>`;
  }, [chartJsonB64, chartWidth, chartHeight, fontSize]);

  if (!chartJsonB64) {
    return (
      <View style={[styles.chartContainer, { padding: 16 }]}>
        <Text style={{ color: colors.textSecondary, textAlign: "center" }}>
          Could not parse chart data
        </Text>
      </View>
    );
  }

  // Estimated height while loading (avoids layout jump when WebView reports real height)
  const displayHeight = isLoading ? chartHeight + 40 : webViewHeight;

  return (
    <View
      style={[styles.chartContainer, { height: displayHeight + 16 }]}
      onLayout={(e) => {
        const w = e.nativeEvent.layout.width;
        if (w > 0 && Math.abs(w - boxWidth) > 1) setBoxWidth(w);
      }}
    >
      {(isLoading || !measured) && <ChartSkeleton width={chartWidth} height={chartHeight} />}
      {measured && (
      <WebView
        originWhitelist={["*"]}
        source={{ html }}
        style={[
          { width: chartWidth, height: displayHeight, backgroundColor: "transparent" },
          isLoading && { position: "absolute", opacity: 0 },
        ]}
        scrollEnabled={false}
        nestedScrollEnabled
        javaScriptEnabled
        allowsInlineMediaPlayback
        mixedContentMode="compatibility"
        cacheEnabled
        cacheMode="LOAD_CACHE_ELSE_NETWORK"
        onMessage={(event: any) => {
          try {
            const data = JSON.parse(event.nativeEvent.data);
            if (data.type === "loaded") {
              setIsLoading(false);
            }
            if (data.height) {
              setWebViewHeight(data.height);
            }
            if (data.type === "error") {
              setIsLoading(false);
              setError(data.error);
            }
          } catch {}
        }}
        onError={() => {
          setIsLoading(false);
          setError("WebView failed to load");
        }}
      />
      )}
      {error && (
        <Text style={{ color: colors.textTertiary, fontSize: 11, textAlign: "center", marginTop: 4 }}>
          {error}
        </Text>
      )}
    </View>
  );
}

// ---- Web implementation (unchanged) ----

function PlotlyChartWeb({ chartJson }: { chartJson: string }) {
  const ref = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    if (!ref.current) return;

    const loadPlotly = async () => {
      try {
        const Plotly = (window as any).Plotly || (await import("plotly.js-dist-min" as any)).default;
        const parsed = JSON.parse(chartJson);
        const data = parsed.data || [];
        const layout = parsed.layout || {};

        // Normalize title: Plotly v2 expects { text: "..." } not a bare string
        if (typeof layout.title === "string") {
          layout.title = { text: layout.title };
        }
        // Ensure axis objects exist so Plotly doesn't crash reading .anchor on undefined
        if (layout.xaxis !== undefined && typeof layout.xaxis !== "object") {
          delete layout.xaxis;
        }
        if (layout.yaxis !== undefined && typeof layout.yaxis !== "object") {
          delete layout.yaxis;
        }

        Plotly.newPlot(
          ref.current,
          data,
          {
            ...layout,
            autosize: true,
            margin: { l: 40, r: 20, t: 40, b: 40 },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
          },
          { responsive: true, displayModeBar: false }
        );
      } catch (e) {
        console.error("Failed to render chart:", e);
      }
    };

    if (!(window as any).Plotly) {
      const script = document.createElement("script");
      script.src = "https://cdn.plot.ly/plotly-2.35.0.min.js";
      script.onload = loadPlotly;
      document.head.appendChild(script);
    } else {
      loadPlotly();
    }
  }, [chartJson]);

  return (
    <View style={styles.chartContainer}>
      <div ref={ref} style={{ width: "100%", minHeight: 350 }} />
    </View>
  );
}

const styles = StyleSheet.create({
  chartContainer: {
    marginTop: 8,
    borderRadius: 12,
    overflow: "hidden",
    backgroundColor: colors.surface,
    padding: 4,
  },
  skeletonContainer: {
    padding: 16,
    justifyContent: "center",
    alignItems: "center",
  },
  skeletonBlock: {
    backgroundColor: colors.surfaceBorder,
    borderRadius: 4,
  },
});
