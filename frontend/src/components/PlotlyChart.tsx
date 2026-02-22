import React from "react";
import { StyleSheet, View, Platform, useWindowDimensions } from "react-native";
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

// ---- Native implementation using WebView + Plotly.js ----

function PlotlyChartNative({ chartJson }: { chartJson: string }) {
  const { WebView } = require("react-native-webview");
  const { width: screenWidth } = useWindowDimensions();
  const [webViewHeight, setWebViewHeight] = React.useState(320);
  const [error, setError] = React.useState<string | null>(null);

  // Detect chart type for layout adjustments
  const chartType = React.useMemo(() => {
    try {
      const parsed = JSON.parse(chartJson);
      const firstTrace = parsed.data?.[0];
      return firstTrace?.type || "bar";
    } catch {
      return "bar";
    }
  }, [chartJson]);

  const isPie = chartType === "pie";

  // Full-width chart with minimal padding for mobile
  const bubbleContentWidth = Math.floor(screenWidth * 0.96) - 16;
  const chartWidth = Math.max(bubbleContentWidth, 200);
  // Shorter aspect ratio for mobile: pie charts are squarer, bar/line are wider
  const chartHeight = isPie
    ? Math.round(chartWidth * 0.75)
    : Math.round(chartWidth * 0.65);
  const fontSize = Math.max(9, Math.min(12, Math.round(chartWidth / 28)));

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
    #status { color: #888; text-align: center; padding: 20px; font-size: 13px; }
    /* Hide modebar on mobile — not useful for touch */
    .modebar { display: none !important; }
    /* Improve touch target for pie slices and bar segments */
    .trace { cursor: pointer; }
    /* Prevent text selection on chart */
    .plot-container { -webkit-user-select: none; user-select: none; }
  </style>
</head>
<body>
  <div id="status">Loading chart...</div>
  <div id="chart"></div>
  <script>
    var CHART_B64 = "${chartJsonB64}";

    function msg(obj) { window.ReactNativeWebView.postMessage(JSON.stringify(obj)); }

    function reportHeight() {
      setTimeout(function() {
        var el = document.getElementById('chart');
        var h = el ? el.offsetHeight : 0;
        if (h > 0) msg({ height: h });
      }, 150);
    }

    function truncateLabel(text, maxLen) {
      if (!text || typeof text !== 'string') return text;
      return text.length > maxLen ? text.substring(0, maxLen - 1) + '…' : text;
    }

    function renderChart() {
      try {
        var raw = decodeURIComponent(escape(atob(CHART_B64)));
        var chartData = JSON.parse(raw);
        var isPie = chartData.data && chartData.data[0] && chartData.data[0].type === 'pie';

        // Truncate long x-axis labels for bar/line charts on mobile
        if (!isPie && chartData.data) {
          chartData.data.forEach(function(trace) {
            if (trace.x && Array.isArray(trace.x)) {
              trace.x = trace.x.map(function(v) { return truncateLabel(String(v), 16); });
            }
            // Add hover info for better mobile tap experience
            trace.hoverinfo = trace.hoverinfo || 'x+y+text';
          });
        }

        // Pie chart mobile optimizations
        if (isPie && chartData.data) {
          chartData.data.forEach(function(trace) {
            // Move labels outside for readability, show % only
            trace.textposition = 'outside';
            trace.textinfo = 'label+percent';
            trace.textfont = { size: ${fontSize} };
            trace.outsidetextfont = { size: ${Math.max(8, fontSize - 1)} };
            trace.insidetextorientation = 'horizontal';
            // Truncate long pie labels
            if (trace.labels && Array.isArray(trace.labels)) {
              trace.labels = trace.labels.map(function(v) { return truncateLabel(String(v), 14); });
            }
            trace.hoverinfo = 'label+value+percent';
            // Slight pull for visual separation
            trace.pull = 0.02;
            trace.hole = 0.3;
          });
        }

        var xaxis = Object.assign({}, chartData.layout && chartData.layout.xaxis || {}, {
          tickangle: -45,
          automargin: true,
          fixedrange: true,
          tickfont: { size: ${Math.max(8, fontSize - 1)} }
        });
        var yaxis = Object.assign({}, chartData.layout && chartData.layout.yaxis || {}, {
          automargin: true,
          fixedrange: true,
          tickfont: { size: ${Math.max(8, fontSize - 1)} }
        });

        var layout = Object.assign({}, chartData.layout || {}, {
          width: ${chartWidth},
          height: ${chartHeight},
          autosize: false,
          margin: isPie
            ? { l: 10, r: 10, t: 36, b: 10, pad: 0 }
            : { l: 4, r: 4, t: 36, b: 60, pad: 0, autoexpand: true },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { size: ${fontSize}, color: '#374151' },
          title: chartData.layout && chartData.layout.title ? {
            text: typeof chartData.layout.title === 'string'
              ? chartData.layout.title
              : (chartData.layout.title.text || ''),
            font: { size: ${fontSize + 2}, color: '#1e1e3a' },
            x: 0.5,
            xanchor: 'center',
            y: 0.98,
            yanchor: 'top'
          } : undefined,
          xaxis: isPie ? undefined : xaxis,
          yaxis: isPie ? undefined : yaxis,
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
            font: { size: ${fontSize}, color: '#fff', family: '-apple-system, sans-serif' }
          }
        });

        document.getElementById('status').style.display = 'none';

        Plotly.newPlot('chart', chartData.data, layout, {
          responsive: false,
          displayModeBar: false,
          displaylogo: false,
          scrollZoom: false,
          staticPlot: false
        }).then(function() {
          reportHeight();
          var gd = document.getElementById('chart');
          gd.on('plotly_afterplot', function() { setTimeout(reportHeight, 100); });
        });
      } catch(e) {
        document.getElementById('status').innerText = 'Chart error: ' + e.message;
        msg({ height: 60, error: e.message });
      }
    }

    // Load Plotly with retry
    var retries = 0;
    function loadPlotly() {
      var script = document.createElement('script');
      script.src = 'https://cdn.plot.ly/plotly-2.35.0.min.js';
      script.onload = function() { setTimeout(renderChart, 50); };
      script.onerror = function() {
        retries++;
        if (retries <= 2) {
          document.getElementById('status').innerText = 'Retrying chart load...';
          setTimeout(loadPlotly, 1000 * retries);
        } else {
          document.getElementById('status').innerText = 'Failed to load chart library. Check your connection.';
          msg({ height: 60, error: 'CDN load failed' });
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

  return (
    <View style={[styles.chartContainer, { height: webViewHeight + 24 }]}>
      <WebView
        originWhitelist={["*"]}
        source={{ html }}
        style={{ width: chartWidth, height: webViewHeight, backgroundColor: "transparent" }}
        scrollEnabled={false}
        nestedScrollEnabled
        javaScriptEnabled
        allowsInlineMediaPlayback
        mixedContentMode="compatibility"
        onMessage={(event: any) => {
          try {
            const data = JSON.parse(event.nativeEvent.data);
            if (data.height) {
              setWebViewHeight(data.height);
            }
            if (data.error) {
              setError(data.error);
            }
          } catch {}
        }}
        onError={() => {
          setError("WebView failed to load");
        }}
      />
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
        Plotly.newPlot(
          ref.current,
          parsed.data,
          {
            ...parsed.layout,
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
});
