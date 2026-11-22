import QtQuick

// A small chart drawn with Canvas: line, filled area or bars, with a value axis
// on the left, grid lines, and either a numeric or a category axis along the
// bottom. Colours follow the app's dark theme.
//
//   kind        "line" | "area" | "bar"
//   series      line/area: [{ values: [..], color, width, dashed }]
//               bar: [{ values: [..], color, borderColor }]
//               Point i of a series is drawn at x = i.
//   categories  bar labels, one per value (thinned out when they would overlap)
//
// Empty or missing data draws the axes and a short note instead of failing.
Canvas {
    id: chart

    property string kind: "line"
    property var series: []
    property var categories: []

    property real yMin: 0
    property real yMax: 10
    property int yTicks: 5
    property real xMin: 0
    property real xMax: 10
    property int xTicks: 5
    // Counts read better on whole-number grid lines: when set, the top of the
    // value axis is raised so every step between grid lines is an integer.
    property bool integerSteps: false

    property color fillColor: "#2d5a2d"
    property string emptyText: "No data yet"

    property color backgroundColor: "#1e1e1e"
    property color plotAreaColor: "#252526"
    property color gridColor: "#3e3e42"
    property color labelColor: "#888888"
    property int labelPixelSize: 10

    antialiasing: true

    onSeriesChanged: requestPaint()
    onCategoriesChanged: requestPaint()
    onYMinChanged: requestPaint()
    onYMaxChanged: requestPaint()
    onXMaxChanged: requestPaint()
    onWidthChanged: requestPaint()
    onHeightChanged: requestPaint()

    function formatTick(v, span, ticks) {
        var step = span / Math.max(ticks - 1, 1)
        if (Math.abs(step - Math.round(step)) < 1e-9 && Math.abs(v - Math.round(v)) < 1e-9)
            return Math.round(v).toString()
        return v.toFixed(1)
    }

    function hasData() {
        if (!series) return false
        for (var s = 0; s < series.length; s++) {
            if (series[s] && series[s].values && series[s].values.length > 0) return true
        }
        return false
    }

    onPaint: {
        var ctx = getContext("2d")
        ctx.reset()
        var w = width, h = height
        if (w <= 0 || h <= 0) return

        ctx.fillStyle = backgroundColor
        ctx.fillRect(0, 0, w, h)
        ctx.font = labelPixelSize + "px sans-serif"

        var steps = Math.max(yTicks - 1, 1)
        var top_ = yMax
        if (integerSteps) top_ = yMin + Math.ceil((yMax - yMin) / steps) * steps
        var ySpan = (top_ - yMin) || 1
        var xSpan = (xMax - xMin) || 1

        // Leave room for the widest value label on the left and one row below.
        var widest = 0
        for (var t = 0; t < yTicks; t++) {
            var tv = yMin + ySpan * t / Math.max(yTicks - 1, 1)
            widest = Math.max(widest, ctx.measureText(formatTick(tv, ySpan, yTicks)).width)
        }
        var left = Math.ceil(widest) + 14
        var right = 12
        var top = 10
        var bottom = labelPixelSize + 12
        var pw = w - left - right
        var ph = h - top - bottom
        if (pw <= 4 || ph <= 4) return

        ctx.fillStyle = plotAreaColor
        ctx.fillRect(left, top, pw, ph)

        function px(x) { return left + (x - xMin) / xSpan * pw }
        function py(y) { return top + ph - (y - yMin) / ySpan * ph }

        // Horizontal grid and value labels.
        ctx.lineWidth = 1
        ctx.strokeStyle = gridColor
        ctx.fillStyle = labelColor
        ctx.textAlign = "right"
        ctx.textBaseline = "middle"
        for (var i = 0; i < yTicks; i++) {
            var yv = yMin + ySpan * i / Math.max(yTicks - 1, 1)
            var yy = Math.round(py(yv)) + 0.5
            ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(left + pw, yy); ctx.stroke()
            ctx.fillText(formatTick(yv, ySpan, yTicks), left - 6, yy)
        }

        ctx.textAlign = "center"
        ctx.textBaseline = "top"
        var labelY = top + ph + 5

        if (kind === "bar") {
            var vals = (series && series.length > 0 && series[0] && series[0].values) ? series[0].values : []
            var n = vals.length
            if (n > 0) {
                var slot = pw / n
                var barW = Math.max(1, slot * 0.7)
                ctx.fillStyle = series[0].color || "#569cd6"
                ctx.strokeStyle = series[0].borderColor || gridColor
                for (var b = 0; b < n; b++) {
                    var v = Math.max(yMin, Math.min(yMin + ySpan, Number(vals[b]) || 0))
                    var bx = left + slot * b + (slot - barW) / 2
                    var by = py(v)
                    var bh = py(Math.max(yMin, 0)) - by
                    if (bh > 0) {
                        ctx.fillRect(bx, by, barW, bh)
                        ctx.strokeRect(bx + 0.5, by + 0.5, barW - 1, bh - 1)
                    }
                }
                // Category labels, skipping some when they would run into each other.
                var cats = categories || []
                var maxLabel = 0
                for (var c = 0; c < cats.length; c++)
                    maxLabel = Math.max(maxLabel, ctx.measureText(String(cats[c])).width)
                var every = Math.max(1, Math.ceil((maxLabel + 8) / slot))
                ctx.fillStyle = labelColor
                for (var k = 0; k < Math.min(n, cats.length); k += every)
                    ctx.fillText(String(cats[k]), left + slot * k + slot / 2, labelY)
            }
        } else {
            // Vertical grid and numeric labels along the bottom.
            ctx.strokeStyle = gridColor
            ctx.fillStyle = labelColor
            for (var j = 0; j < xTicks; j++) {
                var xv = xMin + xSpan * j / Math.max(xTicks - 1, 1)
                var xx = Math.round(px(xv)) + 0.5
                ctx.beginPath(); ctx.moveTo(xx, top); ctx.lineTo(xx, top + ph); ctx.stroke()
                ctx.fillText(Math.round(xv).toString(), xx, labelY)
            }

            ctx.save()
            ctx.beginPath()
            ctx.rect(left, top, pw, ph)
            ctx.clip()
            for (var s = 0; s < (series ? series.length : 0); s++) {
                var ser = series[s]
                if (!ser || !ser.values || ser.values.length === 0) continue
                var pts = ser.values
                var lineColor = ser.color || "#4ec9b0"

                if (kind === "area") {
                    var base = py(Math.max(yMin, 0))
                    ctx.beginPath()
                    ctx.moveTo(px(0), base)
                    for (var a = 0; a < pts.length; a++) ctx.lineTo(px(a), py(Number(pts[a]) || 0))
                    ctx.lineTo(px(pts.length - 1), base)
                    ctx.closePath()
                    ctx.fillStyle = fillColor
                    ctx.fill()
                }

                ctx.strokeStyle = lineColor
                ctx.lineWidth = ser.width || 2
                ctx.setLineDash(ser.dashed ? [5, 4] : [])
                ctx.beginPath()
                for (var p = 0; p < pts.length; p++) {
                    var X = px(p), Y = py(Number(pts[p]) || 0)
                    if (p === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y)
                }
                if (pts.length === 1) {
                    // A single point has no line to draw; mark it instead.
                    ctx.fillStyle = lineColor
                    ctx.fillRect(px(0) - 2, py(Number(pts[0]) || 0) - 2, 4, 4)
                } else {
                    ctx.stroke()
                }
                ctx.setLineDash([])
            }
            ctx.restore()
        }

        if (!hasData()) {
            ctx.fillStyle = labelColor
            ctx.textAlign = "center"
            ctx.textBaseline = "middle"
            ctx.font = (labelPixelSize + 2) + "px sans-serif"
            ctx.fillText(emptyText, left + pw / 2, top + ph / 2)
        }
    }
}
