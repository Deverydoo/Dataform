import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtCharts

Rectangle {
    id: memoryRoot
    color: "#252526"
    radius: 6

    property var episodesData: []
    property var traitsData: []
    property var researchData: []
    property var goalsData: []

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 15
        spacing: 8

        // Header with tabs
        RowLayout {
            Layout.fillWidth: true
            spacing: 4

            Label {
                text: "Memory"
                font.pixelSize: 16
                font.bold: true
                color: "#ffffff"
            }

            Item { Layout.fillWidth: true }

            // Tab buttons
            Button {
                text: "Episodes"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 0

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 0 ? "#4ec9b0" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 0 ? "#1e3a5f" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 0
                    refreshEpisodes()
                }
            }

            Button {
                text: "Traits"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 1

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 1 ? "#4ec9b0" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 1 ? "#1e3a5f" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 1
                    refreshTraits()
                }
            }

            Button {
                text: "Research"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 2

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 2 ? "#569cd6" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 2 ? "#1e2a3f" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 2
                    refreshResearch()
                }
            }

            Button {
                text: "Goals"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 3

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 3 ? "#4ec9b0" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 3 ? "#1e3a2f" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 3
                    refreshGoals()
                }
            }

            Button {
                text: "Analytics"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 4

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 4 ? "#dcdcaa" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 4 ? "#3a3a1e" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 4
                    refreshAnalytics()
                }
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // Tab content
        StackLayout {
            id: tabStack
            Layout.fillWidth: true
            Layout.fillHeight: true
            currentIndex: 0

            // Episodes tab
            ListView {
                id: episodeList
                clip: true
                spacing: 4
                model: episodesData

                delegate: Rectangle {
                    width: episodeList.width
                    height: episodeContent.height + 16
                    color: index % 2 === 0 ? "#2d2d30" : "#252526"
                    radius: 4

                    ColumnLayout {
                        id: episodeContent
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 8
                        spacing: 4

                        RowLayout {
                            Layout.fillWidth: true

                            Label {
                                text: modelData.timestamp || ""
                                font.pixelSize: 10
                                color: "#666666"
                            }

                            Item { Layout.fillWidth: true }

                            // Feedback indicator
                            Label {
                                text: {
                                    if (modelData.userFeedback === 1) return "\uD83D\uDC4D"
                                    if (modelData.userFeedback === -1) return "\uD83D\uDC4E"
                                    return ""
                                }
                                font.pixelSize: 12
                                visible: modelData.userFeedback !== 0
                            }

                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: modelData.corrected ? "#ffa500" : "transparent"
                                visible: modelData.corrected === true
                            }
                        }

                        Label {
                            text: {
                                var t = modelData.userText || ""
                                return t.length > 80 ? t.substring(0, 80) + "..." : t
                            }
                            font.pixelSize: 12
                            color: "#cccccc"
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                        }
                    }
                }

                // Empty state
                Label {
                    anchors.centerIn: parent
                    text: "No episodes yet.\nStart a conversation to build memory."
                    color: "#666666"
                    font.pixelSize: 13
                    horizontalAlignment: Text.AlignHCenter
                    visible: episodesData.length === 0
                }
            }

            // Traits tab
            ListView {
                id: traitList
                clip: true
                spacing: 4
                model: traitsData

                delegate: Rectangle {
                    width: traitList.width
                    height: traitContent.height + 16
                    color: index % 2 === 0 ? "#2d2d30" : "#252526"
                    radius: 4

                    ColumnLayout {
                        id: traitContent
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 8
                        spacing: 4

                        RowLayout {
                            Layout.fillWidth: true

                            // Type badge
                            Rectangle {
                                width: typeBadge.width + 12
                                height: 18
                                radius: 9
                                color: {
                                    var t = modelData.type || ""
                                    if (t === "value") return "#2d5a2d"
                                    if (t === "preference") return "#1e3a5f"
                                    if (t === "policy") return "#5a4a2d"
                                    if (t === "motivation") return "#4a2d5a"
                                    return "#3e3e42"
                                }

                                Label {
                                    id: typeBadge
                                    anchors.centerIn: parent
                                    text: modelData.type || "unknown"
                                    font.pixelSize: 10
                                    color: "#cccccc"
                                }
                            }

                            Item { Layout.fillWidth: true }

                            Label {
                                text: "evidence: " + (modelData.evidenceCount || 0)
                                font.pixelSize: 10
                                color: "#666666"
                            }
                        }

                        Label {
                            text: modelData.statement || ""
                            font.pixelSize: 12
                            color: "#cccccc"
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                        }

                        // Confidence bar
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Rectangle {
                                Layout.fillWidth: true
                                height: 4
                                radius: 2
                                color: "#3e3e42"

                                Rectangle {
                                    width: parent.width * (modelData.confidence || 0)
                                    height: parent.height
                                    radius: 2
                                    color: "#4ec9b0"
                                }
                            }

                            Label {
                                text: ((modelData.confidence || 0) * 100).toFixed(0) + "%"
                                font.pixelSize: 10
                                color: "#888888"
                                Layout.preferredWidth: 35
                            }
                        }
                    }
                }

                // Empty state with scan button
                ColumnLayout {
                    anchors.centerIn: parent
                    width: parent.width - 20
                    spacing: 12
                    visible: traitsData.length === 0

                    Label {
                        text: "No traits learned yet."
                        color: "#666666"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    Button {
                        text: traitExtractor.isExtracting ? "Scanning..." : "Scan Conversations"
                        flat: true
                        enabled: !traitExtractor.isExtracting
                        Layout.alignment: Qt.AlignHCenter

                        contentItem: Text {
                            text: parent.text
                            color: parent.enabled ? "#4ec9b0" : "#666666"
                            font.pixelSize: 12
                            horizontalAlignment: Text.AlignHCenter
                        }
                        background: Rectangle {
                            color: parent.hovered ? "#2d3a2d" : "#2d2d30"
                            radius: 4
                            border.color: "#4ec9b0"
                            border.width: 1
                        }

                        onClicked: {
                            traitExtractor.scanConversationsForTraits()
                        }
                    }

                    Label {
                        text: "Traits will also be scanned during idle time."
                        color: "#555555"
                        font.pixelSize: 11
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                    }

                    // Diagnostic status
                    Label {
                        text: traitExtractor.lastStatus || ""
                        color: "#888844"
                        font.pixelSize: 10
                        horizontalAlignment: Text.AlignHCenter
                        Layout.alignment: Qt.AlignHCenter
                        Layout.fillWidth: true
                        wrapMode: Text.WordWrap
                        visible: (traitExtractor.lastStatus || "").length > 0
                    }

                    // Last LLM response (persistent diagnostic)
                    Label {
                        text: "Last LLM Response:"
                        color: "#666666"
                        font.pixelSize: 10
                        font.bold: true
                        Layout.alignment: Qt.AlignHCenter
                        visible: (traitExtractor.lastResponse || "").length > 0
                    }

                    ScrollView {
                        id: responseScroll
                        Layout.fillWidth: true
                        Layout.preferredHeight: Math.min(200, responseLabel.implicitHeight + 10)
                        Layout.alignment: Qt.AlignHCenter
                        visible: (traitExtractor.lastResponse || "").length > 0
                        clip: true

                        Label {
                            id: responseLabel
                            text: traitExtractor.lastResponse || ""
                            color: "#aaaaaa"
                            font.pixelSize: 10
                            font.family: "Consolas"
                            width: responseScroll.availableWidth
                            wrapMode: Text.WrapAnywhere
                        }
                    }
                }
            }

            // Research tab
            ColumnLayout {
                spacing: 4

                // Approve All button
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    visible: researchData.length > 0

                    Label {
                        text: researchStore.pendingCount + " pending"
                        font.pixelSize: 11
                        color: "#569cd6"
                    }

                    Item { Layout.fillWidth: true }

                    Button {
                        text: "Approve All"
                        flat: true
                        visible: researchStore.pendingCount > 0
                        contentItem: Text {
                            text: parent.text
                            color: "#4ec9b0"
                            font.pixelSize: 11
                        }
                        onClicked: {
                            researchStore.approveAllPending()
                            refreshResearch()
                        }
                    }
                }

                ListView {
                    id: researchList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 4
                    model: researchData

                    delegate: Rectangle {
                        width: researchList.width
                        height: researchContent.height + 16
                        color: index % 2 === 0 ? "#2d2d30" : "#252526"
                        radius: 4

                        ColumnLayout {
                            id: researchContent
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 8
                            spacing: 4

                            RowLayout {
                                Layout.fillWidth: true

                                // Status badge
                                Rectangle {
                                    width: statusLabel.width + 10
                                    height: 16
                                    radius: 8
                                    color: {
                                        if (modelData.status === 1) return "#2d5a2d"
                                        if (modelData.status === -1) return "#5a2d2d"
                                        return "#2d3a5a"
                                    }

                                    Label {
                                        id: statusLabel
                                        anchors.centerIn: parent
                                        text: {
                                            if (modelData.status === 1) return "approved"
                                            if (modelData.status === -1) return "rejected"
                                            return "pending"
                                        }
                                        font.pixelSize: 9
                                        color: "#cccccc"
                                    }
                                }

                                Label {
                                    text: modelData.topic || ""
                                    font.pixelSize: 11
                                    font.bold: true
                                    color: "#569cd6"
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }

                                Label {
                                    text: modelData.timestamp || ""
                                    font.pixelSize: 10
                                    color: "#666666"
                                }
                            }

                            // Summary
                            Label {
                                text: modelData.llmSummary || ""
                                font.pixelSize: 12
                                color: "#cccccc"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Relevance bar + source
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Rectangle {
                                    width: 60
                                    height: 4
                                    radius: 2
                                    color: "#3e3e42"

                                    Rectangle {
                                        width: parent.width * (modelData.relevanceScore || 0)
                                        height: parent.height
                                        radius: 2
                                        color: "#569cd6"
                                    }
                                }

                                Label {
                                    text: ((modelData.relevanceScore || 0) * 100).toFixed(0) + "%"
                                    font.pixelSize: 10
                                    color: "#888888"
                                }

                                Label {
                                    text: modelData.sourceTitle || modelData.sourceUrl || ""
                                    font.pixelSize: 10
                                    color: "#666666"
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }
                            }

                            // Approve/Reject buttons for pending
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                visible: modelData.status === 0

                                Button {
                                    text: "Approve"
                                    flat: true
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#4ec9b0"
                                        font.pixelSize: 11
                                    }
                                    onClicked: {
                                        researchStore.approveFinding(modelData.findingId)
                                        refreshResearch()
                                    }
                                }

                                Button {
                                    text: "Reject"
                                    flat: true
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#ff6666"
                                        font.pixelSize: 11
                                    }
                                    onClicked: {
                                        researchStore.rejectFinding(modelData.findingId)
                                        refreshResearch()
                                    }
                                }

                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // Empty state
                    Label {
                        anchors.centerIn: parent
                        text: "No research findings yet.\nDATAFORM will research during idle time."
                        color: "#666666"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        visible: researchData.length === 0
                    }
                }
            }

            // Goals tab
            ColumnLayout {
                spacing: 4

                // Active goals header
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    visible: goalsData.length > 0

                    Label {
                        text: (typeof goalTracker !== "undefined" ? goalTracker.activeGoalCount : 0) + " active"
                        font.pixelSize: 11
                        color: "#4ec9b0"
                    }

                    Item { Layout.fillWidth: true }

                    Label {
                        text: "Mood: " + (typeof sentimentTracker !== "undefined" && sentimentTracker.currentMood ? sentimentTracker.currentMood : "N/A")
                        font.pixelSize: 11
                        color: "#888888"
                    }
                }

                ListView {
                    id: goalsList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 4
                    model: goalsData

                    delegate: Rectangle {
                        width: goalsList.width
                        height: goalContent.height + 16
                        color: index % 2 === 0 ? "#2d2d30" : "#252526"
                        radius: 4

                        ColumnLayout {
                            id: goalContent
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 8
                            spacing: 4

                            RowLayout {
                                Layout.fillWidth: true

                                // Status badge
                                Rectangle {
                                    width: goalStatusLabel.width + 10
                                    height: 16
                                    radius: 8
                                    color: {
                                        if (modelData.status === 1) return "#2d5a2d"
                                        if (modelData.status === 2) return "#5a2d2d"
                                        return "#2d3a5a"
                                    }

                                    Label {
                                        id: goalStatusLabel
                                        anchors.centerIn: parent
                                        text: {
                                            if (modelData.status === 1) return "achieved"
                                            if (modelData.status === 2) return "abandoned"
                                            return "active"
                                        }
                                        font.pixelSize: 9
                                        color: "#cccccc"
                                    }
                                }

                                Label {
                                    text: modelData.title || ""
                                    font.pixelSize: 12
                                    font.bold: true
                                    color: "#d4d4d4"
                                    elide: Text.ElideRight
                                    Layout.fillWidth: true
                                }

                                Label {
                                    text: modelData.createdTs || ""
                                    font.pixelSize: 10
                                    color: "#666666"
                                }
                            }

                            Label {
                                text: modelData.description || ""
                                font.pixelSize: 11
                                color: "#aaaaaa"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                visible: (modelData.description || "").length > 0
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: "Check-ins: " + (modelData.checkinCount || 0)
                                    font.pixelSize: 10
                                    color: "#666666"
                                }

                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // Empty state
                    Label {
                        anchors.centerIn: parent
                        text: "No goals detected yet.\nDATAFORM will detect goals from your conversations."
                        color: "#666666"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                        visible: goalsData.length === 0
                    }
                }

                // Learning plans section
                ColumnLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    visible: typeof learningEngine !== "undefined" && learningEngine.hasActivePlan

                    Rectangle {
                        Layout.fillWidth: true
                        height: 1
                        color: "#3e3e42"
                    }

                    Label {
                        text: "Learning Plans"
                        font.pixelSize: 12
                        font.bold: true
                        color: "#569cd6"
                    }

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 10

                        Label {
                            text: typeof learningEngine !== "undefined" ? learningEngine.currentTopic : ""
                            font.pixelSize: 12
                            color: "#d4d4d4"
                            Layout.fillWidth: true
                            elide: Text.ElideRight
                        }

                        Label {
                            text: typeof learningEngine !== "undefined"
                                  ? "Lesson " + learningEngine.currentLesson + "/" + learningEngine.totalLessons
                                  : ""
                            font.pixelSize: 11
                            color: "#4ec9b0"
                        }
                    }
                }
            }

            // Analytics tab
            ScrollView {
                clip: true

                ColumnLayout {
                    width: tabStack.width
                    spacing: 12

                    // Mood Over Time
                    Label {
                        text: "Mood Over Time (30 days)"
                        font.pixelSize: 14
                        font.bold: true
                        color: "#dcdcaa"
                    }

                    ChartView {
                        id: moodChart
                        Layout.fillWidth: true
                        Layout.preferredHeight: 200
                        antialiasing: true
                        legend.visible: false
                        backgroundColor: "#1e1e1e"
                        plotAreaColor: "#252526"

                        ValuesAxis {
                            id: moodYAxis
                            min: -1
                            max: 1
                            tickCount: 5
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                        }

                        ValuesAxis {
                            id: moodXAxis
                            min: 0
                            max: 30
                            tickCount: 6
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                            labelFormat: "%d"
                        }

                        LineSeries {
                            id: moodSeries
                            axisX: moodXAxis
                            axisY: moodYAxis
                            color: "#4ec9b0"
                            width: 2
                        }

                        LineSeries {
                            id: energySeries
                            axisX: moodXAxis
                            axisY: moodYAxis
                            color: "#569cd6"
                            width: 1
                            style: Qt.DashLine
                        }
                    }

                    RowLayout {
                        spacing: 16
                        Label { text: "\u2014 Sentiment"; color: "#4ec9b0"; font.pixelSize: 10 }
                        Label { text: "--- Energy"; color: "#569cd6"; font.pixelSize: 10 }
                    }

                    // Trait Growth
                    Label {
                        text: "Trait Growth (90 days)"
                        font.pixelSize: 14
                        font.bold: true
                        color: "#dcdcaa"
                    }

                    ChartView {
                        id: traitChart
                        Layout.fillWidth: true
                        Layout.preferredHeight: 180
                        antialiasing: true
                        legend.visible: false
                        backgroundColor: "#1e1e1e"
                        plotAreaColor: "#252526"

                        ValuesAxis {
                            id: traitYAxis
                            min: 0
                            max: 10
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                        }

                        ValuesAxis {
                            id: traitXAxis
                            min: 0
                            max: 90
                            tickCount: 4
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                            labelFormat: "%d"
                        }

                        AreaSeries {
                            axisX: traitXAxis
                            axisY: traitYAxis
                            color: "#2d5a2d"
                            borderColor: "#4ec9b0"
                            borderWidth: 2

                            upperSeries: LineSeries {
                                id: traitLineSeries
                            }
                        }
                    }

                    // Episode Activity
                    Label {
                        text: "Conversation Activity (30 days)"
                        font.pixelSize: 14
                        font.bold: true
                        color: "#dcdcaa"
                    }

                    ChartView {
                        id: activityChart
                        Layout.fillWidth: true
                        Layout.preferredHeight: 180
                        antialiasing: true
                        legend.visible: false
                        backgroundColor: "#1e1e1e"
                        plotAreaColor: "#252526"

                        ValuesAxis {
                            id: actYAxis
                            min: 0
                            max: 10
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                        }

                        BarCategoryAxis {
                            id: actXAxis
                            labelsColor: "#888888"
                            gridLineColor: "#3e3e42"
                            lineVisible: false
                        }

                        BarSeries {
                            id: activityBarSeries
                            axisX: actXAxis
                            axisY: actYAxis

                            BarSet {
                                id: activityBarSet
                                label: "Episodes"
                                color: "#569cd6"
                                borderColor: "#3e3e42"
                            }
                        }
                    }

                    // Summary stats
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: statsRow.implicitHeight + 20
                        color: "#2d2d30"
                        radius: 6

                        RowLayout {
                            id: statsRow
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 20

                            ColumnLayout {
                                spacing: 2
                                Label { text: memoryStore.episodeCount; font.pixelSize: 20; font.bold: true; color: "#4ec9b0" }
                                Label { text: "Episodes"; font.pixelSize: 10; color: "#888888" }
                            }

                            ColumnLayout {
                                spacing: 2
                                Label { text: memoryStore.traitCount; font.pixelSize: 20; font.bold: true; color: "#569cd6" }
                                Label { text: "Traits"; font.pixelSize: 10; color: "#888888" }
                            }

                            ColumnLayout {
                                spacing: 2
                                Label {
                                    text: typeof sentimentTracker !== "undefined" && sentimentTracker.currentMood
                                          ? sentimentTracker.currentMood : "N/A"
                                    font.pixelSize: 16
                                    font.bold: true
                                    color: "#dcdcaa"
                                }
                                Label { text: "Current Mood"; font.pixelSize: 10; color: "#888888" }
                            }

                            ColumnLayout {
                                spacing: 2
                                Label {
                                    text: typeof sentimentTracker !== "undefined" && sentimentTracker.moodTrend
                                          ? sentimentTracker.moodTrend : "N/A"
                                    font.pixelSize: 16
                                    font.bold: true
                                    color: {
                                        if (text === "improving") return "#4ec9b0"
                                        if (text === "declining") return "#d4534b"
                                        return "#888888"
                                    }
                                }
                                Label { text: "Mood Trend"; font.pixelSize: 10; color: "#888888" }
                            }

                            Item { Layout.fillWidth: true }
                        }
                    }

                    // Empty state overlay
                    Label {
                        Layout.fillWidth: true
                        visible: memoryStore.episodeCount === 0
                        text: "No data yet. Analytics will populate as you chat with DATAFORM."
                        color: "#666666"
                        font.pixelSize: 13
                        horizontalAlignment: Text.AlignHCenter
                    }
                }
            }
        }
    }

    // Refresh data when episodes/traits are inserted
    Connections {
        target: memoryStore

        function onEpisodeInserted(id) {
            refreshEpisodes()
        }

        function onTraitInserted(traitId) {
            refreshTraits()
        }

        function onEpisodeCountChanged() {
            refreshEpisodes()
            if (tabStack.currentIndex === 4) refreshAnalytics()
        }

        function onTraitCountChanged() {
            refreshTraits()
        }
    }

    Connections {
        target: typeof sentimentTracker !== "undefined" ? sentimentTracker : null
        enabled: typeof sentimentTracker !== "undefined"
        function onSentimentAnalyzed() {
            if (tabStack.currentIndex === 4) refreshAnalytics()
        }
    }

    // Refresh goals when new goals are detected
    Connections {
        target: typeof goalTracker !== "undefined" ? goalTracker : null
        enabled: typeof goalTracker !== "undefined"
        function onGoalDetected() {
            refreshGoals()
        }
        function onActiveGoalCountChanged() {
            if (tabStack.currentIndex === 3) refreshGoals()
        }
    }

    // Refresh traits when extraction completes
    Connections {
        target: typeof traitExtractor !== "undefined" ? traitExtractor : null
        enabled: typeof traitExtractor !== "undefined"
        function onTraitsExtracted(count) {
            console.log("MemoryPanel: " + count + " traits extracted, refreshing")
            refreshTraits()
        }
        function onExtractionError(error) {
            console.log("MemoryPanel: trait extraction error: " + error)
        }
    }

    function refreshEpisodes() {
        episodesData = memoryStore.getRecentEpisodesForQml(30)
    }

    function refreshTraits() {
        traitsData = memoryStore.getAllTraitsForQml()
    }

    function refreshResearch() {
        researchData = researchStore.getRecentFindingsForQml(50)
    }

    function refreshGoals() {
        goalsData = memoryStore.getActiveGoalsForQml()
    }

    function refreshAnalytics() {
        // Mood chart
        moodSeries.clear()
        energySeries.clear()
        var sentimentData = sentimentTracker.getSentimentHistoryForQml(30)
        for (var i = 0; i < sentimentData.length; i++) {
            moodSeries.append(i, sentimentData[i].avgScore)
            energySeries.append(i, sentimentData[i].avgEnergy)
        }
        if (sentimentData.length > 0) {
            moodXAxis.max = Math.max(sentimentData.length - 1, 1)
        }

        // Trait growth chart
        traitLineSeries.clear()
        var traitData = memoryStore.getTraitGrowthForQml(90)
        var maxTraits = 10
        for (var j = 0; j < traitData.length; j++) {
            traitLineSeries.append(j, traitData[j].count)
            if (traitData[j].count > maxTraits) maxTraits = traitData[j].count
        }
        traitYAxis.max = maxTraits + 2
        if (traitData.length > 0) {
            traitXAxis.max = Math.max(traitData.length - 1, 1)
        }

        // Activity bar chart
        activityBarSet.values = []
        var categories = []
        var actData = memoryStore.getEpisodeActivityForQml(30)
        var maxAct = 10
        var values = []
        for (var k = 0; k < actData.length; k++) {
            // Show abbreviated date
            var d = actData[k].date
            categories.push(d.substring(5)) // MM-DD
            values.push(actData[k].count)
            if (actData[k].count > maxAct) maxAct = actData[k].count
        }
        actXAxis.categories = categories
        activityBarSet.values = values
        actYAxis.max = maxAct + 2
    }

    Connections {
        target: researchStore

        function onCountsChanged() {
            if (tabStack.currentIndex === 2) refreshResearch()
        }
    }

    Component.onCompleted: {
        refreshEpisodes()
        refreshTraits()
        refreshResearch()
        refreshGoals()
    }
}
