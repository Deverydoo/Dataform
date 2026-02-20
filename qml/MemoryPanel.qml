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
    property var leanTraitsData: []

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 15
        anchors.rightMargin: 24
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

            Button {
                text: "Lean"
                flat: true
                font.pixelSize: 12
                font.bold: tabStack.currentIndex === 5

                contentItem: Text {
                    text: parent.text
                    color: tabStack.currentIndex === 5 ? "#ce9178" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: tabStack.currentIndex === 5 ? "#3a2a1e" : "transparent"
                    radius: 4
                }

                onClicked: {
                    tabStack.currentIndex = 5
                    refreshLean()
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

            // ===== Episodes tab (index 0) =====
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

            // ===== Traits tab (index 1) — Enhanced =====
            ListView {
                id: traitList
                clip: true
                spacing: 4
                model: traitsData

                // Section header + trait card
                delegate: Column {
                    width: traitList.width
                    spacing: 0

                    // Section header: VALUES & POSITIONS / BEHAVIORAL TRAITS
                    Rectangle {
                        width: parent.width
                        height: visible ? 26 : 0
                        color: "transparent"
                        visible: {
                            if (index === 0 && modelData.isValueOrPolicy) return true
                            if (index > 0 && !modelData.isValueOrPolicy) {
                                var prev = traitsData[index - 1]
                                if (prev && prev.isValueOrPolicy) return true
                            }
                            return false
                        }

                        Label {
                            anchors.left: parent.left
                            anchors.leftMargin: 4
                            anchors.verticalCenter: parent.verticalCenter
                            text: modelData.isValueOrPolicy ? "VALUES & POSITIONS" : "BEHAVIORAL TRAITS"
                            font.pixelSize: 11
                            font.bold: true
                            font.letterSpacing: 1
                            color: modelData.isValueOrPolicy ? "#ce9178" : "#569cd6"
                        }
                    }

                    // Trait card
                    Rectangle {
                        width: parent.width
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

                                // Weight tier badge (shown when evidence >= 3)
                                Rectangle {
                                    width: weightLabel.width + 10
                                    height: 16
                                    radius: 8
                                    visible: (modelData.evidenceCount || 0) >= 3
                                    color: {
                                        var tier = modelData.weightTier || "new"
                                        if (tier === "strong") return "#2d5a2d"
                                        if (tier === "moderate") return "#2d3a5a"
                                        return "#3e3e42"
                                    }

                                    Label {
                                        id: weightLabel
                                        anchors.centerIn: parent
                                        text: modelData.weightTier || ""
                                        font.pixelSize: 9
                                        color: "#cccccc"
                                    }
                                }

                                Item { Layout.fillWidth: true }

                                // Evidence count — prominent display
                                Label {
                                    text: {
                                        var count = modelData.evidenceCount || 0
                                        if (count === 0) return "No confirmations yet"
                                        if (count === 1) return "Seen in 1 conversation"
                                        return "Confirmed in " + count + " conversations"
                                    }
                                    font.pixelSize: 10
                                    font.bold: (modelData.evidenceCount || 0) >= 5
                                    color: {
                                        var count = modelData.evidenceCount || 0
                                        if (count >= 10) return "#4ec9b0"
                                        if (count >= 5) return "#569cd6"
                                        return "#888888"
                                    }
                                }

                                // Delete trait button
                                Button {
                                    width: 20
                                    height: 20
                                    flat: true
                                    contentItem: Text {
                                        text: "x"
                                        color: parent.hovered ? "#ff6666" : "#555555"
                                        font.pixelSize: 12
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#3a2020" : "transparent"
                                        radius: 10
                                    }
                                    ToolTip.text: "Remove this trait"
                                    ToolTip.visible: hovered
                                    onClicked: {
                                        memoryStore.removeTrait(modelData.traitId)
                                        refreshTraits()
                                    }
                                }
                            }

                            Label {
                                text: modelData.statement || ""
                                font.pixelSize: 12
                                color: "#cccccc"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Color-graded confidence bar
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
                                        color: {
                                            var conf = modelData.confidence || 0
                                            if (conf >= 0.7) return "#4ec9b0"
                                            if (conf >= 0.4) return "#569cd6"
                                            return "#888888"
                                        }
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

            // ===== Research tab (index 2) =====
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

            // ===== Goals tab (index 3) =====
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

            // ===== Analytics tab (index 4) =====
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

            // ===== Lean tab (index 5) — Political Leaning Meter =====
            ColumnLayout {
                spacing: 8

                // --- Fixed meter section (no scrollbar) ---

                // Disclaimer
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: disclaimerText.implicitHeight + 16
                    color: "#2d2a26"
                    radius: 6
                    border.color: "#5a4a3a"
                    border.width: 1

                    Label {
                        id: disclaimerText
                        anchors.fill: parent
                        anchors.margins: 8
                        text: "This is a curiosity and self-knowledge tool, not a judgment. " +
                              "Political positions are complex and multidimensional \u2014 this " +
                              "simplified meter is derived from your expressed values and policy views."
                        font.pixelSize: 10
                        font.italic: true
                        color: "#999988"
                        wrapMode: Text.WordWrap
                    }
                }

                // 5-Point Meter card
                Rectangle {
                    Layout.fillWidth: true
                    implicitHeight: meterColumn.implicitHeight + 24
                    color: "#2d2d30"
                    radius: 8

                    ColumnLayout {
                        id: meterColumn
                        anchors.fill: parent
                        anchors.margins: 12
                        spacing: 8

                        // Meter gradient bar
                        Rectangle {
                            Layout.fillWidth: true
                            height: 32
                            radius: 16

                            gradient: Gradient {
                                orientation: Gradient.Horizontal
                                GradientStop { position: 0.0; color: "#2266bb" }
                                GradientStop { position: 0.25; color: "#5577cc" }
                                GradientStop { position: 0.5; color: "#8844aa" }
                                GradientStop { position: 0.75; color: "#cc5555" }
                                GradientStop { position: 1.0; color: "#bb2222" }
                            }

                            // Marker indicator
                            Rectangle {
                                id: leanMarker
                                width: 20
                                height: 40
                                radius: 10
                                color: "#ffffff"
                                border.color: "#000000"
                                border.width: 2
                                y: -4
                                x: {
                                    var score = typeof traitExtractor !== "undefined" ? traitExtractor.leanScore : 0.0
                                    var normalized = (score + 1.0) / 2.0
                                    return Math.max(0, Math.min(parent.width - width,
                                        normalized * (parent.width - width)))
                                }

                                Behavior on x {
                                    NumberAnimation { duration: 500; easing.type: Easing.InOutQuad }
                                }
                            }
                        }

                        // 5-point labels
                        RowLayout {
                            Layout.fillWidth: true

                            Label { text: "Left"; font.pixelSize: 10; color: "#6688cc"; Layout.fillWidth: true; horizontalAlignment: Text.AlignLeft }
                            Label { text: "Lean Left"; font.pixelSize: 10; color: "#7799cc"; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                            Label { text: "Center"; font.pixelSize: 10; color: "#aa77cc"; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                            Label { text: "Lean Right"; font.pixelSize: 10; color: "#cc7777"; Layout.fillWidth: true; horizontalAlignment: Text.AlignHCenter }
                            Label { text: "Right"; font.pixelSize: 10; color: "#cc6666"; Layout.fillWidth: true; horizontalAlignment: Text.AlignRight }
                        }

                        // Score and label
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 12

                            Label {
                                text: typeof traitExtractor !== "undefined" && traitExtractor.leanAnalyzedTs
                                      ? traitExtractor.leanLabel : "Not yet analyzed"
                                font.pixelSize: 18
                                font.bold: true
                                color: "#ffffff"
                            }

                            Item { Layout.fillWidth: true }

                            Label {
                                text: typeof traitExtractor !== "undefined" && traitExtractor.leanAnalyzedTs
                                      ? "Last analyzed: " + traitExtractor.leanAnalyzedTs
                                      : ""
                                font.pixelSize: 10
                                color: "#666666"
                            }
                        }

                        // Analyze button
                        Button {
                            text: typeof traitExtractor !== "undefined" && traitExtractor.isAnalyzingLean
                                  ? "Analyzing..." : "Analyze Now"
                            flat: true
                            enabled: typeof traitExtractor !== "undefined" && !traitExtractor.isAnalyzingLean
                            Layout.alignment: Qt.AlignRight

                            contentItem: Text {
                                text: parent.text
                                color: parent.enabled ? "#ce9178" : "#666666"
                                font.pixelSize: 12
                                horizontalAlignment: Text.AlignHCenter
                            }
                            background: Rectangle {
                                color: parent.hovered ? "#3a2a1e" : "#2d2d30"
                                radius: 4
                                border.color: "#ce9178"
                                border.width: 1
                            }

                            onClicked: {
                                traitExtractor.analyzePoliticalLean()
                            }
                        }
                    }
                }

                // --- Scrollable contributions section ---

                // Separator
                Rectangle { Layout.fillWidth: true; height: 1; color: "#3e3e42" }

                // Contributing values header
                Label {
                    text: "Contributing Values & Positions"
                    font.pixelSize: 14
                    font.bold: true
                    color: "#ce9178"
                }

                // Scrollable values list
                ListView {
                    id: leanTraitsList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    spacing: 4
                    model: leanTraitsData

                    delegate: Rectangle {
                        width: leanTraitsList.width
                        height: leanTraitRow.implicitHeight + 12
                        color: index % 2 === 0 ? "#2d2d30" : "#252526"
                        radius: 4

                        RowLayout {
                            id: leanTraitRow
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 6
                            spacing: 8

                            // Type badge
                            Rectangle {
                                width: ltBadge.width + 10
                                height: 16
                                radius: 8
                                color: modelData.type === "value" ? "#2d5a2d" : "#5a4a2d"

                                Label {
                                    id: ltBadge
                                    anchors.centerIn: parent
                                    text: modelData.type || ""
                                    font.pixelSize: 9
                                    color: "#cccccc"
                                }
                            }

                            Label {
                                text: modelData.statement || ""
                                font.pixelSize: 11
                                color: "#cccccc"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                maximumLineCount: 2
                                elide: Text.ElideRight
                            }

                            Label {
                                text: ((modelData.confidence || 0) * 100).toFixed(0) + "%"
                                font.pixelSize: 10
                                color: "#888888"
                                Layout.preferredWidth: 35
                            }
                        }
                    }

                    // Empty state
                    Label {
                        anchors.centerIn: parent
                        visible: leanTraitsData.length === 0
                        text: "No value or policy traits detected yet.\nAs DATAFORM learns your views from conversations,\nthey will appear here."
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
        function onLeanAnalysisComplete(score) {
            console.log("MemoryPanel: lean analysis complete, score=" + score)
            if (tabStack.currentIndex === 5) refreshLean()
        }
    }

    function refreshEpisodes() {
        episodesData = memoryStore.getRecentEpisodesForQml(30)
    }

    function refreshTraits() {
        var all = memoryStore.getAllTraitsForQml()
        // Sort: values & policies first, then behavioral traits
        var valuesFirst = all.filter(function(t) { return t.isValueOrPolicy })
        var others = all.filter(function(t) { return !t.isValueOrPolicy })
        traitsData = valuesFirst.concat(others)
    }

    function refreshResearch() {
        researchData = researchStore.getRecentFindingsForQml(50)
    }

    function refreshGoals() {
        goalsData = memoryStore.getActiveGoalsForQml()
    }

    function refreshLean() {
        leanTraitsData = memoryStore.getValueAndPolicyTraitsForQml()
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
