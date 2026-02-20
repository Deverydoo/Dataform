import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: root
    width: 1400
    height: 900
    minimumWidth: 1000
    minimumHeight: 700
    visible: true
    title: "DATAFORM by GHI"
    color: "#1e1e1e"

    property int currentView: 0 // 0 = Dashboard, 1 = Settings, 2 = Lineage

    // Keyboard shortcuts
    Shortcut {
        sequence: "Ctrl+,"
        onActivated: {
            if (currentView === 1) currentView = 0
            else currentView = 1
        }
    }

    // Header
    Rectangle {
        id: header
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        height: 50
        color: "#2d2d30"

        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 20
            anchors.rightMargin: 20
            spacing: 15

            // Logo and title
            Label {
                text: "DATAFORM"
                font.pixelSize: 20
                font.bold: true
                color: "#4ec9b0"
                Layout.alignment: Qt.AlignVCenter
            }

            // Organization badge
            Label {
                text: "GHI"
                font.pixelSize: 11
                font.bold: true
                color: "#569cd6"
                Layout.alignment: Qt.AlignVCenter
            }

            // Separator dot
            Rectangle {
                width: 4
                height: 4
                radius: 2
                color: "#555555"
                Layout.alignment: Qt.AlignVCenter
            }

            // Subtitle
            Label {
                text: "Adaptive Personal AI"
                font.pixelSize: 12
                color: "#888888"
                Layout.alignment: Qt.AlignVCenter
            }

            Item { Layout.fillWidth: true }

            // Idle status indicator
            RowLayout {
                spacing: 8
                Layout.alignment: Qt.AlignVCenter

                Rectangle {
                    width: 8
                    height: 8
                    radius: 4
                    color: idleScheduler.isSchedulerActive ? "#4ec9b0" : "#858585"

                    SequentialAnimation on opacity {
                        running: idleScheduler.isSchedulerActive
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.4; duration: 800; easing.type: Easing.InOutQuad }
                        NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutQuad }
                    }
                }

                Label {
                    text: "Idle: " + (idleScheduler.isSchedulerActive ? "Active" : "Standby")
                    font.pixelSize: 11
                    color: "#aaaaaa"
                }
            }

            // Training status (Phase 2)
            RowLayout {
                spacing: 6
                Layout.alignment: Qt.AlignVCenter
                visible: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: "#dcdcaa"

                    SequentialAnimation on opacity {
                        running: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 500; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 500; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: typeof reflectionEngine !== "undefined"
                          ? "Training: " + reflectionEngine.phase
                          : ""
                    font.pixelSize: 11
                    color: "#dcdcaa"
                }
            }

            // Adapter version (Phase 2)
            Label {
                text: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
                      ? "v" + adapterManager.activeVersion
                      : ""
                font.pixelSize: 10
                color: "#4ec9b0"
                Layout.alignment: Qt.AlignVCenter
                visible: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
            }

            // Separator
            Rectangle {
                width: 1
                height: 24
                color: "#444444"
                Layout.alignment: Qt.AlignVCenter
            }

            // Memory stats
            Label {
                text: "Episodes: " + memoryStore.episodeCount + " | Traits: " + memoryStore.traitCount
                font.pixelSize: 11
                color: "#888888"
                Layout.alignment: Qt.AlignVCenter
            }

            // System clock
            Label {
                id: systemClock
                font.pixelSize: 11
                color: "#569cd6"
                Layout.alignment: Qt.AlignVCenter
                Timer {
                    interval: 1000
                    running: true
                    repeat: true
                    triggeredOnStart: true
                    onTriggered: {
                        var now = new Date()
                        systemClock.text = Qt.formatDateTime(now, "ddd MMM d, yyyy  h:mm AP")
                    }
                }
            }

            // Thought notification badge
            RowLayout {
                spacing: 6
                Layout.alignment: Qt.AlignVCenter
                visible: typeof thoughtEngine !== "undefined" && thoughtEngine.pendingCount > 0

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: "#569cd6"

                    SequentialAnimation on opacity {
                        running: typeof thoughtEngine !== "undefined" && thoughtEngine.pendingCount > 0
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 1000; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 1000; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: {
                        if (typeof thoughtEngine === "undefined") return ""
                        var n = thoughtEngine.pendingCount
                        return n + " thought" + (n !== 1 ? "s" : "")
                    }
                    font.pixelSize: 11
                    color: "#569cd6"
                    MouseArea {
                        anchors.fill: parent
                        cursorShape: Qt.PointingHandCursor
                        onClicked: {
                            if (typeof thoughtEngine === "undefined") return
                            var top = thoughtEngine.getTopThoughtForQml()
                            if (top.id !== undefined) {
                                orchestrator.startProactiveConversation(top.id)
                            }
                        }
                    }
                }
            }

            // Run eval button
            Button {
                text: "Run Eval"
                flat: true
                font.pixelSize: 11
                Layout.alignment: Qt.AlignVCenter

                ToolTip.text: "Run evaluation suite"
                ToolTip.visible: hovered
                ToolTip.delay: 500

                contentItem: Text {
                    text: parent.text
                    color: parent.hovered ? "#4ec9b0" : "#888888"
                    font: parent.font
                }
                background: Rectangle {
                    color: parent.hovered ? "#2d2d30" : "transparent"
                    radius: 4
                }

                onClicked: {
                    evalSuite.runFullSuite()
                }
            }

            // Lineage button (visible when lineageTracker exists)
            Button {
                text: "Lineage"
                flat: true
                font.pixelSize: 13
                Layout.alignment: Qt.AlignVCenter
                visible: typeof lineageTracker !== "undefined"

                ToolTip.text: "View adapter lineage"
                ToolTip.visible: hovered
                ToolTip.delay: 500

                contentItem: Text {
                    text: parent.text
                    color: currentView === 2 ? "#569cd6"
                         : parent.hovered ? "#ffffff" : "#cccccc"
                    font: parent.font
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                background: Rectangle {
                    color: currentView === 2 ? "#1e3040"
                         : parent.hovered ? "#3e3e42" : "transparent"
                    radius: 4
                }

                onClicked: {
                    currentView = currentView === 2 ? 0 : 2
                }
            }

            // Separator
            Rectangle {
                width: 1
                height: 24
                color: "#444444"
                Layout.alignment: Qt.AlignVCenter
            }

            // Settings button
            Button {
                text: currentView === 1 ? "< Dashboard"
                    : currentView === 2 ? "< Dashboard"
                    : "Settings"
                flat: true
                font.pixelSize: 13
                Layout.alignment: Qt.AlignVCenter

                ToolTip.text: currentView === 0 ? "Open settings" : "Back to dashboard"
                ToolTip.visible: hovered
                ToolTip.delay: 500

                contentItem: Text {
                    text: parent.text
                    color: parent.hovered ? "#ffffff" : "#cccccc"
                    font: parent.font
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }

                background: Rectangle {
                    color: parent.hovered ? "#3e3e42" : "transparent"
                    radius: 4
                }

                onClicked: {
                    currentView = currentView === 0 ? 1 : 0
                }
                visible: currentView !== 2
            }
        }
    }

    // Main content area
    StackLayout {
        id: mainStack
        anchors.top: header.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        currentIndex: currentView

        // Dashboard view (index 0)
        Item {
            RowLayout {
                anchors.fill: parent
                anchors.margins: 8
                spacing: 8

                // Left: Conversation sidebar (collapsible)
                ConversationSidebar {
                    id: conversationSidebar
                    Layout.fillHeight: true
                }

                // Center: Chat window (fills remaining)
                ChatWindow {
                    id: chatWindow
                    Layout.fillHeight: true
                    Layout.fillWidth: true
                    Layout.minimumWidth: 300
                }

                // Right: Idle + Memory panels (28%)
                ColumnLayout {
                    Layout.fillHeight: true
                    Layout.preferredWidth: parent.width * 0.28
                    Layout.minimumWidth: 310
                    Layout.maximumWidth: 500
                    spacing: 8

                    // Idle Mind panel (40% of right side, scrollable)
                    IdlePanel {
                        Layout.fillWidth: true
                        Layout.preferredHeight: parent.height * 0.40
                    }

                    // Memory panel (70% of right side)
                    MemoryPanel {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                    }
                }
            }
        }

        // Settings view (index 1)
        SettingsPanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }

        // Lineage view (index 2)
        LineagePanel {
            Layout.fillWidth: true
            Layout.fillHeight: true
        }
    }
}
