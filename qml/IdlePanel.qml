import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: idleRoot
    color: "#252526"
    radius: 6

    // Fixed header outside scrollable area
    RowLayout {
        id: idleHeader
        anchors.top: parent.top
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.margins: 15
        spacing: 8

        Label {
            text: "Idle Mind"
            font.pixelSize: 16
            font.bold: true
            color: "#ffffff"
        }

        Item { Layout.fillWidth: true }

        // Pulsing status indicator
        Rectangle {
            width: 10
            height: 10
            radius: 5
            color: idleScheduler.isSchedulerActive ? "#4ec9b0" : "#858585"

            SequentialAnimation on opacity {
                running: idleScheduler.isSchedulerActive
                loops: Animation.Infinite
                NumberAnimation { to: 0.3; duration: 1000; easing.type: Easing.InOutSine }
                NumberAnimation { to: 1.0; duration: 1000; easing.type: Easing.InOutSine }
            }
        }
    }

    // Separator under header
    Rectangle {
        id: headerSep
        anchors.top: idleHeader.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.topMargin: 8
        anchors.leftMargin: 15
        anchors.rightMargin: 15
        height: 1
        color: "#3e3e42"
    }

    // Scrollable content area
    Flickable {
        anchors.top: headerSep.bottom
        anchors.left: parent.left
        anchors.right: parent.right
        anchors.bottom: parent.bottom
        anchors.margins: 15
        anchors.topMargin: 8
        contentWidth: width
        contentHeight: idleContent.height
        clip: true
        boundsBehavior: Flickable.StopAtBounds

        ScrollBar.vertical: ScrollBar {
            policy: ScrollBar.AsNeeded
        }

        ColumnLayout {
            id: idleContent
            width: parent.width
            spacing: 10

        // Status message
        Label {
            text: idleScheduler.schedulerStatus
            font.pixelSize: 13
            color: idleScheduler.isSchedulerActive ? "#4ec9b0" : "#aaaaaa"
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        // Metrics grid
        GridLayout {
            columns: 2
            columnSpacing: 15
            rowSpacing: 6
            Layout.fillWidth: true

            Label { text: "Idle time:"; color: "#888888"; font.pixelSize: 12 }
            Label {
                text: {
                    var secs = idleScheduler.idleTimeSeconds
                    if (secs < 60) return secs + "s"
                    if (secs < 3600) return Math.floor(secs / 60) + "m " + (secs % 60) + "s"
                    var h = Math.floor(secs / 3600)
                    var m = Math.floor((secs % 3600) / 60)
                    return h + "h " + m + "m"
                }
                color: "#cccccc"
                font.pixelSize: 12
            }

            Label { text: "Power:"; color: "#888888"; font.pixelSize: 12 }
            Label {
                text: idleScheduler.isPluggedIn ? "AC Connected" : "Battery"
                color: idleScheduler.isPluggedIn ? "#4ec9b0" : "#ffa500"
                font.pixelSize: 12
            }

            Label { text: "Budget:"; color: "#888888"; font.pixelSize: 12 }
            Label {
                text: idleScheduler.computeBudgetPercent + "%"
                color: "#cccccc"
                font.pixelSize: 12
            }

            Label {
                text: "Enabled:"
                color: "#888888"
                font.pixelSize: 12
            }
            Label {
                text: idleScheduler.enabled ? "Yes" : "No"
                color: idleScheduler.enabled ? "#4ec9b0" : "#ff6666"
                font.pixelSize: 12
            }

            Label { text: "Curiosity:"; color: "#888888"; font.pixelSize: 12 }
            Label {
                text: (orchestrator.curiosityLevel * 100).toFixed(0) + "%"
                color: "#cccccc"
                font.pixelSize: 12
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // === Training Progress (Phase 2) ===
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            visible: typeof reflectionEngine !== "undefined"

            RowLayout {
                spacing: 6
                Layout.fillWidth: true

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                           ? "#dcdcaa" : "#3e3e42"

                    SequentialAnimation on opacity {
                        running: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 600; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 600; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                          ? "Training: " + reflectionEngine.phase
                          : "Training: Idle"
                    font.pixelSize: 12
                    font.bold: true
                    color: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                           ? "#dcdcaa" : "#888888"
                }
            }

            // Progress bar (visible during training)
            Rectangle {
                Layout.fillWidth: true
                height: 4
                radius: 2
                color: "#1e1e1e"
                visible: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                         && reflectionEngine.totalTrainingSteps > 0

                Rectangle {
                    width: {
                        if (typeof reflectionEngine === "undefined") return 0
                        var total = reflectionEngine.totalTrainingSteps
                        if (total <= 0) return 0
                        return parent.width * (reflectionEngine.trainingStep / total)
                    }
                    height: parent.height
                    radius: 2
                    color: "#dcdcaa"

                    Behavior on width { NumberAnimation { duration: 300 } }
                }
            }

            // Training stats (visible during training)
            RowLayout {
                spacing: 12
                visible: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting

                Label {
                    text: typeof reflectionEngine !== "undefined"
                          ? "Step " + reflectionEngine.trainingStep + "/" + reflectionEngine.totalTrainingSteps
                          : ""
                    font.pixelSize: 11
                    color: "#aaaaaa"
                }
                Label {
                    text: typeof reflectionEngine !== "undefined" && reflectionEngine.trainingLoss > 0
                          ? "Loss: " + reflectionEngine.trainingLoss.toFixed(4)
                          : ""
                    font.pixelSize: 11
                    color: "#aaaaaa"
                }
            }

            // Status message
            Label {
                text: typeof reflectionEngine !== "undefined" ? reflectionEngine.reflectionStatus : ""
                font.pixelSize: 11
                color: "#888888"
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: typeof reflectionEngine !== "undefined"
            }

            // Adapter info + sessions
            RowLayout {
                spacing: 12
                Layout.fillWidth: true

                Label {
                    text: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
                          ? "Adapter v" + adapterManager.activeVersion
                          : "Base model"
                    font.pixelSize: 11
                    color: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
                           ? "#4ec9b0" : "#666666"
                }
                Label {
                    text: typeof reflectionEngine !== "undefined"
                          ? "Sessions: " + reflectionEngine.sessionsCompleted
                          : ""
                    font.pixelSize: 11
                    color: "#666666"
                }
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
            visible: typeof reflectionEngine !== "undefined"
        }

        // === Evolution Progress (Phase 3) ===
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6
            visible: typeof evolutionEngine !== "undefined"

            RowLayout {
                spacing: 6
                Layout.fillWidth: true

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving
                           ? "#ce9178" : "#3e3e42"

                    SequentialAnimation on opacity {
                        running: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 800; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving
                          ? "Evolution: " + evolutionEngine.evolutionStage
                          : "Evolution: Idle"
                    font.pixelSize: 12
                    font.bold: true
                    color: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving
                           ? "#ce9178" : "#888888"
                }
            }

            // Variant progress bar
            Rectangle {
                Layout.fillWidth: true
                height: 4
                radius: 2
                color: "#1e1e1e"
                visible: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving
                         && evolutionEngine.totalVariants > 0

                Rectangle {
                    width: {
                        if (typeof evolutionEngine === "undefined") return 0
                        var total = evolutionEngine.totalVariants
                        if (total <= 0) return 0
                        var idx = evolutionEngine.currentVariantIndex
                        // During training, show partial progress for the current variant
                        var variantProgress = 0
                        if (evolutionEngine.totalTrainingSteps > 0) {
                            variantProgress = evolutionEngine.trainingStep / evolutionEngine.totalTrainingSteps
                        }
                        return parent.width * ((idx + variantProgress) / total)
                    }
                    height: parent.height
                    radius: 2
                    color: "#ce9178"

                    Behavior on width { NumberAnimation { duration: 300 } }
                }
            }

            // Evolution stats
            GridLayout {
                columns: 2
                columnSpacing: 12
                rowSpacing: 4
                visible: typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving

                Label {
                    text: "Variant:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Label {
                    text: typeof evolutionEngine !== "undefined"
                          ? (evolutionEngine.currentVariantIndex + 1) + " / " + evolutionEngine.totalVariants
                          : ""
                    font.pixelSize: 11
                    color: "#ce9178"
                }

                Label {
                    text: "Step:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Label {
                    text: typeof evolutionEngine !== "undefined" && evolutionEngine.totalTrainingSteps > 0
                          ? evolutionEngine.trainingStep + " / " + evolutionEngine.totalTrainingSteps
                          : "--"
                    font.pixelSize: 11
                    color: "#aaaaaa"
                }

                Label {
                    text: "Loss:"
                    font.pixelSize: 11
                    color: "#888888"
                }
                Label {
                    text: typeof evolutionEngine !== "undefined" && evolutionEngine.trainingLoss > 0
                          ? evolutionEngine.trainingLoss.toFixed(4)
                          : "--"
                    font.pixelSize: 11
                    color: "#aaaaaa"
                }
            }

            // Cycle / consolidation counters
            RowLayout {
                spacing: 12
                Layout.fillWidth: true

                Label {
                    text: typeof evolutionEngine !== "undefined"
                          ? "Cycles: " + evolutionEngine.cyclesCompleted
                          : ""
                    font.pixelSize: 11
                    color: "#666666"
                }
                Label {
                    text: typeof evolutionEngine !== "undefined" && evolutionEngine.consolidationsCompleted > 0
                          ? "Consolidations: " + evolutionEngine.consolidationsCompleted
                          : ""
                    font.pixelSize: 11
                    color: "#666666"
                }
                Label {
                    text: typeof evolutionEngine !== "undefined"
                          ? "Pop: " + evolutionEngine.populationSize
                          : ""
                    font.pixelSize: 11
                    color: "#666666"
                }
            }

            // Status message
            Label {
                text: typeof evolutionEngine !== "undefined" ? evolutionEngine.evolutionStatus : ""
                font.pixelSize: 11
                color: "#888888"
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: typeof evolutionEngine !== "undefined"
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
            visible: typeof evolutionEngine !== "undefined"
        }

        // === Research Activity (Phase 5) ===
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            RowLayout {
                spacing: 6
                Layout.fillWidth: true

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: researchEngine.isResearching ? "#569cd6" : "#3e3e42"

                    SequentialAnimation on opacity {
                        running: researchEngine.isResearching
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 700; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 700; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: researchEngine.isResearching
                          ? "Research: " + researchEngine.currentPhase
                          : "Research: Idle"
                    font.pixelSize: 12
                    font.bold: true
                    color: researchEngine.isResearching ? "#569cd6" : "#888888"
                }
            }

            // Current topic
            Label {
                text: researchEngine.isResearching && researchEngine.currentTopic
                      ? "Topic: " + researchEngine.currentTopic
                      : ""
                font.pixelSize: 11
                color: "#aaaaaa"
                Layout.fillWidth: true
                elide: Text.ElideRight
                visible: researchEngine.isResearching
            }

            // Counts
            RowLayout {
                spacing: 12
                Layout.fillWidth: true

                Label {
                    text: "Pending: " + researchStore.pendingCount
                    font.pixelSize: 11
                    color: researchStore.pendingCount > 0 ? "#569cd6" : "#666666"
                }
                Label {
                    text: "Approved: " + researchStore.approvedCount
                    font.pixelSize: 11
                    color: "#666666"
                }
                Label {
                    text: "Today: " + researchEngine.cyclesCompletedToday
                    font.pixelSize: 11
                    color: "#666666"
                }
                Label {
                    text: "Queue: " + researchEngine.topicQueueSize
                    font.pixelSize: 11
                    color: "#666666"
                }
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // === Distillation Activity (Phase 8) ===
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 6

            RowLayout {
                spacing: 6
                Layout.fillWidth: true

                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: distillationManager.isDistilling ? "#c586c0" : "#3e3e42"

                    SequentialAnimation on opacity {
                        running: distillationManager.isDistilling
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.3; duration: 800; easing.type: Easing.InOutSine }
                        NumberAnimation { to: 1.0; duration: 800; easing.type: Easing.InOutSine }
                    }
                }

                Label {
                    text: distillationManager.isDistilling
                          ? "Distillation: " + distillationManager.currentPhase
                          : "Distillation: Idle"
                    font.pixelSize: 12
                    font.bold: true
                    color: distillationManager.isDistilling ? "#c586c0" : "#888888"
                }
            }

            // Stats row
            RowLayout {
                spacing: 12
                Layout.fillWidth: true

                Label {
                    text: "Pairs: " + distillationManager.pairsCollected
                    font.pixelSize: 11
                    color: distillationManager.pairsCollected > 0 ? "#c586c0" : "#666666"
                }
                Label {
                    text: "Readiness: " + (distillationManager.readinessScore * 100).toFixed(0) + "%"
                    font.pixelSize: 11
                    color: distillationManager.readinessScore >= 0.75 ? "#4ec9b0"
                         : distillationManager.readinessScore >= 0.5 ? "#c586c0" : "#666666"
                }
            }

            // Status message
            Label {
                text: distillationManager.statusMessage
                font.pixelSize: 11
                color: "#888888"
                Layout.fillWidth: true
                wrapMode: Text.WordWrap
                visible: distillationManager.statusMessage !== ""
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // Thought generation status
        RowLayout {
            spacing: 6
            Layout.fillWidth: true
            visible: typeof thoughtEngine !== "undefined"

            Rectangle {
                width: 6; height: 6; radius: 3
                color: typeof thoughtEngine !== "undefined" && thoughtEngine.isGenerating
                       ? "#569cd6" : "#3e3e42"
                Layout.alignment: Qt.AlignVCenter
            }

            Label {
                text: {
                    if (typeof thoughtEngine === "undefined") return ""
                    if (thoughtEngine.isGenerating) return "Thinking..."
                    if (thoughtEngine.pendingCount > 0)
                        return "Pending thoughts: " + thoughtEngine.pendingCount
                    return "No pending thoughts"
                }
                font.pixelSize: 11
                color: typeof thoughtEngine !== "undefined" && thoughtEngine.isGenerating
                       ? "#569cd6" : "#666666"
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
            visible: typeof thoughtEngine !== "undefined"
        }

        // Eval results
        Label {
            text: "Eval: " + (evalSuite.lastReportSummary || "Not run yet")
            font.pixelSize: 11
            color: evalSuite.lastScore >= 0.7 ? "#4ec9b0"
                 : evalSuite.lastScore >= 0.4 ? "#ffa500" : "#888888"
            Layout.fillWidth: true
            wrapMode: Text.WordWrap
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // === Phase 4: Disk usage + health ===
        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: "Disk: " + dataLifecycleManager.diskUsageFormatted
                font.pixelSize: 11
                color: "#888888"
            }

            Rectangle {
                width: 6; height: 6; radius: 3
                color: profileHealthManager.isHealthy ? "#4ec9b0" : "#ffa500"
                Layout.alignment: Qt.AlignVCenter
            }

            Label {
                text: profileHealthManager.healthStatus
                font.pixelSize: 11
                color: profileHealthManager.isHealthy ? "#4ec9b0" : "#ffa500"
            }
        }

        Item { Layout.fillHeight: true }
        }
    }
}
