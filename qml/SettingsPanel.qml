import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

Rectangle {
    id: settingsRoot
    color: "#1e1e1e"

    property int labelWidth: 130

    ColumnLayout {
        anchors.fill: parent
        spacing: 0

        // Title
        Label {
            text: "Settings"
            font.pixelSize: 24
            font.bold: true
            color: "#ffffff"
            Layout.leftMargin: 30
            Layout.topMargin: 20
            Layout.bottomMargin: 12
        }

        // Tab bar
        TabBar {
            id: settingsTabBar
            Layout.fillWidth: true
            Layout.leftMargin: 20
            Layout.rightMargin: 20

            background: Rectangle { color: "transparent" }

            TabButton {
                text: "Models"
                width: implicitWidth
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 13
                    font.bold: settingsTabBar.currentIndex === 0
                    color: settingsTabBar.currentIndex === 0 ? "#4ec9b0" : "#888888"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: settingsTabBar.currentIndex === 0 ? "#1e1e1e" : "#2d2d30"
                    radius: 6
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: 2
                        color: "#4ec9b0"
                        visible: settingsTabBar.currentIndex === 0
                    }
                }
            }

            TabButton {
                text: "Behavior"
                width: implicitWidth
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 13
                    font.bold: settingsTabBar.currentIndex === 1
                    color: settingsTabBar.currentIndex === 1 ? "#569cd6" : "#888888"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: settingsTabBar.currentIndex === 1 ? "#1e1e1e" : "#2d2d30"
                    radius: 6
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: 2
                        color: "#569cd6"
                        visible: settingsTabBar.currentIndex === 1
                    }
                }
            }

            TabButton {
                text: "Training"
                width: implicitWidth
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 13
                    font.bold: settingsTabBar.currentIndex === 2
                    color: settingsTabBar.currentIndex === 2 ? "#dcdcaa" : "#888888"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: settingsTabBar.currentIndex === 2 ? "#1e1e1e" : "#2d2d30"
                    radius: 6
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: 2
                        color: "#dcdcaa"
                        visible: settingsTabBar.currentIndex === 2
                    }
                }
            }

            TabButton {
                text: "Data"
                width: implicitWidth
                contentItem: Text {
                    text: parent.text
                    font.pixelSize: 13
                    font.bold: settingsTabBar.currentIndex === 3
                    color: settingsTabBar.currentIndex === 3 ? "#ce9178" : "#888888"
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: settingsTabBar.currentIndex === 3 ? "#1e1e1e" : "#2d2d30"
                    radius: 6
                    Rectangle {
                        anchors.bottom: parent.bottom
                        anchors.left: parent.left
                        anchors.right: parent.right
                        height: 2
                        color: "#ce9178"
                        visible: settingsTabBar.currentIndex === 3
                    }
                }
            }
        }

        // Divider under tabs
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // Tab content
        StackLayout {
            id: settingsStack
            currentIndex: settingsTabBar.currentIndex
            Layout.fillWidth: true
            Layout.fillHeight: true

            // ============================================================
            // TAB 1: MODELS
            // ============================================================
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: modelsColumn.height + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                ColumnLayout {
                    id: modelsColumn
                    anchors.top: parent.top
                    anchors.topMargin: 20
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.min(parent.width - 40, 640)
                    spacing: 20

                    // === User Section ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: userSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: userSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "User"; font.pixelSize: 16; font.bold: true; color: "#4ec9b0" }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Name:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.userName
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    placeholderText: "Your name"
                                    placeholderTextColor: "#666666"
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onTextChanged: settingsManager.userName = text
                                }
                            }
                        }
                    }

                    // === LLM Configuration ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: llmSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: llmSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "LLM Configuration"; font.pixelSize: 16; font.bold: true; color: "#4ec9b0" }

                            // Provider
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Provider:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                ComboBox {
                                    id: providerCombo
                                    Layout.fillWidth: true
                                    model: ["Ollama", "LM Studio", "OpenAI", "Anthropic", "Local"]
                                    currentIndex: model.indexOf(settingsManager.provider)
                                    onCurrentTextChanged: {
                                        if (currentText !== settingsManager.provider) {
                                            settingsManager.provider = currentText
                                        }
                                    }
                                }
                            }

                            // Model
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Model:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                ComboBox {
                                    id: modelCombo
                                    Layout.fillWidth: true
                                    model: llmProvider.availableModels
                                    currentIndex: Math.max(0, llmProvider.availableModels.indexOf(settingsManager.model))
                                    onCurrentTextChanged: {
                                        if (currentText !== settingsManager.model) {
                                            settingsManager.model = currentText
                                        }
                                    }
                                }
                                Button {
                                    text: "Refresh"
                                    visible: settingsManager.provider === "Ollama" || settingsManager.provider === "LM Studio"
                                    onClicked: llmProvider.refreshModels()
                                    contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                                    background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                                }
                            }

                            // Connection test
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Status:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }

                                Rectangle {
                                    width: 10; height: 10; radius: 5
                                    color: llmProvider.isConnected ? "#4ec9b0" : "#858585"
                                    Layout.alignment: Qt.AlignVCenter
                                }

                                Label {
                                    text: llmProvider.connectionStatus
                                    color: "#aaaaaa"
                                    font.pixelSize: 13
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: "Test Connection"
                                    onClicked: llmProvider.testConnection()
                                    contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                                    background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                                }
                            }

                            // Model context length
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Context Length:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: contextLengthSlider
                                    Layout.fillWidth: true
                                    from: 2048; to: 131072; stepSize: 1024
                                    value: settingsManager.modelContextLength
                                    onMoved: contextDebounce.restart()
                                }
                                Label {
                                    text: {
                                        var v = contextLengthSlider.value
                                        return v >= 1024 ? Math.round(v / 1024) + "K" : v
                                    }
                                    color: "#aaaaaa"; font.pixelSize: 13
                                    Layout.preferredWidth: 40
                                }
                            }
                            Timer {
                                id: contextDebounce
                                interval: 300
                                onTriggered: settingsManager.modelContextLength = contextLengthSlider.value
                            }

                            // API Keys (conditional)
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: settingsManager.provider === "OpenAI"
                                Label { text: "OpenAI Key:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.openAIKey
                                    echoMode: TextInput.Password
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    placeholderText: "sk-..."
                                    placeholderTextColor: "#666666"
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onTextChanged: settingsManager.openAIKey = text
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: settingsManager.provider === "Anthropic"
                                Label { text: "Anthropic Key:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.anthropicKey
                                    echoMode: TextInput.Password
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    placeholderText: "sk-ant-..."
                                    placeholderTextColor: "#666666"
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onTextChanged: settingsManager.anthropicKey = text
                                }
                            }

                            // Custom URLs
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: settingsManager.provider === "LM Studio"
                                Label { text: "LM Studio URL:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.lmStudioUrl
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onTextChanged: settingsManager.lmStudioUrl = text
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: settingsManager.provider === "Ollama"
                                Label { text: "Ollama URL:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.ollamaUrl
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onTextChanged: settingsManager.ollamaUrl = text
                                }
                            }

                            // Local model info (Phase 6)
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                visible: settingsManager.provider === "Local"

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 10
                                    Label { text: "Model Status:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                    Rectangle {
                                        width: 10; height: 10; radius: 5
                                        color: typeof ortInferenceManager !== "undefined" && ortInferenceManager.isModelLoaded
                                               ? "#4ec9b0" : "#858585"
                                        Layout.alignment: Qt.AlignVCenter
                                    }
                                    Label {
                                        text: {
                                            if (typeof ortInferenceManager === "undefined") return "Training support not available"
                                            if (ortInferenceManager.isModelLoaded) {
                                                var ver = (typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0)
                                                          ? " v" + adapterManager.activeVersion : ""
                                                return "Loaded: personal-model" + ver
                                            }
                                            return "No model loaded — train one first"
                                        }
                                        color: typeof ortInferenceManager !== "undefined" && ortInferenceManager.isModelLoaded
                                               ? "#4ec9b0" : "#ffa500"
                                        font.pixelSize: 13
                                    }
                                    Item { Layout.fillWidth: true }
                                }

                                Label {
                                    text: typeof ortInferenceManager !== "undefined" && ortInferenceManager.modelPath
                                          ? ortInferenceManager.modelPath : ""
                                    font.pixelSize: 11
                                    color: "#666666"
                                    Layout.fillWidth: true
                                    elide: Text.ElideMiddle
                                    visible: typeof ortInferenceManager !== "undefined" && ortInferenceManager.isModelLoaded
                                }

                                Label {
                                    text: "The local model learns from your conversations, research, and corrections."
                                    font.pixelSize: 11
                                    color: "#888888"
                                    Layout.fillWidth: true
                                    wrapMode: Text.WordWrap
                                }
                            }
                        }
                    }

                    // === Background Model (llama.cpp) ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: bgModelSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: bgModelSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Background Model"; font.pixelSize: 16; font.bold: true; color: "#4ec9b0" }

                            Label {
                                text: "Embedded model for background tasks (trait extraction, research, goals, sentiment). Runs independently of the chat provider."
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Switch {
                                    id: bgModelSwitch
                                    checked: settingsManager.backgroundModelEnabled
                                    onToggled: settingsManager.backgroundModelEnabled = checked
                                }
                                Label {
                                    text: "Enable embedded background model"
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                }
                            }

                            // Model selector dropdown
                            ColumnLayout {
                                Layout.fillWidth: true
                                spacing: 6
                                visible: bgModelSwitch.checked

                                Label {
                                    text: "Select Model"
                                    color: "#aaaaaa"
                                    font.pixelSize: 12
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    ComboBox {
                                        id: bgModelCombo
                                        Layout.fillWidth: true
                                        model: {
                                            if (typeof llamaCppManager === "undefined") return ["(llama.cpp not available)"]
                                            var models = llamaCppManager.availableModels()
                                            if (models.length === 0) return ["(no .gguf files found)"]
                                            return models
                                        }
                                        currentIndex: {
                                            if (typeof llamaCppManager === "undefined") return 0
                                            var loaded = llamaCppManager.loadedModelName()
                                            var models = llamaCppManager.availableModels()
                                            var idx = models.indexOf(loaded)
                                            return idx >= 0 ? idx : 0
                                        }
                                        onActivated: function(index) {
                                            if (typeof llamaCppManager === "undefined") return
                                            var models = llamaCppManager.availableModels()
                                            if (index < 0 || index >= models.length) return
                                            var selected = "models/background_llm/" + models[index]
                                            settingsManager.backgroundModelPath = selected
                                        }
                                        background: Rectangle { color: "#333333"; radius: 4 }
                                        contentItem: Label {
                                            text: bgModelCombo.displayText
                                            color: "#ffffff"
                                            font.pixelSize: 13
                                            verticalAlignment: Text.AlignVCenter
                                            leftPadding: 8
                                        }
                                        popup.background: Rectangle { color: "#2d2d2d"; border.color: "#555555"; radius: 4 }
                                        delegate: ItemDelegate {
                                            width: bgModelCombo.width
                                            contentItem: Label {
                                                text: modelData
                                                color: highlighted ? "#ffffff" : "#cccccc"
                                                font.pixelSize: 13
                                            }
                                            highlighted: bgModelCombo.highlightedIndex === index
                                            background: Rectangle {
                                                color: highlighted ? "#4ec9b0" : "transparent"
                                            }
                                        }
                                    }

                                    Button {
                                        text: "Refresh"
                                        onClicked: {
                                            // Force re-evaluation of model list
                                            bgModelCombo.model = []
                                            if (typeof llamaCppManager !== "undefined") {
                                                var models = llamaCppManager.availableModels()
                                                bgModelCombo.model = models.length > 0 ? models : ["(no .gguf files found)"]
                                                var loaded = llamaCppManager.loadedModelName()
                                                var idx = models.indexOf(loaded)
                                                if (idx >= 0) bgModelCombo.currentIndex = idx
                                            }
                                        }
                                    }

                                    Button {
                                        text: "Browse..."
                                        onClicked: ggufFileDialog.open()
                                    }
                                }
                            }

                            RowLayout {
                                visible: bgModelSwitch.checked
                                spacing: 8
                                Rectangle {
                                    width: 10; height: 10; radius: 5
                                    color: typeof llamaCppManager !== "undefined" && llamaCppManager.isModelLoaded ? "#4CAF50" : "#F44336"
                                }
                                Label {
                                    text: {
                                        if (typeof llamaCppManager === "undefined") return "llama.cpp not available"
                                        if (llamaCppManager.isModelLoaded)
                                            return "Loaded: " + llamaCppManager.loadedModelName()
                                        return "No model loaded — background tasks use chat provider"
                                    }
                                    color: "#aaaaaa"
                                    font.pixelSize: 11
                                }
                            }

                            Label {
                                visible: bgModelSwitch.checked
                                text: "Place .gguf files in models/background_llm/ and click Refresh. Recommended: 3B-8B Q8_0 models."
                                font.pixelSize: 11
                                color: "#666666"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }
                        }
                    }

                    // Footer
                    Label {
                        text: "DATAFORM v0.6.0 | Global Human Initiative"
                        font.pixelSize: 11
                        color: "#555555"
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                        Layout.bottomMargin: 20
                    }
                }
            }

            // ============================================================
            // TAB 2: BEHAVIOR
            // ============================================================
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: behaviorColumn.height + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                ColumnLayout {
                    id: behaviorColumn
                    anchors.top: parent.top
                    anchors.topMargin: 20
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.min(parent.width - 40, 640)
                    spacing: 20

                    // === Research Settings ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: researchSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: researchSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Research"; font.pixelSize: 16; font.bold: true; color: "#569cd6" }

                            // Research toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Idle Research:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.researchEnabled
                                    onToggled: settingsManager.researchEnabled = checked
                                }
                                Label {
                                    text: settingsManager.researchEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Max daily cycles
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Daily Cycles:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: researchSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 20
                                    stepSize: 1
                                    value: settingsManager.maxResearchPerDay
                                    onPressedChanged: if (!pressed) settingsManager.maxResearchPerDay = value
                                }
                                Label {
                                    text: Math.round(researchSlider.value)
                                    color: "#569cd6"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 30
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // Privacy note
                            Label {
                                text: {
                                    if (settingsManager.privacyLevel === "minimal")
                                        return "Research disabled in minimal privacy mode"
                                    if (settingsManager.privacyLevel === "standard")
                                        return "Findings require your approval before use"
                                    return "High-relevance findings auto-approved"
                                }
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Stats
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: "Pending: " + researchStore.pendingCount
                                    font.pixelSize: 12
                                    color: "#888888"
                                }
                                Label {
                                    text: "Approved: " + researchStore.approvedCount
                                    font.pixelSize: 12
                                    color: "#888888"
                                }
                                Label {
                                    text: "Total: " + researchStore.totalCount
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // === News Settings ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: newsSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: newsSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "News"; font.pixelSize: 16; font.bold: true; color: "#569cd6" }

                            // News toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "News Headlines:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.newsEnabled
                                    onToggled: settingsManager.newsEnabled = checked
                                }
                                Label {
                                    text: settingsManager.newsEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Max news per day slider
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Daily Limit:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: newsSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 10
                                    stepSize: 1
                                    value: settingsManager.maxNewsPerDay
                                    onPressedChanged: if (!pressed) settingsManager.maxNewsPerDay = value
                                }
                                Label {
                                    text: Math.round(newsSlider.value)
                                    color: "#569cd6"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 30
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // RSS Feed management
                            Label { text: "RSS Feeds:"; color: "#cccccc"; font.pixelSize: 13 }

                            Repeater {
                                model: settingsManager.newsFeeds

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 6

                                    Label {
                                        text: modelData
                                        color: "#d4d4d4"
                                        font.pixelSize: 12
                                        elide: Text.ElideMiddle
                                        Layout.fillWidth: true
                                    }

                                    Button {
                                        text: "\u00D7"
                                        flat: true
                                        implicitWidth: 24
                                        implicitHeight: 24
                                        contentItem: Text {
                                            text: parent.text
                                            color: parent.hovered ? "#d4534b" : "#888888"
                                            font.pixelSize: 14
                                            horizontalAlignment: Text.AlignHCenter
                                            verticalAlignment: Text.AlignVCenter
                                        }
                                        background: Rectangle {
                                            color: parent.hovered ? "#3e3e42" : "transparent"
                                            radius: 4
                                        }
                                        onClicked: settingsManager.removeNewsFeed(index)
                                    }
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                TextField {
                                    id: newFeedField
                                    Layout.fillWidth: true
                                    placeholderText: "https://example.com/rss"
                                    placeholderTextColor: "#666666"
                                    color: "#d4d4d4"
                                    font.pixelSize: 12
                                    background: Rectangle {
                                        color: "#1e1e1e"
                                        radius: 4
                                        border.color: newFeedField.activeFocus ? "#569cd6" : "#3e3e42"
                                        border.width: 1
                                    }

                                    Keys.onReturnPressed: {
                                        if (newFeedField.text.trim().length > 0) {
                                            settingsManager.addNewsFeed(newFeedField.text.trim())
                                            newFeedField.text = ""
                                        }
                                    }
                                }

                                Button {
                                    text: "Add"
                                    enabled: newFeedField.text.trim().length > 0
                                    implicitHeight: 30
                                    implicitWidth: 50
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#ffffff" : "#666666"
                                        font.pixelSize: 12
                                        horizontalAlignment: Text.AlignHCenter
                                        verticalAlignment: Text.AlignVCenter
                                    }
                                    background: Rectangle {
                                        color: parent.enabled ? "#569cd6" : "#3e3e42"
                                        radius: 4
                                    }
                                    onClicked: {
                                        settingsManager.addNewsFeed(newFeedField.text.trim())
                                        newFeedField.text = ""
                                    }
                                }
                            }

                            // Privacy note
                            Label {
                                text: {
                                    if (settingsManager.privacyLevel === "minimal")
                                        return "News disabled in minimal privacy mode"
                                    var count = settingsManager.newsFeeds.length
                                    return "DATAFORM fetches from " + count + " feed(s) during idle time, cycling round-robin"
                                }
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Stats
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12
                                visible: typeof newsEngine !== "undefined"

                                Label {
                                    text: "Today: " + (typeof newsEngine !== "undefined" ? newsEngine.cyclesCompletedToday : 0) + " cycles"
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Label {
                                    text: typeof newsEngine !== "undefined" && newsEngine.isFetchingNews
                                          ? "Status: " + newsEngine.currentPhase
                                          : "Status: Idle"
                                    font.pixelSize: 12
                                    color: typeof newsEngine !== "undefined" && newsEngine.isFetchingNews
                                           ? "#569cd6" : "#888888"
                                }

                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // === Agentic Features ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: agenticSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: agenticSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Agentic Features"; font.pixelSize: 16; font.bold: true; color: "#569cd6" }

                            // Goal Tracking toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Goal Tracking:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.goalsEnabled
                                    onToggled: settingsManager.goalsEnabled = checked
                                }
                                Label {
                                    text: settingsManager.goalsEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Sentiment Tracking toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Mood Tracking:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.sentimentTrackingEnabled
                                    onToggled: settingsManager.sentimentTrackingEnabled = checked
                                }
                                Label {
                                    text: settingsManager.sentimentTrackingEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Teach Me toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Teach Me Mode:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.teachMeEnabled
                                    onToggled: settingsManager.teachMeEnabled = checked
                                }
                                Label {
                                    text: settingsManager.teachMeEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Explanation
                            Label {
                                text: "Goal tracking detects intentions in conversation and checks in periodically. "
                                      + "Mood tracking analyzes sentiment patterns over time. "
                                      + "Teach Me mode creates multi-session learning plans on topics you request."
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Stats
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: "Goals: " + (typeof goalTracker !== "undefined" ? goalTracker.activeGoalCount : 0)
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Label {
                                    text: "Mood: " + (typeof sentimentTracker !== "undefined" ? sentimentTracker.currentMood : "N/A")
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Label {
                                    text: "Reminders: " + (typeof reminderEngine !== "undefined" ? reminderEngine.pendingCount : 0)
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // === Proactive Dialog Settings ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: proactiveSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: proactiveSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Proactive Dialog"; font.pixelSize: 16; font.bold: true; color: "#569cd6" }

                            // Enable toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Proactive Thoughts:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.proactiveDialogsEnabled
                                    onToggled: settingsManager.proactiveDialogsEnabled = checked
                                }
                                Label {
                                    text: settingsManager.proactiveDialogsEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Max thoughts per day
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Daily Limit:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: thoughtsSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 15
                                    stepSize: 1
                                    value: settingsManager.maxThoughtsPerDay
                                    onPressedChanged: if (!pressed) settingsManager.maxThoughtsPerDay = value
                                }
                                Label {
                                    text: Math.round(thoughtsSlider.value)
                                    color: "#569cd6"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 30
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // Explanation
                            Label {
                                text: "DATAFORM generates thoughts during idle time based on research findings, "
                                      + "curiosity overflow, and training observations. Thoughts appear as cards in "
                                      + "the sidebar. Click one to start a conversation where DATAFORM speaks first."
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            // Pending count + dismiss
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12
                                visible: typeof thoughtEngine !== "undefined"

                                Label {
                                    text: "Pending: " + (typeof thoughtEngine !== "undefined" ? thoughtEngine.pendingCount : 0)
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: "Dismiss All"
                                    visible: typeof thoughtEngine !== "undefined" && thoughtEngine.pendingCount > 0
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.hovered ? "#ff6b6b" : "#888888"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#3e3e42" : "transparent"
                                        radius: 4
                                    }
                                    onClicked: thoughtEngine.dismissAllThoughts()
                                }
                            }
                        }
                    }

                    // === Semantic Memory ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: semanticSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: semanticSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Semantic Memory"; font.pixelSize: 16; font.bold: true; color: "#c586c0" }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Semantic Search:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: 140 }
                                Switch {
                                    checked: settingsManager.semanticSearchEnabled
                                    onToggled: settingsManager.semanticSearchEnabled = checked
                                }
                                Label {
                                    text: settingsManager.semanticSearchEnabled ? "Enabled" : "Disabled"
                                    font.pixelSize: 12
                                    color: "#888888"
                                }
                                Item { Layout.fillWidth: true }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Embed Model:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: 140 }
                                TextField {
                                    Layout.fillWidth: true
                                    text: settingsManager.embeddingModel
                                    color: "#d4d4d4"
                                    font.pixelSize: 13
                                    placeholderText: "nomic-embed-text"
                                    placeholderTextColor: "#666666"
                                    background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                                    onEditingFinished: settingsManager.embeddingModel = text
                                }
                            }

                            Label {
                                text: "Semantic search uses vector embeddings to find relevant memories by meaning "
                                      + "rather than keywords. Requires an embedding model in Ollama "
                                      + "(run: ollama pull nomic-embed-text). Embeddings are generated during idle time."
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12
                                visible: typeof embeddingManager !== "undefined"

                                Label {
                                    text: "Embedded: " + (typeof embeddingManager !== "undefined" ? embeddingManager.embeddedCount : 0)
                                    font.pixelSize: 12
                                    color: "#888888"
                                }
                                Label {
                                    text: "Pending: " + (typeof embeddingManager !== "undefined" ? embeddingManager.pendingCount : 0)
                                    font.pixelSize: 12
                                    color: "#888888"
                                }
                                Label {
                                    text: "Status: " + (typeof embeddingManager !== "undefined" ? embeddingManager.status : "N/A")
                                    font.pixelSize: 12
                                    color: typeof embeddingManager !== "undefined" && embeddingManager.isEmbedding
                                           ? "#c586c0" : "#888888"
                                }
                                Item { Layout.fillWidth: true }
                            }
                        }
                    }

                    // Footer
                    Label {
                        text: "DATAFORM v0.6.0 | Global Human Initiative"
                        font.pixelSize: 11
                        color: "#555555"
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                        Layout.bottomMargin: 20
                    }
                }
            }

            // ============================================================
            // TAB 3: TRAINING
            // ============================================================
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: trainingColumn.height + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                ColumnLayout {
                    id: trainingColumn
                    anchors.top: parent.top
                    anchors.topMargin: 20
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.min(parent.width - 40, 640)
                    spacing: 20

                    // === Learning Settings ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: learningSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: learningSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Learning"; font.pixelSize: 16; font.bold: true; color: "#dcdcaa" }

                            // Idle learning toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Idle Learning:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.idleLearningEnabled
                                    onToggled: settingsManager.idleLearningEnabled = checked
                                }
                                Label {
                                    text: settingsManager.idleLearningEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Compute budget
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Compute Budget:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: budgetSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 50
                                    stepSize: 1
                                    value: settingsManager.computeBudgetPercent
                                    onPressedChanged: if (!pressed) settingsManager.computeBudgetPercent = value
                                }
                                Label {
                                    text: Math.round(budgetSlider.value) + "%"
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 45
                                    horizontalAlignment: Text.AlignRight
                                }
                            }
                        }
                    }

                    // === Adapter Management ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: adapterSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8
                        visible: typeof adapterManager !== "undefined"

                        ColumnLayout {
                            id: adapterSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Adapters"; font.pixelSize: 16; font.bold: true; color: "#dcdcaa" }

                            // Active adapter info
                            GridLayout {
                                columns: 2
                                columnSpacing: 15
                                rowSpacing: 8
                                Layout.fillWidth: true

                                Label { text: "Active Version:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
                                          ? "v" + adapterManager.activeVersion
                                          : "None (base model)"
                                    color: typeof adapterManager !== "undefined" && adapterManager.activeVersion >= 0
                                           ? "#4ec9b0" : "#aaaaaa"
                                    font.pixelSize: 13
                                }

                                Label { text: "Total Versions:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: typeof adapterManager !== "undefined" ? "" + adapterManager.versionCount : "0"
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                }

                                Label { text: "Status:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: typeof adapterManager !== "undefined" ? adapterManager.adapterStatus : "N/A"
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                }

                                Label { text: "Training:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                                          ? reflectionEngine.phase + " - " + reflectionEngine.reflectionStatus
                                          : "Idle"
                                    color: typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting
                                           ? "#dcdcaa" : "#aaaaaa"
                                    font.pixelSize: 13
                                    Layout.fillWidth: true
                                    elide: Text.ElideRight
                                }
                            }

                            // Action buttons
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Button {
                                    text: "Train Now"
                                    enabled: typeof reflectionEngine !== "undefined" && !reflectionEngine.isReflecting
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#dcdcaa" : "#666666"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.enabled && parent.hovered ? "#3e3e42" : "#2d2d30"
                                        radius: 4
                                        border.color: parent.enabled ? "#5a5a3e" : "#3e3e42"
                                    }
                                    onClicked: {
                                        if (typeof reflectionEngine !== "undefined") {
                                            reflectionEngine.triggerReflection()
                                        }
                                    }
                                }

                                Button {
                                    text: "Evolve Now"
                                    enabled: typeof evolutionEngine !== "undefined" && !evolutionEngine.isEvolving
                                    visible: typeof evolutionEngine !== "undefined"
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#ce9178" : "#666666"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.enabled && parent.hovered ? "#3e3e42" : "#2d2d30"
                                        radius: 4
                                        border.color: parent.enabled ? "#5a3e30" : "#3e3e42"
                                    }
                                    onClicked: {
                                        if (typeof evolutionEngine !== "undefined") {
                                            evolutionEngine.triggerEvolutionCycle()
                                        }
                                    }
                                }

                                Button {
                                    text: "Pause"
                                    visible: (typeof reflectionEngine !== "undefined" && reflectionEngine.isReflecting)
                                             || (typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving)
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#ffa500"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#3e3e42" : "#2d2d30"
                                        radius: 4
                                        border.color: "#5a4520"
                                    }
                                    onClicked: {
                                        if (typeof evolutionEngine !== "undefined" && evolutionEngine.isEvolving) {
                                            evolutionEngine.pauseEvolution()
                                        } else if (typeof reflectionEngine !== "undefined") {
                                            reflectionEngine.pauseReflection()
                                        }
                                    }
                                }

                                Button {
                                    text: "Rollback"
                                    enabled: typeof adapterManager !== "undefined" && adapterManager.activeVersion > 0
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#ff6666" : "#666666"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.enabled && parent.hovered ? "#3e2020" : "#2d2d30"
                                        radius: 4
                                        border.color: parent.enabled ? "#5a2d2d" : "#3e3e42"
                                    }
                                    onClicked: {
                                        if (typeof adapterManager !== "undefined") {
                                            adapterManager.rollback()
                                        }
                                    }
                                }

                                Item { Layout.fillWidth: true }
                            }

                            // === Evolution Settings ===
                            Rectangle {
                                Layout.fillWidth: true
                                height: 1
                                color: "#3e3e42"
                                visible: typeof evolutionEngine !== "undefined"
                            }

                            Label {
                                text: "Evolution"
                                font.pixelSize: 14
                                font.bold: true
                                color: "#ce9178"
                                visible: typeof evolutionEngine !== "undefined"
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: typeof evolutionEngine !== "undefined"
                                Label { text: "Evolution:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.evolutionEnabled
                                    onToggled: settingsManager.evolutionEnabled = checked
                                }
                                Label {
                                    text: settingsManager.evolutionEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: typeof evolutionEngine !== "undefined"
                                Label { text: "Population:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: popSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 8
                                    stepSize: 1
                                    value: settingsManager.populationSize
                                    onPressedChanged: if (!pressed) settingsManager.populationSize = value
                                }
                                Label {
                                    text: Math.round(popSlider.value)
                                    color: "#ce9178"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 30
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: typeof evolutionEngine !== "undefined"
                                Label { text: "Consolidation:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Button {
                                    text: "Merge Now"
                                    enabled: typeof evolutionEngine !== "undefined" && !evolutionEngine.isEvolving
                                             && evolutionEngine.cyclesCompleted > 0
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#ce9178" : "#666666"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.enabled && parent.hovered ? "#3e3e42" : "#2d2d30"
                                        radius: 4
                                        border.color: parent.enabled ? "#5a3e30" : "#3e3e42"
                                    }
                                    onClicked: {
                                        if (typeof evolutionEngine !== "undefined") {
                                            evolutionEngine.triggerConsolidation()
                                        }
                                    }
                                }
                                Label {
                                    text: typeof evolutionEngine !== "undefined"
                                          ? "Cycles: " + evolutionEngine.cyclesCompleted
                                            + " | Merges: " + evolutionEngine.consolidationsCompleted
                                          : ""
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // === Distillation Settings (Phase 8) ===
                            Rectangle {
                                Layout.fillWidth: true
                                height: 1
                                color: "#3e3e42"
                            }

                            Label {
                                text: "Distillation"
                                font.pixelSize: 14
                                font.bold: true
                                color: "#c586c0"
                            }

                            Label {
                                text: "During idle time, generates high-quality training examples from your chat model to teach the local model your conversation patterns. Over time, this enables offline use."
                                font.pixelSize: 11
                                color: "#888888"
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Distillation:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.distillationEnabled
                                    onToggled: settingsManager.distillationEnabled = checked
                                }
                                Label {
                                    text: settingsManager.distillationEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Daily cycles:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: distillCyclesSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 10
                                    stepSize: 1
                                    value: settingsManager.distillationDailyCycles
                                    onPressedChanged: if (!pressed) settingsManager.distillationDailyCycles = value
                                }
                                Label {
                                    text: Math.round(distillCyclesSlider.value)
                                    color: "#c586c0"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 30
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // Distillation stats
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: "Pairs: " + distillationManager.pairsCollected
                                    font.pixelSize: 11
                                    color: distillationManager.pairsCollected > 0 ? "#c586c0" : "#666666"
                                }
                                Label {
                                    text: "Trained: " + distillationManager.pairsUsedInTraining
                                    font.pixelSize: 11
                                    color: "#666666"
                                }
                                Label {
                                    text: "Readiness: " + (distillationManager.readinessScore * 100).toFixed(0) + "%"
                                    font.pixelSize: 11
                                    color: distillationManager.readinessScore >= 0.75 ? "#4ec9b0"
                                         : distillationManager.readinessScore >= 0.5 ? "#c586c0" : "#666666"
                                }
                            }

                            // Teacher info + Distill Now button
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Label {
                                    text: "Teacher: " + settingsManager.provider + "/" + settingsManager.model
                                    font.pixelSize: 11
                                    color: "#888888"
                                    Layout.fillWidth: true
                                    elide: Text.ElideRight
                                }

                                Button {
                                    text: "Distill Now"
                                    enabled: settingsManager.distillationEnabled
                                             && !distillationManager.isDistilling
                                    contentItem: Text {
                                        text: parent.text
                                        color: parent.enabled ? "#c586c0" : "#666666"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.enabled && parent.hovered ? "#3e3e42" : "#2d2d30"
                                        radius: 4
                                        border.color: parent.enabled ? "#5a3050" : "#3e3e42"
                                    }
                                    onClicked: distillationManager.distillNow()
                                }
                            }
                        }
                    }

                    // Footer
                    Label {
                        text: "DATAFORM v0.6.0 | Global Human Initiative"
                        font.pixelSize: 11
                        color: "#555555"
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                        Layout.bottomMargin: 20
                    }
                }
            }

            // ============================================================
            // TAB 4: DATA
            // ============================================================
            Flickable {
                Layout.fillWidth: true
                Layout.fillHeight: true
                contentWidth: width
                contentHeight: dataColumn.height + 40
                clip: true
                boundsBehavior: Flickable.StopAtBounds
                ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                ColumnLayout {
                    id: dataColumn
                    anchors.top: parent.top
                    anchors.topMargin: 20
                    anchors.horizontalCenter: parent.horizontalCenter
                    width: Math.min(parent.width - 40, 640)
                    spacing: 20

                    // === Privacy & Security ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: privacySection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: privacySection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Privacy & Security"; font.pixelSize: 16; font.bold: true; color: "#ce9178" }

                            // Privacy level
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Privacy:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: ["minimal", "standard", "full"]
                                    currentIndex: model.indexOf(settingsManager.privacyLevel)
                                    onCurrentTextChanged: {
                                        if (currentText !== settingsManager.privacyLevel) {
                                            settingsManager.privacyLevel = currentText
                                        }
                                    }
                                }
                            }

                            // Encryption mode
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Encryption:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                ComboBox {
                                    id: encryptionModeCombo
                                    Layout.fillWidth: true
                                    model: ["portable", "machine_locked"]
                                    currentIndex: model.indexOf(settingsManager.encryptionMode)
                                    onCurrentTextChanged: {
                                        if (currentText !== settingsManager.encryptionMode) {
                                            settingsManager.encryptionMode = currentText
                                        }
                                    }
                                }
                            }
                            Label {
                                text: settingsManager.encryptionMode === "machine_locked"
                                      ? "Data encrypted with this machine's unique key. Cannot be moved to another computer."
                                      : "Data encrypted with a portable key. Profile can be copied to another machine."
                                color: "#888888"
                                font.pixelSize: 11
                                wrapMode: Text.WordWrap
                                Layout.fillWidth: true
                            }
                        }
                    }

                    // === Lifecycle ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: lifecycleSection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: lifecycleSection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Lifecycle"; font.pixelSize: 16; font.bold: true; color: "#ce9178" }

                            // Episode retention slider
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Retain Episodes:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: retentionSlider
                                    Layout.fillWidth: true
                                    from: 90
                                    to: 3650
                                    stepSize: 30
                                    value: settingsManager.episodeRetentionDays
                                    onPressedChanged: if (!pressed) settingsManager.episodeRetentionDays = value
                                }
                                Label {
                                    text: {
                                        var days = Math.round(retentionSlider.value)
                                        if (days >= 365) return (days / 365).toFixed(1) + " yr"
                                        return days + " days"
                                    }
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 55
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // Auto-backup toggle
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Auto Backup:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Switch {
                                    checked: settingsManager.autoBackupEnabled
                                    onToggled: settingsManager.autoBackupEnabled = checked
                                }
                                Label {
                                    text: settingsManager.autoBackupEnabled ? "Enabled" : "Disabled"
                                    color: "#888888"
                                    font.pixelSize: 12
                                }
                                Item { Layout.fillWidth: true }
                            }

                            // Max disk usage slider
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                Label { text: "Max Disk Usage:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Slider {
                                    id: diskSlider
                                    Layout.fillWidth: true
                                    from: 1
                                    to: 100
                                    stepSize: 1
                                    value: settingsManager.maxDiskUsageGB
                                    onPressedChanged: if (!pressed) settingsManager.maxDiskUsageGB = value
                                }
                                Label {
                                    text: Math.round(diskSlider.value) + " GB"
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                    Layout.preferredWidth: 55
                                    horizontalAlignment: Text.AlignRight
                                }
                            }

                            // Disk usage + health info
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 12

                                Label {
                                    text: "Current: " + dataLifecycleManager.diskUsageFormatted
                                    font.pixelSize: 12
                                    color: "#888888"
                                }

                                Rectangle { width: 1; height: 14; color: "#444444"; Layout.alignment: Qt.AlignVCenter }

                                Rectangle {
                                    width: 8; height: 8; radius: 4
                                    color: profileHealthManager.isHealthy ? "#4ec9b0" : "#ffa500"
                                    Layout.alignment: Qt.AlignVCenter
                                }

                                Label {
                                    text: profileHealthManager.healthStatus
                                    font.pixelSize: 12
                                    color: profileHealthManager.isHealthy ? "#4ec9b0" : "#ffa500"
                                }

                                Item { Layout.fillWidth: true }
                            }

                            // Action buttons
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Button {
                                    text: "Backup Now"
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#569cd6"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#203040" : "#2d2d30"
                                        radius: 4
                                        border.color: "#2d4a6a"
                                    }
                                    onClicked: profileHealthManager.createBackup()
                                }

                                Button {
                                    text: "Check Health"
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#4ec9b0"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#1e3530" : "#2d2d30"
                                        radius: 4
                                        border.color: "#2d5a4a"
                                    }
                                    onClicked: profileHealthManager.runStartupCheck()
                                }

                                Item { Layout.fillWidth: true }
                            }

                            // Model generation info
                            Rectangle {
                                Layout.fillWidth: true
                                height: 1
                                color: "#3e3e42"
                            }

                            Label {
                                text: "Model Generation"
                                font.pixelSize: 14
                                font.bold: true
                                color: "#ce9178"
                            }

                            GridLayout {
                                columns: 2
                                columnSpacing: 15
                                rowSpacing: 6
                                Layout.fillWidth: true

                                Label { text: "Family:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: modelGenManager.modelFamily
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                }

                                Label { text: "Variant:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: modelGenManager.modelVariant
                                    color: "#cccccc"
                                    font.pixelSize: 13
                                }

                                Label { text: "Generation:"; color: "#888888"; font.pixelSize: 13; Layout.preferredWidth: settingsRoot.labelWidth }
                                Label {
                                    text: "#" + modelGenManager.currentGenerationId
                                    color: "#569cd6"
                                    font.pixelSize: 13
                                }
                            }
                        }
                    }

                    // === Profile Portability ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: portabilitySection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: portabilitySection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Profile Portability"; font.pixelSize: 16; font.bold: true; color: "#ce9178" }

                            Label {
                                text: "Export your entire profile to move between machines (laptop, desktop, etc.). "
                                      + "The export includes all conversations, traits, research, adapters, and settings."
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Button {
                                    text: "Export Profile"
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#569cd6"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#203040" : "#2d2d30"
                                        radius: 4
                                        border.color: "#2d4a6a"
                                    }
                                    onClicked: exportFolderDialog.open()
                                }

                                Button {
                                    text: "Import Profile"
                                    contentItem: Text {
                                        text: parent.text
                                        color: "#ce9178"
                                        font.pixelSize: 12
                                    }
                                    background: Rectangle {
                                        color: parent.hovered ? "#302520" : "#2d2d30"
                                        radius: 4
                                        border.color: "#5a3e30"
                                    }
                                    onClicked: importFolderDialog.open()
                                }

                                Item { Layout.fillWidth: true }
                            }

                            Label {
                                id: portabilityStatusLabel
                                text: ""
                                font.pixelSize: 12
                                color: "#4ec9b0"
                                visible: text.length > 0
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap

                                Timer {
                                    id: portabilityStatusTimer
                                    interval: 5000
                                    onTriggered: portabilityStatusLabel.text = ""
                                }
                            }

                            Label {
                                text: "Profile path: " + profileManager.profilePath
                                font.pixelSize: 10
                                color: "#555555"
                                Layout.fillWidth: true
                                elide: Text.ElideMiddle
                            }
                        }
                    }

                    // === Memory Management ===
                    Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: memorySection.implicitHeight + 30
                        color: "#252526"
                        radius: 8

                        ColumnLayout {
                            id: memorySection
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 15
                            spacing: 12

                            Label { text: "Memory"; font.pixelSize: 16; font.bold: true; color: "#ce9178" }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 20
                                Label { text: "Episodes: " + memoryStore.episodeCount; color: "#cccccc"; font.pixelSize: 13 }
                                Label { text: "Traits: " + memoryStore.traitCount; color: "#cccccc"; font.pixelSize: 13 }
                                Item { Layout.fillWidth: true }
                            }

                            RowLayout {
                                spacing: 10
                                Button {
                                    text: "Clear All Memory"
                                    contentItem: Text { text: parent.text; color: "#ff6666"; font.pixelSize: 12 }
                                    background: Rectangle { color: parent.hovered ? "#3e2020" : "#2d2d30"; radius: 4; border.color: "#5a2d2d" }
                                    onClicked: clearDialog.open()
                                }
                            }
                        }
                    }

                    // Footer
                    Label {
                        text: "DATAFORM v0.6.0 | Global Human Initiative"
                        font.pixelSize: 11
                        color: "#555555"
                        Layout.alignment: Qt.AlignHCenter
                        Layout.topMargin: 10
                        Layout.bottomMargin: 20
                    }
                }
            }
        }
    }

    // === Dialogs ===

    FileDialog {
        id: ggufFileDialog
        title: "Select GGUF Model"
        nameFilters: ["GGUF Models (*.gguf)", "All Files (*)"]
        onAccepted: {
            var path = selectedFile.toString()
            // Strip file:/// prefix on Windows
            if (path.startsWith("file:///"))
                path = path.substring(8)
            settingsManager.backgroundModelPath = path
        }
    }

    Dialog {
        id: clearDialog
        title: "Clear All Memory"
        anchors.centerIn: parent
        modal: true
        standardButtons: Dialog.Yes | Dialog.No

        Label {
            text: "This will permanently delete all episodes and traits.\nAre you sure?"
            color: "#d4d4d4"
        }

        background: Rectangle { color: "#2d2d30"; radius: 8; border.color: "#3e3e42" }

        onAccepted: {
            memoryStore.clearAllMemory()
        }
    }

    FolderDialog {
        id: exportFolderDialog
        title: "Choose export destination"
        onAccepted: {
            var path = selectedFolder.toString()
            if (Qt.platform.os === "windows") {
                path = path.replace(/^file:\/\/\//, "")
            } else {
                path = path.replace(/^file:\/\//, "")
            }
            orchestrator.exportProfile(path)
        }
    }

    FolderDialog {
        id: importFolderDialog
        title: "Select profile folder to import"
        onAccepted: {
            var path = selectedFolder.toString()
            if (Qt.platform.os === "windows") {
                path = path.replace(/^file:\/\/\//, "")
            } else {
                path = path.replace(/^file:\/\//, "")
            }
            orchestrator.importProfile(path)
        }
    }

    // Profile export/import feedback
    Connections {
        target: profileManager

        function onExportComplete(success, message) {
            if (success) {
                portabilityStatusLabel.text = "Exported to: " + message
                portabilityStatusLabel.color = "#4ec9b0"
            } else {
                portabilityStatusLabel.text = "Export failed: " + message
                portabilityStatusLabel.color = "#ff6b6b"
            }
            portabilityStatusTimer.start()
        }

        function onImportComplete(success, message) {
            if (success) {
                portabilityStatusLabel.text = "Imported from: " + message
                portabilityStatusLabel.color = "#4ec9b0"
            } else {
                portabilityStatusLabel.text = "Import failed: " + message
                portabilityStatusLabel.color = "#ff6b6b"
            }
            portabilityStatusTimer.start()
        }
    }
}
