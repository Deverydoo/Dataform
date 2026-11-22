import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs

// Voice tab content for SettingsPanel.
//
// Split into its own file because SettingsPanel.qml is already long, and because
// this whole tab is meaningless in a build without the voice loop.
Flickable {
    id: voiceSettingsRoot

    property int labelWidth: 130

    readonly property bool built: voiceBuildEnabled
    readonly property bool sessionReady: built && typeof voiceSession !== 'undefined'
                                          && voiceSession !== null

    contentWidth: width
    contentHeight: voiceColumn.height + 40
    clip: true
    boundsBehavior: Flickable.StopAtBounds
    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

    function stripFileUrl(url) {
        var path = url.toString()
        if (Qt.platform.os === "windows")
            return path.replace(/^file:\/\/\//, "")
        return path.replace(/^file:\/\//, "")
    }

    ColumnLayout {
        id: voiceColumn
        anchors.top: parent.top
        anchors.topMargin: 20
        anchors.horizontalCenter: parent.horizontalCenter
        width: Math.min(parent.width - 40, 640)
        spacing: 20

        // === Not built ===
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: notBuiltCol.implicitHeight + 30
            color: "#252526"
            radius: 8
            visible: !voiceSettingsRoot.built

            ColumnLayout {
                id: notBuiltCol
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 15
                spacing: 8

                Label { text: "Voice is not in this build"; font.pixelSize: 16; font.bold: true; color: "#f48771" }
                Label {
                    Layout.fillWidth: true
                    wrapMode: Text.Wrap
                    font.pixelSize: 12
                    color: "#9a9a9a"
                    text: "This copy of Dataform was compiled without the voice loop, which needs "
                          + "a built Muninn checkout. Reconfigure with "
                          + "-DMUNINN_ROOT=<path to Muninn-Faster-Whisper> and rebuild."
                }
            }
        }

        // === Master switch ===
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: enableSection.implicitHeight + 30
            color: "#252526"
            radius: 8
            visible: voiceSettingsRoot.built

            ColumnLayout {
                id: enableSection
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 15
                spacing: 12

                Label { text: "Spoken Assistant"; font.pixelSize: 16; font.bold: true; color: "#4fc1ff" }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Enable voice:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    Switch {
                        checked: settingsManager.voiceEnabled
                        onToggled: settingsManager.voiceEnabled = checked
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        font.pixelSize: 11
                        color: "#888888"
                        text: "Loads the speech model and enables push-to-talk."
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Speak replies:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    Switch {
                        checked: settingsManager.speakReplies
                        onToggled: settingsManager.speakReplies = checked
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Assistant name:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: settingsManager.assistantName
                        color: "#d4d4d4"
                        font.pixelSize: 13
                        placeholderText: "What it answers to"
                        placeholderTextColor: "#666666"
                        background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                        onEditingFinished: settingsManager.assistantName = text
                    }
                }

                // Live status, so a misconfiguration is visible here rather than
                // only when the talk button refuses to work.
                Rectangle {
                    Layout.fillWidth: true
                    Layout.topMargin: 4
                    implicitHeight: statusCol.implicitHeight + 16
                    color: "#1e1e1e"
                    radius: 4
                    visible: voiceSettingsRoot.sessionReady

                    ColumnLayout {
                        id: statusCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 8
                        spacing: 4

                        RowLayout {
                            spacing: 6
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: voiceSettingsRoot.sessionReady && voiceSession.voiceInputReady ? "#4ec9b0" : "#f48771"
                            }
                            Label {
                                font.pixelSize: 11
                                color: "#cccccc"
                                text: voiceSettingsRoot.sessionReady && voiceSession.voiceInputReady
                                      ? "Listening ready — " + audioCapture.deviceName
                                      : "Input not ready"
                            }
                        }

                        RowLayout {
                            spacing: 6
                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: voiceSettingsRoot.sessionReady && voiceSession.voiceOutputReady ? "#4ec9b0" : "#f48771"
                            }
                            Label {
                                font.pixelSize: 11
                                color: "#cccccc"
                                text: voiceSettingsRoot.sessionReady && voiceSession.voiceOutputReady
                                      ? "Speech ready — " + speechSynthesizer.backendName
                                      : "Speech not ready"
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.Wrap
                            font.pixelSize: 11
                            color: "#f48771"
                            visible: voiceSettingsRoot.sessionReady && voiceSession.lastError.length > 0
                            text: voiceSettingsRoot.sessionReady ? voiceSession.lastError : ""
                        }
                    }
                }
            }
        }

        // === Speech recognition ===
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: sttSection.implicitHeight + 30
            color: "#252526"
            radius: 8
            visible: voiceSettingsRoot.built

            ColumnLayout {
                id: sttSection
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 15
                spacing: 12

                Label { text: "Speech Recognition (Muninn)"; font.pixelSize: 16; font.bold: true; color: "#4fc1ff" }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Model folder:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: settingsManager.sttModelPath
                        color: "#d4d4d4"
                        font.pixelSize: 12
                        placeholderText: "CTranslate2 Whisper model directory"
                        placeholderTextColor: "#666666"
                        background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                        onEditingFinished: settingsManager.sttModelPath = text
                    }
                    Button {
                        text: "Browse…"
                        onClicked: sttFolderDialog.open()
                        contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                        background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: voiceSettingsRoot.labelWidth + 10
                    wrapMode: Text.Wrap
                    font.pixelSize: 11
                    color: "#888888"
                    text: "A faster-whisper model folder, e.g. faster-whisper-large-v3-turbo. "
                          + "Turbo is the usual choice: accuracy close to large-v3 at a fraction of the latency."
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Device:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    ComboBox {
                        Layout.preferredWidth: 120
                        model: ["cuda", "cpu", "auto"]
                        currentIndex: Math.max(0, model.indexOf(settingsManager.sttDevice))
                        onActivated: settingsManager.sttDevice = currentText
                    }
                    Label { text: "Precision:"; color: "#cccccc"; font.pixelSize: 13 }
                    ComboBox {
                        Layout.preferredWidth: 120
                        model: ["float16", "int8", "float32"]
                        currentIndex: Math.max(0, model.indexOf(settingsManager.sttComputeType))
                        onActivated: settingsManager.sttComputeType = currentText
                    }
                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Language:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    ComboBox {
                        Layout.preferredWidth: 140
                        // "auto" costs a detection pass per utterance, so a fixed
                        // language is the default for a single-user assistant.
                        model: ["en", "auto", "es", "fr", "de", "it", "pt", "nl", "ru", "ja", "ko", "zh"]
                        currentIndex: Math.max(0, model.indexOf(settingsManager.sttLanguage))
                        onActivated: settingsManager.sttLanguage = currentText
                    }
                    Item { Layout.fillWidth: true }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Silero VAD:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: settingsManager.sttVadModelPath
                        color: "#d4d4d4"
                        font.pixelSize: 12
                        placeholderText: "Auto — silero_vad.onnx beside the model folder"
                        placeholderTextColor: "#666666"
                        background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                        onEditingFinished: settingsManager.sttVadModelPath = text
                    }
                    Button {
                        text: "Browse…"
                        onClicked: vadModelDialog.open()
                        contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                        background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: voiceSettingsRoot.labelWidth + 10
                    wrapMode: Text.Wrap
                    font.pixelSize: 11
                    color: "#888888"
                    text: "Neural speech detection, better than the energy fallback in a noisy room. "
                          + "Left empty, Dataform looks for silero_vad.onnx next to the Whisper model "
                          + "folder and quietly uses energy detection if it is not there."
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Microphone:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    ComboBox {
                        id: micCombo
                        Layout.fillWidth: true
                        model: {
                            if (!voiceSettingsRoot.sessionReady)
                                return ["(system default)"]
                            var devices = audioCapture.availableDevices()
                            return ["(system default)"].concat(devices)
                        }
                        currentIndex: {
                            if (settingsManager.micDeviceName.length === 0)
                                return 0
                            var idx = model.indexOf(settingsManager.micDeviceName)
                            return idx >= 0 ? idx : 0
                        }
                        onActivated: settingsManager.micDeviceName = (currentIndex === 0 ? "" : currentText)
                    }
                }
            }
        }

        // === Speech synthesis ===
        Rectangle {
            Layout.fillWidth: true
            implicitHeight: ttsSection.implicitHeight + 30
            color: "#252526"
            radius: 8
            visible: voiceSettingsRoot.built

            ColumnLayout {
                id: ttsSection
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 15
                spacing: 12

                Label { text: "Speech Synthesis"; font.pixelSize: 16; font.bold: true; color: "#4fc1ff" }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    Label { text: "Engine:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    ComboBox {
                        id: backendCombo
                        Layout.preferredWidth: 160
                        model: ["piper", "sapi", "none"]
                        currentIndex: Math.max(0, model.indexOf(settingsManager.ttsBackend))
                        onActivated: settingsManager.ttsBackend = currentText
                    }
                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        font.pixelSize: 11
                        color: "#888888"
                        text: {
                            switch (settingsManager.ttsBackend) {
                            case "piper": return "Fast local neural voice. Needs a Piper release and a voice model."
                            case "sapi":  return "Built into Windows. Robotic, but works with no downloads."
                            default:      return "Replies are shown but never spoken."
                            }
                        }
                    }
                }

                // --- Piper ---
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    visible: settingsManager.ttsBackend === "piper"
                    Label { text: "piper.exe:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: settingsManager.piperExePath
                        color: "#d4d4d4"
                        font.pixelSize: 12
                        placeholderText: "Path to piper.exe"
                        placeholderTextColor: "#666666"
                        background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                        onEditingFinished: settingsManager.piperExePath = text
                    }
                    Button {
                        text: "Browse…"
                        onClicked: piperExeDialog.open()
                        contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                        background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    visible: settingsManager.ttsBackend === "piper"
                    Label { text: "Voice model:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: settingsManager.piperVoicePath
                        color: "#d4d4d4"
                        font.pixelSize: 12
                        placeholderText: "Path to a voice .onnx"
                        placeholderTextColor: "#666666"
                        background: Rectangle { color: "#1e1e1e"; radius: 4; border.color: "#3e3e42" }
                        onEditingFinished: settingsManager.piperVoicePath = text
                    }
                    Button {
                        text: "Browse…"
                        onClicked: piperVoiceDialog.open()
                        contentItem: Text { text: parent.text; color: "#cccccc"; font.pixelSize: 12 }
                        background: Rectangle { color: parent.hovered ? "#3e3e42" : "#2d2d30"; radius: 4; border.color: "#3e3e42" }
                    }
                }

                Label {
                    Layout.fillWidth: true
                    Layout.leftMargin: voiceSettingsRoot.labelWidth + 10
                    wrapMode: Text.Wrap
                    font.pixelSize: 11
                    color: "#888888"
                    visible: settingsManager.ttsBackend === "piper"
                    text: "The voice needs its .onnx.json config beside the .onnx file — that is "
                          + "where the sample rate comes from. Voices are at rhasspy/piper-voices on Hugging Face."
                }

                // --- SAPI ---
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    visible: settingsManager.ttsBackend === "sapi"
                    Label { text: "Voice:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    ComboBox {
                        Layout.fillWidth: true
                        model: {
                            var voices = voiceSettingsRoot.sessionReady
                                         ? speechSynthesizer.availableSapiVoices() : []
                            return ["(system default)"].concat(voices)
                        }
                        currentIndex: {
                            if (settingsManager.sapiVoiceName.length === 0)
                                return 0
                            var idx = model.indexOf(settingsManager.sapiVoiceName)
                            return idx >= 0 ? idx : 0
                        }
                        onActivated: settingsManager.sapiVoiceName = (currentIndex === 0 ? "" : currentText)
                    }
                }

                // --- Shared ---
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    visible: settingsManager.ttsBackend !== "none"
                    Label { text: "Speed:"; color: "#cccccc"; font.pixelSize: 13; Layout.preferredWidth: voiceSettingsRoot.labelWidth }
                    Slider {
                        id: speedSlider
                        Layout.fillWidth: true
                        from: 0.5
                        to: 2.0
                        stepSize: 0.05
                        value: settingsManager.ttsSpeed
                        onMoved: settingsManager.ttsSpeed = value
                    }
                    Label {
                        text: speedSlider.value.toFixed(2) + "×"
                        color: "#9a9a9a"
                        font.pixelSize: 12
                        Layout.preferredWidth: 44
                    }
                }

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 10
                    visible: settingsManager.ttsBackend !== "none"

                    Item { Layout.preferredWidth: voiceSettingsRoot.labelWidth }

                    Button {
                        text: "Test voice"
                        enabled: voiceSettingsRoot.sessionReady && voiceSession.voiceOutputReady
                        onClicked: voiceSession.announce(
                            "This is " + settingsManager.assistantName + ", ready when you are.")
                        contentItem: Text {
                            text: parent.text
                            color: parent.enabled ? "#ffffff" : "#666666"
                            font.pixelSize: 12
                        }
                        background: Rectangle {
                            color: parent.enabled ? (parent.hovered ? "#5fd4c0" : "#4ec9b0") : "#3e3e42"
                            radius: 4
                        }
                    }

                    Label {
                        Layout.fillWidth: true
                        wrapMode: Text.Wrap
                        font.pixelSize: 11
                        color: "#f48771"
                        visible: voiceSettingsRoot.sessionReady && !voiceSession.voiceOutputReady
                        text: voiceSettingsRoot.sessionReady ? speechSynthesizer.unavailableReason : ""
                    }
                }
            }
        }
    }

    FolderDialog {
        id: sttFolderDialog
        title: "Select the Whisper model folder"
        onAccepted: settingsManager.sttModelPath = voiceSettingsRoot.stripFileUrl(selectedFolder)
    }

    FileDialog {
        id: vadModelDialog
        title: "Select the Silero VAD model"
        nameFilters: ["ONNX models (*.onnx)", "All Files (*)"]
        onAccepted: settingsManager.sttVadModelPath = voiceSettingsRoot.stripFileUrl(selectedFile)
    }

    FileDialog {
        id: piperExeDialog
        title: "Select piper.exe"
        nameFilters: ["Executables (*.exe)", "All Files (*)"]
        onAccepted: settingsManager.piperExePath = voiceSettingsRoot.stripFileUrl(selectedFile)
    }

    FileDialog {
        id: piperVoiceDialog
        title: "Select a Piper voice model"
        nameFilters: ["Piper voices (*.onnx)", "All Files (*)"]
        onAccepted: settingsManager.piperVoicePath = voiceSettingsRoot.stripFileUrl(selectedFile)
    }
}
