import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Push-to-talk control and turn-state display.
//
// Sits above the chat input. Hidden entirely in a text-only build so the chat
// does not carry a dead control, and collapsed to a single hint line when voice
// is built but not yet configured.
Rectangle {
    id: voiceBar

    readonly property bool available: voiceBuildEnabled && typeof voiceSession !== 'undefined'
                                      && voiceSession !== null
    readonly property bool ready: available && voiceSession.voiceInputReady
    readonly property int turnState: available ? voiceSession.state : 0

    // VoiceSession.State, mirrored because Q_ENUM values are not addressable by
    // name from a context property.
    readonly property int stateIdle: 0
    readonly property int stateCapturing: 1
    readonly property int stateTranscribing: 2
    readonly property int stateThinking: 3
    readonly property int stateExecuting: 4
    readonly property int stateSpeaking: 5
    readonly property int stateCancelled: 6
    readonly property int stateError: 7

    function stateColor(s) {
        switch (s) {
        case stateCapturing:    return "#4ec9b0"
        case stateTranscribing: return "#569cd6"
        case stateThinking:     return "#c586c0"
        case stateExecuting:    return "#dcdcaa"
        case stateSpeaking:     return "#4fc1ff"
        case stateCancelled:    return "#888888"
        case stateError:        return "#f48771"
        default:                return "#3e3e42"
        }
    }

    visible: available
    Layout.fillWidth: true
    implicitHeight: visible ? content.implicitHeight + 12 : 0
    color: "#2d2d30"
    radius: 4

    RowLayout {
        id: content
        anchors.fill: parent
        anchors.margins: 6
        spacing: 8

        // --- Push to talk ---
        Button {
            id: talkButton
            enabled: voiceBar.ready
            implicitHeight: 30
            implicitWidth: 104
            focusPolicy: Qt.NoFocus

            contentItem: RowLayout {
                spacing: 5
                Item { Layout.fillWidth: true }
                Rectangle {
                    width: 8; height: 8; radius: 4
                    color: voiceBar.turnState === voiceBar.stateCapturing ? "#ffffff" : "#1e1e1e"
                    opacity: voiceBar.turnState === voiceBar.stateCapturing ? 1.0 : 0.5
                    SequentialAnimation on opacity {
                        running: voiceBar.turnState === voiceBar.stateCapturing
                        loops: Animation.Infinite
                        NumberAnimation { to: 0.25; duration: 600 }
                        NumberAnimation { to: 1.0; duration: 600 }
                    }
                }
                Text {
                    text: voiceBar.turnState === voiceBar.stateCapturing ? "Listening" : "Hold to talk"
                    color: talkButton.enabled ? "#ffffff" : "#666666"
                    font.pixelSize: 11
                }
                Item { Layout.fillWidth: true }
            }

            background: Rectangle {
                radius: 4
                color: voiceBar.turnState === voiceBar.stateCapturing
                       ? "#4ec9b0"
                       : (talkButton.enabled ? (talkButton.hovered ? "#4a4a50" : "#3e3e42") : "#333337")
            }

            // Press and release rather than onClicked: the plan specifies
            // push-to-talk, so capture must follow the button being held.
            onPressed: voiceSession.startListening()
            onReleased: voiceSession.stopListening()
            onCanceled: voiceSession.cancel()

            ToolTip.visible: hovered && !enabled
            ToolTip.text: voiceBar.available && voiceSession.lastError.length > 0
                          ? voiceSession.lastError
                          : "Configure the speech model in Settings > Voice."
        }

        // --- Input level ---
        Rectangle {
            Layout.preferredWidth: 46
            Layout.preferredHeight: 4
            radius: 2
            color: "#1e1e1e"
            visible: voiceBar.turnState === voiceBar.stateCapturing

            Rectangle {
                // RMS rarely exceeds ~0.3 for normal speech, so the meter is
                // scaled to make ordinary levels legible instead of a stub.
                width: parent.width * Math.min(1.0, (voiceBar.available ? voiceSession.inputLevel : 0) * 3.5)
                height: parent.height
                radius: 2
                color: "#4ec9b0"
                Behavior on width { NumberAnimation { duration: 60 } }
            }
        }

        // --- State pill ---
        Rectangle {
            Layout.preferredHeight: 20
            Layout.preferredWidth: stateLabel.implicitWidth + 18
            radius: 10
            color: "#1e1e1e"
            border.width: 1
            border.color: voiceBar.stateColor(voiceBar.turnState)
            visible: voiceBar.turnState !== voiceBar.stateIdle

            Text {
                id: stateLabel
                anchors.centerIn: parent
                text: voiceBar.available ? voiceSession.stateLabel : ""
                color: voiceBar.stateColor(voiceBar.turnState)
                font.pixelSize: 10
            }
        }

        // --- Live transcript / error ---
        Text {
            Layout.fillWidth: true
            elide: Text.ElideRight
            font.pixelSize: 11
            font.italic: true
            color: voiceBar.turnState === voiceBar.stateError ? "#f48771" : "#9a9a9a"

            text: {
                if (!voiceBar.available)
                    return ""
                if (voiceBar.turnState === voiceBar.stateError)
                    return voiceSession.lastError
                if (voiceSession.partialTranscript.length > 0)
                    return voiceSession.partialTranscript
                if (!voiceBar.ready)
                    return voiceSession.lastError.length > 0
                           ? voiceSession.lastError
                           : "Voice is not configured yet — Settings > Voice."
                if (!voiceSession.voiceOutputReady && voiceSession.speakReplies)
                    return "Replies will be shown, not spoken: " + speechSynthesizer.unavailableReason
                return ""
            }
        }

        // --- Speak replies toggle ---
        Button {
            implicitHeight: 24
            implicitWidth: 24
            focusPolicy: Qt.NoFocus
            enabled: voiceBar.available
            checkable: true
            checked: voiceBar.available && voiceSession.speakReplies

            onToggled: voiceSession.speakReplies = checked

            contentItem: Text {
                text: parent.checked ? "▶" : "✕"
                color: parent.checked ? "#4fc1ff" : "#777777"
                font.pixelSize: 11
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 4
                color: parent.hovered ? "#3e3e42" : "transparent"
            }

            ToolTip.visible: hovered
            ToolTip.text: checked ? "Spoken replies on" : "Spoken replies off"
        }

        // --- Cancel ---
        Button {
            implicitHeight: 24
            implicitWidth: 58
            focusPolicy: Qt.NoFocus
            visible: voiceBar.available && voiceSession.canCancel

            onClicked: voiceSession.cancel()

            contentItem: Text {
                text: "Stop"
                color: "#f48771"
                font.pixelSize: 11
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }
            background: Rectangle {
                radius: 4
                color: parent.hovered ? "#4a3330" : "#3e3e42"
            }

            ToolTip.visible: hovered
            ToolTip.text: "Stop listening or speaking. Actions already completed are kept."
        }
    }

    // Keyboard alternative to holding the button. A shortcut cannot report key
    // release, so this toggles: one press starts, the next ends the utterance.
    Shortcut {
        sequence: "Ctrl+Space"
        enabled: voiceBar.ready
        onActivated: {
            if (voiceBar.turnState === voiceBar.stateCapturing)
                voiceSession.stopListening()
            else
                voiceSession.startListening()
        }
    }

    Shortcut {
        sequence: "Esc"
        enabled: voiceBar.available && voiceSession.canCancel
        onActivated: voiceSession.cancel()
    }
}
