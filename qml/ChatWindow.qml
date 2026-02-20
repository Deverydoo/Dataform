import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: chatRoot
    color: "#252526"
    radius: 6

    // Pending images for multimodal messages (base64 strings)
    property var pendingImages: []
    property string lastMessageDate: ""

    ListModel {
        id: messageModel
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 10
        spacing: 8

        // Header
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: "Chat"
                font.pixelSize: 16
                font.bold: true
                color: "#ffffff"
            }

            Item { Layout.fillWidth: true }

            // Curiosity level indicator
            RowLayout {
                spacing: 4
                visible: typeof whyEngine !== 'undefined'

                Label {
                    text: "Curiosity:"
                    font.pixelSize: 10
                    color: "#666666"
                }

                Rectangle {
                    width: 40
                    height: 4
                    radius: 2
                    color: "#3e3e42"

                    Rectangle {
                        width: parent.width * (orchestrator.curiosityLevel || 0.5)
                        height: parent.height
                        radius: 2
                        color: "#4ec9b0"
                    }
                }
            }

            Label {
                text: llmProvider.currentModel || "No model selected"
                font.pixelSize: 11
                color: "#888888"
            }

            // Connection indicator
            Rectangle {
                width: 8
                height: 8
                radius: 4
                color: llmProvider.isConnected ? "#4ec9b0" : "#858585"
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
        }

        // Messages area
        ListView {
            id: messageList
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true
            model: messageModel
            spacing: 8
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                delegate: Item {
                    width: messageList.width - 20
                    height: model.messageType === "date_header" ? dateHeaderLoader.height : messageBubble.height

                    // Date header display
                    Loader {
                        id: dateHeaderLoader
                        active: model.messageType === "date_header"
                        visible: active
                        width: parent.width
                        sourceComponent: Label {
                            text: model.content
                            color: "#666666"
                            font.pixelSize: 11
                            horizontalAlignment: Text.AlignHCenter
                            topPadding: 8
                            bottomPadding: 4
                        }
                    }

                    // Normal message bubble
                    Rectangle {
                        id: messageBubble
                        width: parent.width
                        height: model.messageType === "date_header" ? 0 : messageContent.height + feedbackRow.height + 20
                        visible: model.messageType !== "date_header"
                        radius: 8
                        color: {
                            if (model.messageType === "system") return "#2d2d30"
                            if (model.messageType === "agent") return "#1e3a5f"
                            return "#2d4a2d"
                        }

                        ColumnLayout {
                            anchors.fill: parent
                            anchors.margins: 10
                            spacing: 6

                            // Message header
                            RowLayout {
                                Layout.fillWidth: true

                                Label {
                                    text: {
                                        if (model.messageType === "system") return "System"
                                        if (model.messageType === "agent") {
                                            if (index === 0 && model.isInitiated)
                                                return "DATAFORM (initiated)"
                                            return "DATAFORM"
                                        }
                                        return settingsManager.userName
                                    }
                                    font.pixelSize: 11
                                    font.bold: true
                                    color: {
                                        if (model.messageType === "system") return "#b0b0b0"
                                        if (model.messageType === "agent") return "#4ec9b0"
                                        return "#6a9955"
                                    }
                                }

                                Item { Layout.fillWidth: true }

                                Label {
                                    text: model.timestamp || ""
                                    font.pixelSize: 10
                                    color: "#666666"
                                }
                            }

                            // Attached images (shown before text for user messages)
                            Flow {
                                Layout.fillWidth: true
                                spacing: 6
                                visible: (model.imageData || "") !== ""

                                Repeater {
                                    model: {
                                        var raw = (typeof messageModel.get(index) !== 'undefined')
                                                  ? messageModel.get(index).imageData : ""
                                        if (!raw || raw === "") return []
                                        return raw.split("|IMG|")
                                    }

                                    Image {
                                        source: "data:image/png;base64," + modelData
                                        width: Math.min(200, sourceSize.width)
                                        height: width * (sourceSize.height / Math.max(1, sourceSize.width))
                                        fillMode: Image.PreserveAspectFit
                                        smooth: true

                                        Rectangle {
                                            anchors.fill: parent
                                            color: "transparent"
                                            border.color: "#4ec9b0"
                                            border.width: 1
                                            radius: 4
                                        }
                                    }
                                }
                            }

                            // Message content (Markdown for agent, plain for user)
                            Text {
                                id: messageContent
                                Layout.fillWidth: true
                                text: model.content || ""
                                visible: (model.content || "") !== ""
                                textFormat: model.messageType === "agent" ? Text.MarkdownText : Text.PlainText
                                wrapMode: Text.WordWrap
                                font.pixelSize: 13
                                color: "#d4d4d4"
                                linkColor: "#4ec9b0"
                                onLinkActivated: function(link) { Qt.openUrlExternally(link) }
                            }

                            // Feedback buttons (only on latest agent message)
                            RowLayout {
                                id: feedbackRow
                                Layout.fillWidth: true
                                visible: model.messageType === "agent" && index === messageModel.count - 1
                                spacing: 8

                                Item { Layout.fillWidth: true }

                                Button {
                                    text: "\uD83D\uDC4D"
                                    font.pixelSize: 14
                                    flat: true
                                    implicitWidth: 36
                                    implicitHeight: 28

                                    ToolTip.text: "Good response"
                                    ToolTip.visible: hovered
                                    ToolTip.delay: 500

                                    background: Rectangle {
                                        color: parent.hovered ? "#2d5a2d" : "transparent"
                                        radius: 4
                                    }

                                    onClicked: {
                                        orchestrator.submitFeedback(1)
                                        visible = false
                                    }
                                }

                                Button {
                                    text: "\uD83D\uDC4E"
                                    font.pixelSize: 14
                                    flat: true
                                    implicitWidth: 36
                                    implicitHeight: 28

                                    ToolTip.text: "Bad response"
                                    ToolTip.visible: hovered
                                    ToolTip.delay: 500

                                    background: Rectangle {
                                        color: parent.hovered ? "#5a2d2d" : "transparent"
                                        radius: 4
                                    }

                                    onClicked: {
                                        orchestrator.submitFeedback(-1)
                                        visible = false
                                    }
                                }
                            }
                        }
                    }
                }

                // Auto-scroll to bottom
                onCountChanged: {
                    Qt.callLater(function() {
                        messageList.positionViewAtEnd()
                    })
                }
        }

        // Image preview strip (pending images to send)
        RowLayout {
            Layout.fillWidth: true
            spacing: 6
            visible: chatRoot.pendingImages.length > 0

            Repeater {
                model: chatRoot.pendingImages

                Rectangle {
                    width: 56
                    height: 56
                    radius: 6
                    color: "#1e1e1e"
                    border.color: "#4ec9b0"
                    border.width: 1

                    Image {
                        anchors.centerIn: parent
                        width: 48
                        height: 48
                        source: "data:image/png;base64," + modelData
                        fillMode: Image.PreserveAspectCrop
                        smooth: true
                    }

                    // Remove button
                    Rectangle {
                        anchors.top: parent.top
                        anchors.right: parent.right
                        anchors.margins: -4
                        width: 16
                        height: 16
                        radius: 8
                        color: "#d4534b"

                        Text {
                            anchors.centerIn: parent
                            text: "\u00D7"
                            color: "#ffffff"
                            font.pixelSize: 11
                            font.bold: true
                        }

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: {
                                var imgs = chatRoot.pendingImages.slice()
                                imgs.splice(index, 1)
                                chatRoot.pendingImages = imgs
                            }
                        }
                    }
                }
            }

            Item { Layout.fillWidth: true }

            Label {
                text: chatRoot.pendingImages.length + " image(s) attached"
                font.pixelSize: 10
                color: "#4ec9b0"
            }
        }

        // Input area with DropArea for image drag-drop
        Rectangle {
            id: inputContainer
            Layout.fillWidth: true
            Layout.minimumHeight: 44
            Layout.maximumHeight: Math.max(44, chatRoot.height * 0.35)
            Layout.preferredHeight: Math.min(
                Math.max(44, inputField.contentHeight + 24),
                Layout.maximumHeight
            )
            color: dropArea.containsDrag ? "#2a3a2a" : "#1e1e1e"
            radius: 6
            border.color: dropArea.containsDrag ? "#4ec9b0" : (inputField.activeFocus ? "#4ec9b0" : "#3e3e42")
            border.width: dropArea.containsDrag ? 2 : 1

            Behavior on Layout.preferredHeight {
                NumberAnimation { duration: 80; easing.type: Easing.OutCubic }
            }

            DropArea {
                id: dropArea
                anchors.fill: parent
                keys: ["text/uri-list"]

                onDropped: function(drop) {
                    if (drop.hasUrls) {
                        for (var i = 0; i < drop.urls.length; i++) {
                            var base64 = clipboardHelper.loadImageFileBase64(drop.urls[i].toString())
                            if (base64.length > 0) {
                                var imgs = chatRoot.pendingImages.slice()
                                imgs.push(base64)
                                chatRoot.pendingImages = imgs
                            }
                        }
                    }
                }
            }

            RowLayout {
                anchors.fill: parent
                anchors.margins: 6
                spacing: 8

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

                    TextArea {
                        id: inputField
                        placeholderText: chatRoot.pendingImages.length > 0
                            ? "Add a message about the image(s)..."
                            : "Type a message..."
                        placeholderTextColor: "#666666"
                        color: "#d4d4d4"
                        font.pixelSize: 13
                        wrapMode: TextArea.Wrap
                        background: null
                        enabled: !orchestrator.isProcessing

                        onTextChanged: {
                            // Notify idle scheduler of user activity
                            if (typeof idleScheduler !== 'undefined') {
                                idleScheduler.pauseImmediately()
                            }
                        }

                        Keys.onPressed: function(event) {
                            // Ctrl+V: check clipboard for image first
                            if (event.key === Qt.Key_V && (event.modifiers & Qt.ControlModifier)) {
                                if (clipboardHelper.hasImage()) {
                                    var base64 = clipboardHelper.getImageBase64()
                                    if (base64.length > 0) {
                                        var imgs = chatRoot.pendingImages.slice()
                                        imgs.push(base64)
                                        chatRoot.pendingImages = imgs
                                        event.accepted = true
                                        return
                                    }
                                }
                                // No image in clipboard — fall through to normal paste
                            }
                        }

                        Keys.onReturnPressed: function(event) {
                            if (event.modifiers & Qt.ShiftModifier) {
                                event.accepted = false
                            } else {
                                event.accepted = true
                                sendMessage()
                            }
                        }
                    }
                }

                Button {
                    id: sendButton
                    text: "Send"
                    enabled: (inputField.text.trim().length > 0 || chatRoot.pendingImages.length > 0)
                             && !orchestrator.isProcessing
                    Layout.alignment: Qt.AlignBottom
                    implicitHeight: 32
                    implicitWidth: 60

                    contentItem: Text {
                        text: parent.text
                        color: parent.enabled ? "#ffffff" : "#666666"
                        font.pixelSize: 12
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }

                    background: Rectangle {
                        color: parent.enabled ? "#4ec9b0" : "#3e3e42"
                        radius: 4
                    }

                    onClicked: sendMessage()
                }
            }
        }

        // Helper text
        RowLayout {
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            spacing: 6
            BusyIndicator {
                running: orchestrator.isProcessing
                Layout.preferredWidth: 16
                Layout.preferredHeight: 16
                visible: orchestrator.isProcessing
            }
            Label {
                text: orchestrator.isProcessing ? "DATAFORM is thinking..."
                        : "Enter to send | Shift+Enter new line | Ctrl+V paste image | Drop images"
                color: "#666666"
                font.pixelSize: 11
                Layout.fillWidth: true
            }
        }
    }

    // --- Connections ---

    property bool isStreaming: false

    Connections {
        target: orchestrator

        function onAssistantResponseReady(response) {
            if (chatRoot.isStreaming) {
                // Streaming populated the message — replace with clean final response
                // (strips think tags, tool calls, etc. that were visible during streaming)
                var lastIdx = messageModel.count - 1
                if (lastIdx >= 0) {
                    messageModel.setProperty(lastIdx, "content", response)
                }
                chatRoot.isStreaming = false
            } else {
                removeProcessingMessage()
                addMessage("agent", response)
            }
        }

        function onErrorOccurred(error) {
            chatRoot.isStreaming = false
            removeProcessingMessage()
            addMessage("system", "Error: " + error)
        }

        function onProcessingStarted() {
            addMessage("system", "Thinking...")
            messageModel.setProperty(messageModel.count - 1, "isProcessing", true)
        }

        function onConversationLoaded(conversationId, messages) {
            messageModel.clear()
            chatRoot.lastMessageDate = ""

            if (messages.length === 0) {
                addMessage("system", "DATAFORM online. New conversation started.")
            } else {
                // Prepend system greeting, then append all loaded messages
                // (C++ already provides date_header items and all required properties)
                messageModel.append({
                    "messageType": "system",
                    "content": "DATAFORM online. Conversation resumed.",
                    "timestamp": new Date().toLocaleTimeString(Qt.locale(), "HH:mm:ss"),
                    "isProcessing": false,
                    "imageData": "",
                    "isInitiated": false
                })
                for (var i = 0; i < messages.length; i++) {
                    messageModel.append(messages[i])
                }
                // Track last date from loaded messages so addMessage() won't double-insert headers
                if (messages.length > 0) {
                    var last = messages[messages.length - 1]
                    if (last.timestamp && last.timestamp !== "") {
                        chatRoot.lastMessageDate = new Date().toLocaleDateString(Qt.locale(), "yyyy-MM-dd")
                    }
                }
            }
        }

        function onCurrentConversationIdChanged() {
            if (orchestrator.currentConversationId < 0) {
                messageModel.clear()
                chatRoot.lastMessageDate = ""
                addMessage("system", "DATAFORM online. New conversation started.")
            }
        }
    }

    // Token streaming from local model (Phase 6)
    Connections {
        target: llmProvider

        function onTokenStreamed(token) {
            if (!chatRoot.isStreaming) {
                // First token - replace "Thinking..." with streaming message
                removeProcessingMessage()
                addMessage("agent", token)
                chatRoot.isStreaming = true
            } else {
                // Append token to the last message
                var lastIdx = messageModel.count - 1
                if (lastIdx >= 0) {
                    var item = messageModel.get(lastIdx)
                    if (!item) return
                    var current = item.content || ""
                    messageModel.setProperty(lastIdx, "content", current + token)
                }
            }
        }
    }

    // --- Functions ---

    function sendMessage() {
        var text = inputField.text.trim()
        var hasImages = chatRoot.pendingImages.length > 0

        if (text.length === 0 && !hasImages) return

        if (hasImages) {
            // Show message with images inline
            addMessageWithImages("user", text, chatRoot.pendingImages)
            orchestrator.handleUserMessageWithImages(text, chatRoot.pendingImages)
            chatRoot.pendingImages = []
        } else {
            addMessage("user", text)
            orchestrator.handleUserMessage(text)
        }

        inputField.text = ""
    }

    function addMessage(type, content) {
        var now = new Date()
        var dateStr = now.toLocaleDateString(Qt.locale(), "yyyy-MM-dd")
        if (dateStr !== chatRoot.lastMessageDate) {
            chatRoot.lastMessageDate = dateStr
            messageModel.append({
                "messageType": "date_header",
                "content": now.toLocaleDateString(Qt.locale(), "dddd, MMMM d, yyyy"),
                "timestamp": "",
                "isProcessing": false,
                "imageData": "",
                "isInitiated": false
            })
        }
        var timestamp = now.toLocaleTimeString(Qt.locale(), "HH:mm:ss")
        messageModel.append({
            "messageType": type,
            "content": content,
            "timestamp": timestamp,
            "isProcessing": false,
            "imageData": "",
            "isInitiated": false
        })
    }

    function addMessageWithImages(type, content, images) {
        var now = new Date()
        var timestamp = now.toLocaleTimeString(Qt.locale(), "HH:mm:ss")
        // Join images with a separator for ListModel string storage
        var imgStr = images.join("|IMG|")
        messageModel.append({
            "messageType": type,
            "content": content,
            "timestamp": timestamp,
            "isProcessing": false,
            "imageData": imgStr,
            "isInitiated": false
        })
    }

    function removeProcessingMessage() {
        for (var i = messageModel.count - 1; i >= 0; i--) {
            if (messageModel.get(i).isProcessing) {
                messageModel.remove(i)
                break
            }
        }
    }

    // Welcome message
    Component.onCompleted: {
        addMessage("system", "DATAFORM online. All systems nominal.")
    }
}
