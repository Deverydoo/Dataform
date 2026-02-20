import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: sidebar
    color: "#252526"
    radius: 6

    property bool collapsed: false
    property int expandedWidth: 250
    property int collapsedWidth: 36

    implicitWidth: collapsed ? collapsedWidth : expandedWidth
    Layout.minimumWidth: collapsed ? collapsedWidth : expandedWidth
    Layout.maximumWidth: collapsed ? collapsedWidth : expandedWidth

    Behavior on implicitWidth {
        NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
    }
    Behavior on Layout.minimumWidth {
        NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
    }
    Behavior on Layout.maximumWidth {
        NumberAnimation { duration: 150; easing.type: Easing.OutQuad }
    }

    // Conversation list data
    ListModel {
        id: conversationModel
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: collapsed ? 2 : 8
        spacing: 6
        visible: !collapsed

        // Header
        RowLayout {
            Layout.fillWidth: true
            spacing: 6

            Label {
                text: "Conversations"
                font.pixelSize: 13
                font.bold: true
                color: "#cccccc"
                Layout.fillWidth: true
            }

            // Collapse button
            Button {
                implicitWidth: 24
                implicitHeight: 24
                flat: true

                contentItem: Text {
                    text: "\u25C0"
                    color: parent.hovered ? "#ffffff" : "#888888"
                    font.pixelSize: 11
                    horizontalAlignment: Text.AlignHCenter
                    verticalAlignment: Text.AlignVCenter
                }
                background: Rectangle {
                    color: parent.hovered ? "#3e3e42" : "transparent"
                    radius: 4
                }

                onClicked: sidebar.collapsed = true
            }
        }

        // New Chat button
        Button {
            Layout.fillWidth: true
            implicitHeight: 32

            contentItem: RowLayout {
                spacing: 6
                Text {
                    text: "+"
                    color: "#4ec9b0"
                    font.pixelSize: 14
                    font.bold: true
                }
                Text {
                    text: "New Chat"
                    color: "#d4d4d4"
                    font.pixelSize: 12
                    Layout.fillWidth: true
                }
            }

            background: Rectangle {
                color: parent.hovered ? "#3e3e42" : "#2d2d30"
                radius: 4
                border.color: "#3e3e42"
                border.width: 1
            }

            onClicked: {
                orchestrator.startNewConversation()
            }
        }

        // Search field
        TextField {
            id: searchField
            Layout.fillWidth: true
            placeholderText: "Search conversations..."
            placeholderTextColor: "#666666"
            color: "#d4d4d4"
            font.pixelSize: 12
            background: Rectangle {
                color: "#1e1e1e"
                radius: 4
                border.color: searchField.activeFocus ? "#4ec9b0" : "#3e3e42"
                border.width: 1
            }

            onTextChanged: {
                if (text.length >= 2) {
                    searchDebounceTimer.restart()
                } else {
                    searchResultsModel.clear()
                    searchResultsPane.visible = false
                }
            }

            Keys.onEscapePressed: {
                text = ""
                searchResultsModel.clear()
                searchResultsPane.visible = false
                focus = false
            }
        }

        Timer {
            id: searchDebounceTimer
            interval: 300
            onTriggered: {
                searchResultsModel.clear()
                var results = memoryStore.searchConversationsForQml(searchField.text, 20)
                if (!results) results = []
                for (var i = 0; i < results.length; i++) {
                    searchResultsModel.append(results[i])
                }
                searchResultsPane.visible = searchResultsModel.count > 0
            }
        }

        // Search results
        ColumnLayout {
            id: searchResultsPane
            Layout.fillWidth: true
            spacing: 2
            visible: false

            Label {
                text: searchResultsModel.count + " result(s)"
                font.pixelSize: 10
                color: "#4ec9b0"
            }

            ListModel { id: searchResultsModel }

            Repeater {
                model: searchResultsModel

                Rectangle {
                    Layout.fillWidth: true
                    height: searchResultCol.height + 10
                    radius: 4
                    color: searchResultArea.containsMouse ? "#2d2d30" : "#1e1e1e"
                    border.color: "#3e3e42"
                    border.width: 1

                    ColumnLayout {
                        id: searchResultCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 6
                        spacing: 2

                        Label {
                            Layout.fillWidth: true
                            text: model.userText || ""
                            font.pixelSize: 11
                            color: "#d4d4d4"
                            elide: Text.ElideRight
                            maximumLineCount: 1
                        }

                        Label {
                            Layout.fillWidth: true
                            text: model.assistantText || ""
                            font.pixelSize: 10
                            color: "#888888"
                            elide: Text.ElideRight
                            maximumLineCount: 1
                        }
                    }

                    MouseArea {
                        id: searchResultArea
                        anchors.fill: parent
                        hoverEnabled: true
                        onClicked: {
                            var convId = model.conversationId
                            if (convId > 0) {
                                orchestrator.loadConversation(convId)
                            }
                            searchField.text = ""
                            searchResultsModel.clear()
                            searchResultsPane.visible = false
                        }
                    }
                }
            }

            // Separator after search results
            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#3e3e42"
            }
        }

        // Separator
        Rectangle {
            Layout.fillWidth: true
            height: 1
            color: "#3e3e42"
            visible: !searchResultsPane.visible
        }

        // Thought cards section
        ColumnLayout {
            Layout.fillWidth: true
            spacing: 8
            visible: typeof thoughtEngine !== "undefined" && thoughtEngine.pendingCount > 0

            Label {
                text: "DATAFORM wants to discuss"
                font.pixelSize: 10
                font.bold: true
                color: "#569cd6"
                Layout.fillWidth: true
            }

            Repeater {
                model: {
                    // Depend on pendingCount so this re-evaluates when thoughts change
                    var count = (typeof thoughtEngine !== "undefined") ? thoughtEngine.pendingCount : 0
                    if (count <= 0) return []
                    return thoughtEngine.getPendingThoughtsForQml(3)
                }

                Rectangle {
                    Layout.fillWidth: true
                    height: thoughtCardCol.height + 16
                    radius: 4
                    color: "#2a2d3e"
                    border.color: "#569cd6"
                    border.width: 1

                    ColumnLayout {
                        id: thoughtCardCol
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 8
                        spacing: 4

                        // Type badge row
                        Rectangle {
                            width: typeBadge.width + 8
                            height: 16
                            radius: 3
                            color: {
                                if (modelData.type === "research_proposal") return "#1e4a1e"
                                if (modelData.type === "curiosity_overflow") return "#4a3a1e"
                                if (modelData.type === "evolution_observation") return "#1e3a4a"
                                if (modelData.type === "news_insight") return "#4a1e4a"
                                if (modelData.type === "reminder") return "#5a3a1e"
                                if (modelData.type === "goal_checkin") return "#2d5a2d"
                                if (modelData.type === "daily_digest") return "#3a3a5a"
                                if (modelData.type === "mood_pattern") return "#4a2d4a"
                                if (modelData.type === "lesson_ready") return "#1e4a3a"
                                return "#3e3e42"
                            }

                            Label {
                                id: typeBadge
                                anchors.centerIn: parent
                                text: {
                                    if (modelData.type === "research_proposal") return "research"
                                    if (modelData.type === "curiosity_overflow") return "curious"
                                    if (modelData.type === "evolution_observation") return "training"
                                    if (modelData.type === "training_observation") return "training"
                                    if (modelData.type === "news_insight") return "news"
                                    if (modelData.type === "reminder") return "reminder"
                                    if (modelData.type === "goal_checkin") return "goal"
                                    if (modelData.type === "daily_digest") return "digest"
                                    if (modelData.type === "mood_pattern") return "mood"
                                    if (modelData.type === "lesson_ready") return "lesson"
                                    return modelData.type || ""
                                }
                                font.pixelSize: 9
                                color: "#cccccc"
                            }
                        }

                        // Thought title
                        Label {
                            Layout.fillWidth: true
                            text: modelData.title || ""
                            font.pixelSize: 11
                            color: "#d4d4d4"
                            elide: Text.ElideRight
                            maximumLineCount: 2
                            wrapMode: Text.WordWrap
                        }

                        // Action buttons row
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            // Discuss button
                            Button {
                                Layout.fillWidth: true
                                implicitHeight: 24
                                flat: true

                                contentItem: Text {
                                    text: "Discuss"
                                    color: parent.hovered ? "#ffffff" : "#8ab4f8"
                                    font.pixelSize: 10
                                    font.bold: true
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle {
                                    color: parent.hovered ? "#2a4a7a" : "#1e2a4f"
                                    radius: 3
                                    border.color: "#569cd6"
                                    border.width: 1
                                }

                                onClicked: {
                                    orchestrator.startProactiveConversation(modelData.id)
                                }
                            }

                            // Dismiss button
                            Button {
                                implicitWidth: 60
                                implicitHeight: 24
                                flat: true

                                contentItem: Text {
                                    text: "Dismiss"
                                    color: parent.hovered ? "#ff6b6b" : "#888888"
                                    font.pixelSize: 10
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle {
                                    color: parent.hovered ? "#3e2020" : "#2d2d30"
                                    radius: 3
                                    border.color: parent.hovered ? "#ff6b6b" : "#3e3e42"
                                    border.width: 1
                                }

                                onClicked: {
                                    thoughtEngine.dismissThought(modelData.id)
                                }
                            }
                        }
                    }
                }
            }

            // Separator after thought cards
            Rectangle {
                Layout.fillWidth: true
                height: 1
                color: "#3e3e42"
            }
        }

        // Conversation list
        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true

            ListView {
                id: conversationList
                model: conversationModel
                spacing: 2

                delegate: Rectangle {
                    width: conversationList.width
                    height: delegateColumn.height + 12
                    radius: 4
                    color: {
                        if (model.id === orchestrator.currentConversationId)
                            return model.isProactive ? "#1e2a4f" : "#1e3a5f"
                        if (delegateArea.containsMouse)
                            return "#2d2d30"
                        return "transparent"
                    }
                    border.color: model.isProactive ? "#569cd6" : "transparent"
                    border.width: model.isProactive ? 1 : 0

                    ColumnLayout {
                        id: delegateColumn
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.verticalCenter: parent.verticalCenter
                        anchors.margins: 8
                        spacing: 2

                        // Title row with proactive indicator
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 4

                            // DATAFORM-initiated indicator
                            Rectangle {
                                visible: model.isProactive
                                width: 6
                                height: 6
                                radius: 3
                                color: "#569cd6"
                                Layout.alignment: Qt.AlignVCenter
                            }

                            Label {
                                Layout.fillWidth: true
                                text: model.title || "Untitled"
                                font.pixelSize: 12
                                color: {
                                    if (model.id === orchestrator.currentConversationId)
                                        return "#ffffff"
                                    return model.isProactive ? "#8ab4f8" : "#cccccc"
                                }
                                elide: Text.ElideRight
                                maximumLineCount: 1
                            }
                        }

                        // Metadata row
                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 6

                            Label {
                                text: formatRelativeTime(model.lastActivityTs)
                                font.pixelSize: 10
                                color: "#666666"
                            }

                            Item { Layout.fillWidth: true }

                            // Message count badge
                            Rectangle {
                                visible: model.messageCount > 0
                                width: countLabel.width + 8
                                height: 14
                                radius: 7
                                color: "#3e3e42"

                                Label {
                                    id: countLabel
                                    anchors.centerIn: parent
                                    text: model.messageCount
                                    font.pixelSize: 9
                                    color: "#888888"
                                }
                            }
                        }
                    }

                    MouseArea {
                        id: delegateArea
                        anchors.fill: parent
                        hoverEnabled: true
                        acceptedButtons: Qt.LeftButton | Qt.RightButton

                        onClicked: function(mouse) {
                            if (mouse.button === Qt.RightButton) {
                                contextMenu.conversationId = model.id
                                contextMenu.conversationTitle = model.title
                                contextMenu.popup()
                            } else {
                                orchestrator.loadConversation(model.id)
                            }
                        }
                    }
                }
            }
        }
    }

    // Collapsed state: just show expand button
    Button {
        anchors.centerIn: parent
        visible: collapsed
        implicitWidth: 28
        implicitHeight: 28
        flat: true

        contentItem: Text {
            text: "\u25B6"
            color: parent.hovered ? "#ffffff" : "#888888"
            font.pixelSize: 11
            horizontalAlignment: Text.AlignHCenter
            verticalAlignment: Text.AlignVCenter
        }
        background: Rectangle {
            color: parent.hovered ? "#3e3e42" : "transparent"
            radius: 4
        }

        onClicked: sidebar.collapsed = false
    }

    // Context menu for conversations
    Menu {
        id: contextMenu
        property int conversationId: -1
        property string conversationTitle: ""

        background: Rectangle {
            implicitWidth: 140
            color: "#2d2d30"
            border.color: "#3e3e42"
            border.width: 1
            radius: 4
        }

        MenuItem {
            text: "Rename"
            height: 32
            onTriggered: {
                renameDialog.conversationId = contextMenu.conversationId
                renameDialog.currentTitle = contextMenu.conversationTitle
                renameField.text = contextMenu.conversationTitle
                renameDialog.open()
            }

            contentItem: Text {
                text: parent.text
                color: parent.hovered ? "#ffffff" : "#cccccc"
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
                leftPadding: 12
            }
            background: Rectangle {
                color: parent.hovered ? "#3e3e42" : "transparent"
            }
        }

        MenuItem {
            text: "Delete"
            height: 32
            onTriggered: {
                var cid = contextMenu.conversationId
                if (cid >= 0) {
                    var wasActive = (cid === orchestrator.currentConversationId)
                    memoryStore.deleteConversation(cid)
                    refreshConversations()
                    if (wasActive) orchestrator.startNewConversation()
                }
            }

            contentItem: Text {
                text: parent.text
                color: parent.hovered ? "#ff6b6b" : "#ff6666"
                font.pixelSize: 12
                verticalAlignment: Text.AlignVCenter
                leftPadding: 12
            }
            background: Rectangle {
                color: parent.hovered ? "#3e3e42" : "transparent"
            }
        }
    }

    // Rename dialog
    Dialog {
        id: renameDialog
        property int conversationId: -1
        property string currentTitle: ""

        title: "Rename Conversation"
        anchors.centerIn: Overlay.overlay
        modal: true
        width: 300

        background: Rectangle {
            color: "#2d2d30"
            border.color: "#3e3e42"
            border.width: 1
            radius: 6
        }

        header: Label {
            text: "Rename Conversation"
            font.pixelSize: 14
            font.bold: true
            color: "#cccccc"
            padding: 12
        }

        contentItem: TextField {
            id: renameField
            color: "#d4d4d4"
            font.pixelSize: 13
            placeholderText: "Enter new title..."
            placeholderTextColor: "#666666"
            background: Rectangle {
                color: "#1e1e1e"
                radius: 4
                border.color: renameField.activeFocus ? "#4ec9b0" : "#3e3e42"
                border.width: 1
            }
        }

        standardButtons: Dialog.Ok | Dialog.Cancel

        onAccepted: {
            if (renameField.text.trim().length > 0) {
                memoryStore.updateConversationTitle(conversationId, renameField.text.trim())
            }
        }
    }


    // --- Connections ---

    Connections {
        target: memoryStore

        function onConversationListChanged() {
            refreshConversations()
        }

        function onEpisodeInserted(id) {
            refreshConversations()
        }
    }

    Connections {
        target: orchestrator

        function onConversationLoaded(id, messages) {
            // Sidebar just needs to update highlight — ListView handles this
            // via currentConversationId binding
        }
    }

    Connections {
        target: typeof thoughtEngine !== "undefined" ? thoughtEngine : null
        enabled: typeof thoughtEngine !== "undefined"

        function onPendingCountChanged() {
            refreshThoughts()
        }
    }

    // --- Functions ---

    function refreshConversations() {
        var conversations = memoryStore.getConversationsForQml(50)
        conversationModel.clear()
        for (var i = 0; i < conversations.length; i++) {
            conversationModel.append(conversations[i])
        }
    }

    function formatRelativeTime(tsString) {
        if (!tsString) return ""
        var ts = new Date(tsString.replace(" ", "T"))
        var now = new Date()
        var diffMs = now - ts
        var diffMin = Math.floor(diffMs / 60000)
        var diffHr = Math.floor(diffMs / 3600000)
        var diffDay = Math.floor(diffMs / 86400000)

        if (diffMin < 1) return "now"
        if (diffMin < 60) return diffMin + "m ago"
        if (diffHr < 24) return diffHr + "h ago"
        if (diffDay < 7) return diffDay + "d ago"
        return ts.toLocaleDateString(Qt.locale(), "MMM d")
    }

    function refreshThoughts() {
        // Thought cards use Repeater with direct model binding,
        // pendingCountChanged triggers re-evaluation
    }

    Component.onCompleted: {
        refreshConversations()
    }
}
