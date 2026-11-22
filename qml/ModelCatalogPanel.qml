import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Popup {
    id: catalogRoot

    modal: true
    focus: true
    closePolicy: Popup.CloseOnEscape
    anchors.centerIn: Overlay.overlay
    width: Math.min(parent ? parent.width - 60 : 900, 900)
    height: Math.min(parent ? parent.height - 60 : 700, 700)
    padding: 0

    // Bumped whenever something on disk changes, to re-run the model queries.
    property int refreshTick: 0
    property string roleFilter: "all"
    property string pendingDownloadId: ""

    background: Rectangle {
        color: "#1e1e1e"
        radius: 10
        border.color: "#3c3c3c"
        border.width: 1
    }

    Overlay.modal: Rectangle { color: "#cc000000" }

    signal modelInstalled(string path)

    Connections {
        target: typeof modelCatalog !== "undefined" ? modelCatalog : null

        function onDownloadFinished(id, path) {
            catalogRoot.refreshTick++
            catalogRoot.modelInstalled(path)
        }
        function onDownloadFailed(id, error) {
            catalogRoot.refreshTick++
        }
        function onOllamaInstallFinished(id, tag) {
            ollamaResult.text = "Created Ollama model `" + tag + "`. Pick it in the Model dropdown."
            ollamaResult.color = "#4CAF50"
        }
        function onOllamaInstallFailed(id, error) {
            ollamaResult.text = "Ollama install failed: " + error
            ollamaResult.color = "#F44336"
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 14

        // === Header ===
        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            ColumnLayout {
                Layout.fillWidth: true
                spacing: 2
                Label {
                    text: "Model Catalog"
                    font.pixelSize: 20
                    font.bold: true
                    color: "#ffffff"
                }
                Label {
                    text: "Curated GGUF builds. Downloads are resumable and land in models/."
                    font.pixelSize: 11
                    color: "#888888"
                }
            }

            Button {
                text: "Close"
                enabled: typeof modelCatalog === "undefined" || !modelCatalog.isDownloading
                onClicked: catalogRoot.close()
            }

            Label {
                visible: typeof modelCatalog !== "undefined" && modelCatalog.isDownloading
                text: "Cancel the download to close"
                font.pixelSize: 10
                color: "#666666"
            }
        }

        // === Role filter ===
        RowLayout {
            Layout.fillWidth: true
            spacing: 8

            Repeater {
                model: [
                    { key: "all", label: "All" },
                    { key: "chat", label: "Chat brain" },
                    { key: "background", label: "Background brain" }
                ]
                delegate: Button {
                    text: modelData.label
                    checkable: true
                    checked: catalogRoot.roleFilter === modelData.key
                    onClicked: catalogRoot.roleFilter = modelData.key
                    contentItem: Label {
                        text: parent.text
                        font.pixelSize: 12
                        color: parent.checked ? "#1e1e1e" : "#cccccc"
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                    background: Rectangle {
                        radius: 6
                        color: parent.checked ? "#4ec9b0" : "#2d2d30"
                        border.color: "#3c3c3c"
                    }
                }
            }

            Item { Layout.fillWidth: true }

            Label {
                text: "Chat brain = user-facing conversation · Background brain = trait/sentiment/JSON tasks"
                font.pixelSize: 10
                color: "#666666"
            }
        }

        // === Model list ===
        Flickable {
            Layout.fillWidth: true
            Layout.fillHeight: true
            contentWidth: width
            contentHeight: cardsColumn.height + 20
            clip: true
            boundsBehavior: Flickable.StopAtBounds
            ScrollBar.vertical: ScrollBar { policy: ScrollBar.AlwaysOff }

            ColumnLayout {
                id: cardsColumn
                width: parent.width
                spacing: 12

                Repeater {
                    model: {
                        catalogRoot.refreshTick // dependency, forces re-query
                        if (typeof modelCatalog === "undefined") return []
                        return modelCatalog.modelsForRole(catalogRoot.roleFilter)
                    }

                    delegate: Rectangle {
                        Layout.fillWidth: true
                        implicitHeight: cardCol.implicitHeight + 28
                        color: "#252526"
                        radius: 8
                        border.width: 1
                        border.color: modelData.installed ? "#4ec9b0" : "#333333"

                        readonly property bool busy: typeof modelCatalog !== "undefined"
                                                     && modelCatalog.downloadingId === modelData.id

                        ColumnLayout {
                            id: cardCol
                            anchors.left: parent.left
                            anchors.right: parent.right
                            anchors.top: parent.top
                            anchors.margins: 14
                            spacing: 8

                            // Title row
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Label {
                                    text: modelData.displayName
                                    font.pixelSize: 15
                                    font.bold: true
                                    color: "#ffffff"
                                }

                                // Quant badge
                                Rectangle {
                                    implicitWidth: quantLabel.implicitWidth + 12
                                    implicitHeight: 18
                                    radius: 9
                                    color: "#37373d"
                                    Label {
                                        id: quantLabel
                                        anchors.centerIn: parent
                                        text: modelData.quant
                                        font.pixelSize: 10
                                        color: "#cccccc"
                                    }
                                }

                                // Alignment badge
                                Rectangle {
                                    visible: modelData.alignment !== "instruct"
                                    implicitWidth: alignLabel.implicitWidth + 12
                                    implicitHeight: 18
                                    radius: 9
                                    color: modelData.alignment === "roleplay" ? "#4a2f5c" : "#5c3a2f"
                                    Label {
                                        id: alignLabel
                                        anchors.centerIn: parent
                                        text: modelData.alignment
                                        font.pixelSize: 10
                                        color: modelData.alignment === "roleplay" ? "#d0a9e8" : "#e8bfa9"
                                    }
                                }

                                // Role badge
                                Rectangle {
                                    implicitWidth: roleLabel.implicitWidth + 12
                                    implicitHeight: 18
                                    radius: 9
                                    color: "#2f3f5c"
                                    Label {
                                        id: roleLabel
                                        anchors.centerIn: parent
                                        text: modelData.role === "both" ? "chat + background" : modelData.role
                                        font.pixelSize: 10
                                        color: "#a9c9e8"
                                    }
                                }

                                Item { Layout.fillWidth: true }

                                Label {
                                    visible: modelData.installed
                                    text: "Installed"
                                    font.pixelSize: 11
                                    font.bold: true
                                    color: "#4ec9b0"
                                }
                            }

                            // Spec line
                            Label {
                                text: modelData.family + " · " + modelData.approxSize
                                      + " on disk · ~" + modelData.vramGb + " GB VRAM at "
                                      + (modelData.contextLength / 1024) + "K context · " + modelData.license
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: modelData.description
                                font.pixelSize: 12
                                color: "#bbbbbb"
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                            }

                            Label {
                                text: modelData.repoId + " / " + modelData.fileName
                                font.pixelSize: 10
                                font.family: "Consolas"
                                color: "#5a5a5a"
                                Layout.fillWidth: true
                                elide: Text.ElideMiddle
                            }

                            // Action row
                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Button {
                                    visible: !modelData.installed
                                    enabled: typeof modelCatalog !== "undefined" && !modelCatalog.isDownloading
                                    text: modelData.partialBytes > 0
                                          ? "Resume (" + modelCatalog.formatBytes(modelData.partialBytes) + " done)"
                                          : "Download"
                                    onClicked: catalogRoot.requestDownload(modelData)
                                }

                                Button {
                                    visible: modelData.installed && modelData.role !== "background"
                                    text: "Register with Ollama"
                                    ToolTip.visible: hovered
                                    ToolTip.text: "Runs `ollama create " + modelData.id
                                                  + "` so it appears in the chat Model dropdown"
                                    onClicked: {
                                        ollamaResult.text = "Running ollama create…"
                                        ollamaResult.color = "#888888"
                                        modelCatalog.installToOllama(modelData.id)
                                    }
                                }

                                Button {
                                    visible: modelData.installed && modelData.role !== "chat"
                                    text: "Use as background brain"
                                    onClicked: {
                                        settingsManager.backgroundModelPath =
                                            "models/background_llm/" + modelData.fileName
                                        settingsManager.backgroundModelEnabled = true
                                        ollamaResult.text = "Background brain set to " + modelData.fileName
                                                            + " — restart to load it."
                                        ollamaResult.color = "#4ec9b0"
                                    }
                                }

                                Label {
                                    visible: modelData.ollamaTag !== ""
                                    text: "or: ollama pull " + modelData.ollamaTag
                                    font.pixelSize: 10
                                    font.family: "Consolas"
                                    color: "#666666"
                                }

                                Item { Layout.fillWidth: true }

                                Button {
                                    visible: modelData.installed || modelData.partialBytes > 0
                                    enabled: typeof modelCatalog !== "undefined" && !modelCatalog.isDownloading
                                    text: modelData.installed ? "Delete" : "Discard partial"
                                    onClicked: {
                                        modelCatalog.removeDownload(modelData.id)
                                        catalogRoot.refreshTick++
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // === Active download ===
        Rectangle {
            Layout.fillWidth: true
            visible: typeof modelCatalog !== "undefined" && modelCatalog.isDownloading
            implicitHeight: dlCol.implicitHeight + 24
            color: "#252526"
            radius: 8

            ColumnLayout {
                id: dlCol
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 12
                spacing: 8

                RowLayout {
                    Layout.fillWidth: true
                    Label {
                        text: typeof modelCatalog !== "undefined" ? modelCatalog.statusText : ""
                        font.pixelSize: 12
                        color: "#cccccc"
                        Layout.fillWidth: true
                        elide: Text.ElideRight
                    }
                    Label {
                        text: typeof modelCatalog !== "undefined"
                              ? Math.round(modelCatalog.progress * 100) + "%" : ""
                        font.pixelSize: 12
                        font.bold: true
                        color: "#4ec9b0"
                    }
                    Button {
                        text: "Cancel"
                        onClicked: modelCatalog.cancelDownload()
                    }
                }

                ProgressBar {
                    Layout.fillWidth: true
                    from: 0
                    to: 1
                    value: typeof modelCatalog !== "undefined" ? modelCatalog.progress : 0
                }
            }
        }

        Label {
            id: ollamaResult
            Layout.fillWidth: true
            text: ""
            font.pixelSize: 11
            color: "#888888"
            wrapMode: Text.WordWrap
            visible: text !== ""
        }
    }

    // Gate the first uncensored download behind a single acknowledgement.
    function requestDownload(entry) {
        if (entry.uncensored && !modelCatalog.uncensoredAcknowledged) {
            catalogRoot.pendingDownloadId = entry.id
            uncensoredDialog.modelName = entry.displayName
            uncensoredDialog.open()
            return
        }
        modelCatalog.startDownload(entry.id)
    }

    Dialog {
        id: uncensoredDialog
        anchors.centerIn: Overlay.overlay
        modal: true
        width: 460
        title: "Reduced-alignment model"

        property string modelName: ""

        contentItem: ColumnLayout {
            spacing: 10
            Label {
                text: uncensoredDialog.modelName + " has had its refusal behaviour removed or "
                      + "heavily reduced. It will follow instructions that a standard assistant "
                      + "model would decline, and its factual claims are not filtered."
                wrapMode: Text.WordWrap
                color: "#cccccc"
                font.pixelSize: 12
                Layout.fillWidth: true
                Layout.maximumWidth: 420
            }
            Label {
                text: "It runs entirely on this machine — nothing it generates leaves your computer. "
                      + "You are asked once; later downloads of this kind will not prompt."
                wrapMode: Text.WordWrap
                color: "#888888"
                font.pixelSize: 11
                Layout.fillWidth: true
                Layout.maximumWidth: 420
            }
        }

        standardButtons: Dialog.Cancel | Dialog.Ok

        onAccepted: {
            modelCatalog.acknowledgeUncensored()
            if (catalogRoot.pendingDownloadId !== "")
                modelCatalog.startDownload(catalogRoot.pendingDownloadId)
            catalogRoot.pendingDownloadId = ""
        }
        onRejected: catalogRoot.pendingDownloadId = ""
    }
}
