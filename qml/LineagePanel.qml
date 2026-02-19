import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: lineageRoot
    color: "#1e1e1e"

    property var lineageData: typeof lineageTracker !== "undefined" ? lineageTracker.getLineageForQml() : []
    property var eraData: typeof lineageTracker !== "undefined" ? lineageTracker.getEraTimelineForQml() : []

    Connections {
        target: typeof lineageTracker !== "undefined" ? lineageTracker : null
        function onLineageChanged() {
            lineageData = lineageTracker.getLineageForQml()
            eraData = lineageTracker.getEraTimelineForQml()
        }
    }

    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 16

        // Title
        RowLayout {
            Layout.fillWidth: true
            spacing: 12

            Label {
                text: "Adapter Lineage"
                font.pixelSize: 24
                font.bold: true
                color: "#ffffff"
            }

            Item { Layout.fillWidth: true }

            Label {
                text: typeof lineageTracker !== "undefined"
                      ? lineageTracker.totalNodes + " adapters | Depth " + lineageTracker.currentDepth
                      : "No lineage data"
                font.pixelSize: 12
                color: "#888888"
            }
        }

        // Era comparison section
        Rectangle {
            Layout.fillWidth: true
            height: eraSection.height + 30
            color: "#252526"
            radius: 8
            visible: eraData.length > 0

            ColumnLayout {
                id: eraSection
                anchors.left: parent.left
                anchors.right: parent.right
                anchors.top: parent.top
                anchors.margins: 15
                spacing: 10

                Label {
                    text: "Era Progression"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#569cd6"
                }

                Repeater {
                    model: eraData

                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 12

                        Label {
                            text: modelData.eraLabel
                            font.pixelSize: 13
                            font.bold: true
                            color: "#569cd6"
                            Layout.preferredWidth: 60
                        }

                        // Score bar
                        Rectangle {
                            Layout.fillWidth: true
                            height: 20
                            radius: 3
                            color: "#1e1e1e"

                            Rectangle {
                                width: parent.width * Math.min(1.0, modelData.evalScore)
                                height: parent.height
                                radius: 3
                                color: modelData.evalScore >= 0.7 ? "#4ec9b0"
                                     : modelData.evalScore >= 0.4 ? "#dcdcaa" : "#ce9178"

                                Behavior on width { NumberAnimation { duration: 500 } }
                            }

                            Label {
                                anchors.centerIn: parent
                                text: modelData.evalScore.toFixed(2)
                                font.pixelSize: 11
                                color: "#ffffff"
                            }
                        }

                        Label {
                            text: "v" + modelData.adapterVersion
                            font.pixelSize: 11
                            color: "#4ec9b0"
                            Layout.preferredWidth: 40
                        }

                        Label {
                            text: modelData.cyclesCompleted + " cycles"
                            font.pixelSize: 11
                            color: "#888888"
                            Layout.preferredWidth: 60
                        }
                    }
                }
            }
        }

        // Lineage timeline
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#252526"
            radius: 8

            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 15
                spacing: 10

                Label {
                    text: "Adapter Timeline"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#4ec9b0"
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: lineageData
                    spacing: 2

                    delegate: Rectangle {
                        width: ListView.view.width
                        height: 52
                        color: modelData.status === "active" ? "#2a3a35" : "#1e1e1e"
                        radius: 4

                        RowLayout {
                            anchors.fill: parent
                            anchors.leftMargin: 12
                            anchors.rightMargin: 12
                            spacing: 12

                            // Version timeline dot
                            Rectangle {
                                width: 12
                                height: 12
                                radius: 6
                                color: modelData.status === "active" ? "#4ec9b0"
                                     : modelData.status === "archived" ? "#888888"
                                     : modelData.status === "rejected" ? "#ff6666"
                                     : "#dcdcaa"
                                Layout.alignment: Qt.AlignVCenter
                            }

                            // Version number
                            Label {
                                text: "v" + modelData.version
                                font.pixelSize: 14
                                font.bold: true
                                color: modelData.status === "active" ? "#4ec9b0" : "#cccccc"
                                Layout.preferredWidth: 40
                            }

                            // Origin badge
                            Rectangle {
                                width: originLabel.width + 12
                                height: 18
                                radius: 9
                                color: modelData.origin === "evolution" ? "#3e3020"
                                     : modelData.origin === "consolidation" ? "#203040"
                                     : modelData.origin === "reflection" ? "#302030"
                                     : "#303030"
                                visible: modelData.origin !== ""

                                Label {
                                    id: originLabel
                                    anchors.centerIn: parent
                                    text: modelData.origin || ""
                                    font.pixelSize: 10
                                    color: modelData.origin === "evolution" ? "#ce9178"
                                         : modelData.origin === "consolidation" ? "#569cd6"
                                         : modelData.origin === "reflection" ? "#c586c0"
                                         : "#888888"
                                }
                            }

                            // Eval score
                            Label {
                                text: modelData.evalScore > 0 ? modelData.evalScore.toFixed(2) : "--"
                                font.pixelSize: 12
                                color: modelData.evalScore >= 0.7 ? "#4ec9b0"
                                     : modelData.evalScore >= 0.4 ? "#dcdcaa" : "#888888"
                                Layout.preferredWidth: 40
                            }

                            // Loss
                            Label {
                                text: modelData.finalLoss > 0 ? "L:" + modelData.finalLoss.toFixed(3) : ""
                                font.pixelSize: 11
                                color: "#888888"
                                Layout.preferredWidth: 60
                            }

                            // Date
                            Label {
                                text: modelData.trainingDate
                                font.pixelSize: 11
                                color: "#666666"
                                Layout.fillWidth: true
                                horizontalAlignment: Text.AlignRight
                            }

                            // Child count
                            Label {
                                text: modelData.childCount > 0 ? modelData.childCount + " children" : ""
                                font.pixelSize: 10
                                color: "#555555"
                                Layout.preferredWidth: 70
                            }
                        }
                    }
                }
            }
        }
    }
}
