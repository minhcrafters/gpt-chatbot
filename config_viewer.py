import sys
import pickle
from PySide6.QtWidgets import QApplication, QMainWindow, QTreeWidgetItem, QFileDialog
from config_viewer_ui import Ui_MainWindow


class ConfigViewer(QMainWindow):
    def __init__(self):
        super(ConfigViewer, self).__init__()

        # Set up the UI
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Connect the tree widget to a double click event (for example, to load files)
        self.ui.treeWidget.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Load pickle data when the window opens
        self.load_pickle_data()

    def load_pickle_data(self):
        """Load pickle data and display it in the QTreeWidget."""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Pickle File", "", "Pickle Files (*.pickle *.pkl);;All Files (*)"
        )

        if file_name:
            try:
                with open(file_name, "rb") as f:
                    data = pickle.load(f)

                self.populate_tree(data)

            except Exception as e:
                print(f"Error loading pickle file: {e}")

    def populate_tree(self, data, parent=None):
        """
        Populate the QTreeWidget recursively with the contents of the pickle file.
        :param data: The data to display (can be dict, list, or primitive types).
        :param parent: The parent QTreeWidgetItem (default is root).
        """
        if parent is None:
            parent = self.ui.treeWidget.invisibleRootItem()

        if isinstance(data, dict):
            for key, value in data.items():
                item = QTreeWidgetItem(parent)
                item.setText(0, str(key))
                if isinstance(value, (dict, list)):
                    item.setText(1, "")
                    self.populate_tree(value, item)
                else:
                    item.setText(1, str(value))

        elif isinstance(data, list):
            for i, value in enumerate(data):
                item = QTreeWidgetItem(parent)
                item.setText(0, f"Item {i}")
                if isinstance(value, (dict, list)):
                    item.setText(1, "")
                    self.populate_tree(value, item)
                else:
                    item.setText(1, str(value))

        else:
            item = QTreeWidgetItem(parent)
            item.setText(0, "Value")
            item.setText(1, str(data))

    def on_item_double_clicked(self, item, column):
        """Handle double-click events on the tree widget."""
        print(f"Item double-clicked: {item.text(0)}, column: {column}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConfigViewer()
    window.show()
    sys.exit(app.exec())
