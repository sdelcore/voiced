"""DBusMenu implementation for tray icon context menu.

Implements com.canonical.dbusmenu D-Bus interface for displaying
transcription history in the system tray menu.
"""

import logging
from collections.abc import Callable

from dasbus.server.interface import dbus_interface, dbus_signal
from dasbus.server.template import InterfaceTemplate
from dasbus.typing import Bool, Int, List, Str, Structure, Tuple, Variant

logger = logging.getLogger(__name__)

DBUSMENU_INTERFACE = "com.canonical.dbusmenu"
DBUSMENU_PATH = "/MenuBar"

# Menu item IDs
MENU_ROOT_ID = 0
MENU_HISTORY_HEADER_ID = 1
MENU_SEPARATOR_ID = 2
MENU_QUIT_ID = 3
MENU_HISTORY_START_ID = 100  # History items start at 100


@dbus_interface(DBUSMENU_INTERFACE)
class DBusMenuInterface(InterfaceTemplate):
    """D-Bus interface for com.canonical.dbusmenu."""

    @dbus_signal
    def ItemsPropertiesUpdated(
        self,
        updated_props: List[Tuple[Int, Structure]],
        removed_props: List[Tuple[Int, List[Str]]],
    ):
        """Signal: Properties of items have changed."""
        pass

    @dbus_signal
    def LayoutUpdated(self, revision: Int, parent: Int):
        """Signal: Menu layout has changed."""
        pass

    @dbus_signal
    def ItemActivationRequested(self, id: Int, timestamp: Int):
        """Signal: Item activation requested via hotkey."""
        pass

    @property
    def Version(self) -> Int:
        """DBusMenu API version."""
        return 3

    @property
    def Status(self) -> Str:
        """Menu status: 'normal' or 'notice'."""
        return "normal"

    @property
    def TextDirection(self) -> Str:
        """Text direction: 'ltr' or 'rtl'."""
        return "ltr"

    @property
    def IconThemePath(self) -> List[Str]:
        """Additional icon theme paths."""
        return []

    def GetLayout(
        self,
        parent_id: Int,
        recursion_depth: Int,
        property_names: List[Str],
    ) -> Tuple[Int, Tuple[Int, Structure, List]]:
        """Get the menu layout structure."""
        return self.implementation.get_layout(parent_id, recursion_depth, property_names)

    def GetGroupProperties(
        self,
        ids: List[Int],
        property_names: List[Str],
    ) -> List[Tuple[Int, Structure]]:
        """Get properties for multiple items."""
        return self.implementation.get_group_properties(ids, property_names)

    def GetProperty(self, id: Int, name: Str) -> Variant:
        """Get a single property value."""
        return self.implementation.get_property(id, name)

    def Event(
        self,
        id: Int,
        event_id: Str,
        data: Variant,
        timestamp: Int,
    ) -> None:
        """Handle menu item events (clicked, hovered, etc.)."""
        self.implementation.handle_event(id, event_id, data, timestamp)

    def EventGroup(
        self,
        events: List[Tuple[Int, Str, Variant, Int]],
    ) -> List[Int]:
        """Handle multiple events at once."""
        errors = []
        for id, event_id, data, timestamp in events:
            try:
                self.implementation.handle_event(id, event_id, data, timestamp)
            except Exception:
                errors.append(id)
        return errors

    def AboutToShow(self, id: Int) -> Bool:
        """Called before a menu is shown."""
        return self.implementation.about_to_show(id)

    def AboutToShowGroup(self, ids: List[Int]) -> Tuple[List[Int], List[Int]]:
        """Called before multiple menus are shown."""
        updates_needed = []
        errors = []
        for menu_id in ids:
            try:
                if self.implementation.about_to_show(menu_id):
                    updates_needed.append(menu_id)
            except Exception:
                errors.append(menu_id)
        return (updates_needed, errors)


class DBusMenuImplementation:
    """Implementation of DBusMenu for transcription history."""

    def __init__(
        self,
        get_history: Callable[[], list],
        get_history_by_id: Callable[[int], object | None],
        get_revision: Callable[[], int],
        on_copy_history: Callable[[str], None],
        on_quit: Callable[[], None],
    ):
        """Initialize the menu implementation.

        Args:
            get_history: Callback to get history entries
            get_history_by_id: Callback to get specific history entry
            get_revision: Callback to get current revision number
            on_copy_history: Callback when history item is selected
            on_quit: Callback when quit is selected
        """
        self._get_history = get_history
        self._get_history_by_id = get_history_by_id
        self._get_revision = get_revision
        self._on_copy_history = on_copy_history
        self._on_quit = on_quit
        self._interface: DBusMenuInterface | None = None

    def _build_item_properties(self, item_id: int) -> dict:
        """Build properties dict for a menu item."""
        if item_id == MENU_ROOT_ID:
            return {"children-display": "submenu"}
        elif item_id == MENU_HISTORY_HEADER_ID:
            return {
                "type": "standard",
                "label": "Recent Transcriptions",
                "enabled": False,
            }
        elif item_id == MENU_SEPARATOR_ID:
            return {"type": "separator"}
        elif item_id == MENU_QUIT_ID:
            return {
                "type": "standard",
                "label": "Quit",
                "enabled": True,
            }
        elif item_id >= MENU_HISTORY_START_ID:
            # History item - get from history
            history_index = item_id - MENU_HISTORY_START_ID
            entries = self._get_history()
            if 0 <= history_index < len(entries):
                entry = entries[history_index]
                return {
                    "type": "standard",
                    "label": entry.preview,
                    "enabled": True,
                }

        return {}

    def get_layout(
        self,
        parent_id: int,
        recursion_depth: int,
        property_names: list[str],
    ) -> tuple:
        """Return menu layout starting from parent_id."""
        revision = self._get_revision()

        if parent_id == MENU_ROOT_ID:
            # Build complete menu structure
            children = []

            # Header
            children.append(
                self._build_layout_item(MENU_HISTORY_HEADER_ID, recursion_depth - 1, property_names)
            )

            # History items
            entries = self._get_history()
            for i in range(len(entries)):
                item_id = MENU_HISTORY_START_ID + i
                children.append(
                    self._build_layout_item(item_id, recursion_depth - 1, property_names)
                )

            # Separator (only if we have history)
            if entries:
                children.append(
                    self._build_layout_item(MENU_SEPARATOR_ID, recursion_depth - 1, property_names)
                )

            # Quit
            children.append(
                self._build_layout_item(MENU_QUIT_ID, recursion_depth - 1, property_names)
            )

            root_props = self._build_item_properties(MENU_ROOT_ID)
            return (revision, (MENU_ROOT_ID, root_props, children))

        # Non-root item requested
        props = self._build_item_properties(parent_id)
        return (revision, (parent_id, props, []))

    def _build_layout_item(
        self,
        item_id: int,
        remaining_depth: int,
        property_names: list[str],
    ) -> tuple:
        """Build a single layout item tuple."""
        props = self._build_item_properties(item_id)

        # Filter properties if specific ones requested
        if property_names:
            props = {k: v for k, v in props.items() if k in property_names}

        children: list = []  # Our menu items don't have submenus
        return (item_id, props, children)

    def get_group_properties(
        self,
        ids: list[int],
        property_names: list[str],
    ) -> list:
        """Get properties for multiple items."""
        result = []
        for item_id in ids:
            props = self._build_item_properties(item_id)
            if property_names:
                props = {k: v for k, v in props.items() if k in property_names}
            result.append((item_id, props))
        return result

    def get_property(self, item_id: int, name: str) -> object:
        """Get a single property."""
        props = self._build_item_properties(item_id)
        return props.get(name)

    def handle_event(
        self,
        item_id: int,
        event_id: str,
        data: object,
        timestamp: int,
    ) -> None:
        """Handle menu item events."""
        if event_id != "clicked":
            return

        logger.debug(f"Menu item {item_id} clicked")

        if item_id == MENU_QUIT_ID:
            self._on_quit()
        elif item_id >= MENU_HISTORY_START_ID:
            # History item clicked - copy to clipboard
            history_index = item_id - MENU_HISTORY_START_ID
            entries = self._get_history()
            if 0 <= history_index < len(entries):
                entry = entries[history_index]
                logger.info(f"Copying history item {entry.id} to clipboard")
                self._on_copy_history(entry.text)

    def about_to_show(self, item_id: int) -> bool:
        """Called before showing a menu. Return True if layout needs update."""
        # We could refresh history here if needed
        return False

    def emit_layout_updated(self, revision: int, parent: int = 0) -> None:
        """Emit LayoutUpdated signal."""
        if self._interface:
            try:
                self._interface.LayoutUpdated(revision, parent)
            except Exception as e:
                logger.warning(f"Failed to emit LayoutUpdated: {e}")
