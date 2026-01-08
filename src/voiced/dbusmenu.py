"""DBusMenu implementation for tray icon context menu.

Implements com.canonical.dbusmenu D-Bus interface for displaying
transcription history in the system tray menu.

Uses Gio.DBus directly to ensure correct D-Bus signatures.
"""

import logging
from collections.abc import Callable

from gi.repository import Gio, GLib

logger = logging.getLogger(__name__)

DBUSMENU_INTERFACE = "com.canonical.dbusmenu"
DBUSMENU_PATH = "/MenuBar"

# Menu item IDs
MENU_ROOT_ID = 0
MENU_HISTORY_HEADER_ID = 1
MENU_SEPARATOR_ID = 2
MENU_QUIT_ID = 3
MENU_HISTORY_START_ID = 100  # History items start at 100

# D-Bus introspection XML with correct signatures
DBUSMENU_XML = """
<node>
  <interface name="com.canonical.dbusmenu">
    <method name="GetLayout">
      <arg type="i" name="parentId" direction="in"/>
      <arg type="i" name="recursionDepth" direction="in"/>
      <arg type="as" name="propertyNames" direction="in"/>
      <arg type="u" name="revision" direction="out"/>
      <arg type="(ia{sv}av)" name="layout" direction="out"/>
    </method>
    <method name="GetGroupProperties">
      <arg type="ai" name="ids" direction="in"/>
      <arg type="as" name="propertyNames" direction="in"/>
      <arg type="a(ia{sv})" name="properties" direction="out"/>
    </method>
    <method name="GetProperty">
      <arg type="i" name="id" direction="in"/>
      <arg type="s" name="name" direction="in"/>
      <arg type="v" name="value" direction="out"/>
    </method>
    <method name="Event">
      <arg type="i" name="id" direction="in"/>
      <arg type="s" name="eventId" direction="in"/>
      <arg type="v" name="data" direction="in"/>
      <arg type="u" name="timestamp" direction="in"/>
    </method>
    <method name="EventGroup">
      <arg type="a(isvu)" name="events" direction="in"/>
      <arg type="ai" name="idErrors" direction="out"/>
    </method>
    <method name="AboutToShow">
      <arg type="i" name="id" direction="in"/>
      <arg type="b" name="needUpdate" direction="out"/>
    </method>
    <method name="AboutToShowGroup">
      <arg type="ai" name="ids" direction="in"/>
      <arg type="ai" name="updatesNeeded" direction="out"/>
      <arg type="ai" name="idErrors" direction="out"/>
    </method>
    <signal name="ItemsPropertiesUpdated">
      <arg type="a(ia{sv})" name="updatedProps"/>
      <arg type="a(ias)" name="removedProps"/>
    </signal>
    <signal name="LayoutUpdated">
      <arg type="u" name="revision"/>
      <arg type="i" name="parent"/>
    </signal>
    <signal name="ItemActivationRequested">
      <arg type="i" name="id"/>
      <arg type="u" name="timestamp"/>
    </signal>
    <property name="Version" type="u" access="read"/>
    <property name="TextDirection" type="s" access="read"/>
    <property name="Status" type="s" access="read"/>
    <property name="IconThemePath" type="as" access="read"/>
  </interface>
</node>
"""


class DBusMenuImplementation:
    """Implementation of DBusMenu for transcription history using Gio.DBus."""

    def __init__(
        self,
        get_history: Callable[[], list],
        get_history_by_id: Callable[[int], object | None],
        get_revision: Callable[[], int],
        on_copy_history: Callable[[str], None],
        on_quit: Callable[[], None],
    ):
        """Initialize the menu implementation."""
        self._get_history = get_history
        self._get_history_by_id = get_history_by_id
        self._get_revision = get_revision
        self._on_copy_history = on_copy_history
        self._on_quit = on_quit
        self._connection: Gio.DBusConnection | None = None
        self._registration_id: int = 0

    def register(self, connection: Gio.DBusConnection) -> bool:
        """Register the DBusMenu on the given connection."""
        self._connection = connection

        node_info = Gio.DBusNodeInfo.new_for_xml(DBUSMENU_XML)
        interface_info = node_info.lookup_interface(DBUSMENU_INTERFACE)

        self._registration_id = connection.register_object(
            DBUSMENU_PATH,
            interface_info,
            self._handle_method_call,
            self._handle_get_property,
            None,  # set_property not needed
        )

        if self._registration_id == 0:
            logger.error("Failed to register DBusMenu object")
            return False

        logger.info("DBusMenu registered at %s", DBUSMENU_PATH)
        return True

    def unregister(self) -> None:
        """Unregister the DBusMenu."""
        if self._connection and self._registration_id:
            self._connection.unregister_object(self._registration_id)
            self._registration_id = 0

    def _handle_method_call(
        self,
        connection: Gio.DBusConnection,
        sender: str,
        object_path: str,
        interface_name: str,
        method_name: str,
        parameters: GLib.Variant,
        invocation: Gio.DBusMethodInvocation,
    ) -> None:
        """Handle incoming D-Bus method calls."""
        try:
            if method_name == "GetLayout":
                parent_id, recursion_depth, property_names = parameters.unpack()
                result = self._get_layout(parent_id, recursion_depth, property_names)
                invocation.return_value(result)

            elif method_name == "GetGroupProperties":
                ids, property_names = parameters.unpack()
                result = self._get_group_properties(ids, property_names)
                invocation.return_value(GLib.Variant.new_tuple(GLib.Variant("a(ia{sv})", result)))

            elif method_name == "GetProperty":
                item_id, name = parameters.unpack()
                result = self._get_property(item_id, name)
                invocation.return_value(GLib.Variant.new_tuple(result))

            elif method_name == "Event":
                item_id, event_id, data, timestamp = parameters.unpack()
                self._handle_event(item_id, event_id, data, timestamp)
                invocation.return_value(None)

            elif method_name == "EventGroup":
                events = parameters.unpack()[0]
                errors = self._handle_event_group(events)
                invocation.return_value(GLib.Variant.new_tuple(GLib.Variant("ai", errors)))

            elif method_name == "AboutToShow":
                item_id = parameters.unpack()[0]
                need_update = self._about_to_show(item_id)
                invocation.return_value(GLib.Variant.new_tuple(GLib.Variant("b", need_update)))

            elif method_name == "AboutToShowGroup":
                ids = parameters.unpack()[0]
                updates, errors = self._about_to_show_group(ids)
                invocation.return_value(GLib.Variant("(aiai)", (updates, errors)))

            else:
                invocation.return_error_literal(
                    Gio.DBusError.quark(),
                    Gio.DBusError.UNKNOWN_METHOD,
                    f"Unknown method: {method_name}",
                )

        except Exception as e:
            logger.exception("Error handling %s", method_name)
            invocation.return_error_literal(
                Gio.DBusError.quark(),
                Gio.DBusError.FAILED,
                str(e),
            )

    def _handle_get_property(
        self,
        connection: Gio.DBusConnection,
        sender: str,
        object_path: str,
        interface_name: str,
        property_name: str,
    ) -> GLib.Variant:
        """Handle D-Bus property get requests."""
        if property_name == "Version":
            return GLib.Variant("u", 3)
        elif property_name == "TextDirection":
            return GLib.Variant("s", "ltr")
        elif property_name == "Status":
            return GLib.Variant("s", "normal")
        elif property_name == "IconThemePath":
            return GLib.Variant("as", [])
        return None

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

    def _props_to_variant_dict(self, props: dict) -> dict:
        """Convert properties dict to {str: GLib.Variant}."""
        variant_props = {}
        for key, value in props.items():
            if isinstance(value, bool):
                variant_props[key] = GLib.Variant("b", value)
            elif isinstance(value, int):
                variant_props[key] = GLib.Variant("i", value)
            elif isinstance(value, str):
                variant_props[key] = GLib.Variant("s", value)
            else:
                variant_props[key] = GLib.Variant("s", str(value))
        return variant_props

    def _build_layout_item(
        self,
        item_id: int,
        remaining_depth: int,
        property_names: list[str],
    ) -> GLib.Variant:
        """Build a single layout item as GLib.Variant with signature (ia{sv}av)."""
        props = self._build_item_properties(item_id)
        if property_names:
            props = {k: v for k, v in props.items() if k in property_names}
        variant_props = self._props_to_variant_dict(props)
        children: list = []  # Our menu items don't have submenus
        return GLib.Variant("(ia{sv}av)", (item_id, variant_props, children))

    def _get_layout(
        self,
        parent_id: int,
        recursion_depth: int,
        property_names: list[str],
    ) -> GLib.Variant:
        """Return menu layout with signature (u(ia{sv}av))."""
        revision = self._get_revision()

        if parent_id == MENU_ROOT_ID:
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
            root_variant_props = self._props_to_variant_dict(root_props)

            return GLib.Variant(
                "(u(ia{sv}av))", (revision, (MENU_ROOT_ID, root_variant_props, children))
            )

        # Non-root item requested
        props = self._build_item_properties(parent_id)
        variant_props = self._props_to_variant_dict(props)
        return GLib.Variant("(u(ia{sv}av))", (revision, (parent_id, variant_props, [])))

    def _get_group_properties(
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
            variant_props = self._props_to_variant_dict(props)
            result.append((item_id, variant_props))
        return result

    def _get_property(self, item_id: int, name: str) -> GLib.Variant:
        """Get a single property."""
        props = self._build_item_properties(item_id)
        value = props.get(name)
        if isinstance(value, bool):
            return GLib.Variant("b", value)
        elif isinstance(value, int):
            return GLib.Variant("i", value)
        elif isinstance(value, str):
            return GLib.Variant("s", value)
        return GLib.Variant("s", "")

    def _handle_event(
        self,
        item_id: int,
        event_id: str,
        data: object,
        timestamp: int,
    ) -> None:
        """Handle menu item events."""
        if event_id != "clicked":
            return

        logger.debug("Menu item %d clicked", item_id)

        if item_id == MENU_QUIT_ID:
            self._on_quit()
        elif item_id >= MENU_HISTORY_START_ID:
            history_index = item_id - MENU_HISTORY_START_ID
            entries = self._get_history()
            if 0 <= history_index < len(entries):
                entry = entries[history_index]
                logger.info("Copying history item %d to clipboard", entry.id)
                self._on_copy_history(entry.text)

    def _handle_event_group(self, events: list) -> list[int]:
        """Handle multiple events at once."""
        errors = []
        for event in events:
            item_id, event_id, data, timestamp = event
            try:
                self._handle_event(item_id, event_id, data, timestamp)
            except Exception:
                errors.append(item_id)
        return errors

    def _about_to_show(self, item_id: int) -> bool:
        """Called before showing a menu."""
        return False

    def _about_to_show_group(self, ids: list[int]) -> tuple[list[int], list[int]]:
        """Called before showing multiple menus."""
        updates_needed = []
        errors = []
        for menu_id in ids:
            try:
                if self._about_to_show(menu_id):
                    updates_needed.append(menu_id)
            except Exception:
                errors.append(menu_id)
        return (updates_needed, errors)

    def emit_layout_updated(self, revision: int, parent: int = 0) -> None:
        """Emit LayoutUpdated signal."""
        if self._connection:
            try:
                self._connection.emit_signal(
                    None,  # destination (None = broadcast)
                    DBUSMENU_PATH,
                    DBUSMENU_INTERFACE,
                    "LayoutUpdated",
                    GLib.Variant("(ui)", (revision, parent)),
                )
            except Exception as e:
                logger.warning("Failed to emit LayoutUpdated: %s", e)
