"""System tray icon for voiced using StatusNotifierItem (SNI) via D-Bus.

This implementation uses the org.kde.StatusNotifierItem D-Bus interface
which is supported by waybar and other Wayland-native status bars.

Includes DBusMenu support for transcription history menu.
"""

import logging
import os
import struct
import threading
from collections.abc import Callable
from enum import Enum

from dasbus.connection import SessionMessageBus
from dasbus.loop import EventLoop
from dasbus.server.interface import dbus_interface, dbus_signal
from dasbus.server.template import InterfaceTemplate
from dasbus.typing import Bool, Byte, Int, List, ObjPath, Str, Tuple
from PIL import Image, ImageDraw

from voiced.dbusmenu import DBUSMENU_PATH, DBusMenuImplementation

logger = logging.getLogger(__name__)

# Icon size for the tray (higher res for crispness)
ICON_SIZE = 24

# D-Bus constants
SNI_INTERFACE = "org.kde.StatusNotifierItem"
SNW_INTERFACE = "org.kde.StatusNotifierWatcher"
SNW_PATH = "/StatusNotifierWatcher"
SNI_PATH = "/StatusNotifierItem"

# Modern color palette
COLORS = {
    "idle": (99, 102, 241, 255),  # Indigo-500
    "recording": (239, 68, 68, 255),  # Red-500
    "transcribing": (245, 158, 11, 255),  # Amber-500
    "success": (34, 197, 94, 255),  # Green-500
}


class TrayState(Enum):
    """Tray icon state."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


def _draw_microphone(draw: ImageDraw.Draw, color: tuple, size: int, stroke: int = 2) -> None:
    """Draw a Lucide-style microphone icon.

    Based on Lucide's mic icon path structure:
    - Rounded rectangle for mic body
    - Arc for the holder/stand
    - Vertical line for the stand

    Args:
        draw: PIL ImageDraw object
        color: RGBA color tuple
        size: Icon size in pixels
        stroke: Stroke width
    """
    # Scale factor (Lucide icons are 24x24)
    s = size / 24.0

    # Mic body - rounded rectangle (x=9, y=2, w=6, h=13, rx=3)
    # In PIL we draw a rounded rectangle
    mic_left = int(9 * s)
    mic_top = int(2 * s)
    mic_right = int(15 * s)
    mic_bottom = int(15 * s)
    mic_radius = int(3 * s)

    # Draw mic body outline
    draw.rounded_rectangle(
        [(mic_left, mic_top), (mic_right, mic_bottom)],
        radius=mic_radius,
        outline=color,
        width=stroke,
    )

    # Holder arc - "M19 10v2a7 7 0 0 1-14 0v-2"
    # This is an arc from (5, 10) to (19, 10) curving down to y=12 with radius 7
    arc_left = int(5 * s)
    arc_top = int(5 * s)  # Center at y=12, radius=7, so top = 12-7=5
    arc_right = int(19 * s)
    arc_bottom = int(19 * s)  # bottom = 12+7=19

    # Draw the arc (bottom half of ellipse, from 0 to 180 degrees)
    draw.arc(
        [(arc_left, arc_top), (arc_right, arc_bottom)],
        start=0,
        end=180,
        fill=color,
        width=stroke,
    )

    # Stand line - "M12 19v3" (vertical line from y=19 to y=22)
    stand_x = int(12 * s)
    stand_top = int(19 * s)
    stand_bottom = int(22 * s)
    draw.line(
        [(stand_x, stand_top), (stand_x, stand_bottom)],
        fill=color,
        width=stroke,
    )


def _draw_recording_indicator(draw: ImageDraw.Draw, size: int) -> None:
    """Draw a pulsing recording dot indicator.

    Args:
        draw: PIL ImageDraw object
        size: Icon size in pixels
    """
    # Small red dot in top-right corner
    dot_radius = int(size * 0.15)
    dot_x = size - dot_radius - 1
    dot_y = dot_radius + 1

    # Outer glow
    glow_color = (239, 68, 68, 100)
    draw.ellipse(
        [
            (dot_x - dot_radius - 2, dot_y - dot_radius - 2),
            (dot_x + dot_radius + 2, dot_y + dot_radius + 2),
        ],
        fill=glow_color,
    )

    # Inner dot
    dot_color = (255, 60, 60, 255)
    draw.ellipse(
        [
            (dot_x - dot_radius, dot_y - dot_radius),
            (dot_x + dot_radius, dot_y + dot_radius),
        ],
        fill=dot_color,
    )


def _draw_processing_indicator(draw: ImageDraw.Draw, size: int) -> None:
    """Draw a subtle processing indicator (dots).

    Args:
        draw: PIL ImageDraw object
        size: Icon size in pixels
    """
    # Three small dots at the bottom
    dot_radius = 1
    dot_y = size - 3
    dot_color = (245, 158, 11, 200)

    for i, offset in enumerate([-4, 0, 4]):
        dot_x = size // 2 + offset
        draw.ellipse(
            [
                (dot_x - dot_radius, dot_y - dot_radius),
                (dot_x + dot_radius, dot_y + dot_radius),
            ],
            fill=dot_color,
        )


def create_icon_pixmap(state: TrayState) -> list:
    """Create a modern icon pixmap for the given state.

    Uses Lucide-style microphone icon design with state-based coloring.

    Args:
        state: Current tray state.

    Returns:
        List containing (width, height, bytes) tuple for StatusNotifierItem.
    """
    # Create image with transparency
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Get color based on state
    if state == TrayState.RECORDING:
        color = COLORS["recording"]
    elif state == TrayState.TRANSCRIBING:
        color = COLORS["transcribing"]
    else:
        color = COLORS["idle"]

    # Draw the microphone
    _draw_microphone(draw, color, ICON_SIZE, stroke=2)

    # Add state-specific indicators
    if state == TrayState.RECORDING:
        _draw_recording_indicator(draw, ICON_SIZE)
    elif state == TrayState.TRANSCRIBING:
        _draw_processing_indicator(draw, ICON_SIZE)

    # Convert RGBA to ARGB (StatusNotifierItem expects ARGB32 in network byte order)
    pixels = list(image.getdata())
    argb_data = b""
    for r, g, b, a in pixels:
        argb_data += struct.pack(">BBBB", a, r, g, b)

    return [(ICON_SIZE, ICON_SIZE, argb_data)]


@dbus_interface(SNI_INTERFACE)
class StatusNotifierItemInterface(InterfaceTemplate):
    """D-Bus interface for StatusNotifierItem."""

    @dbus_signal
    def NewTitle(self):
        """Signal emitted when the title changes."""
        pass

    @dbus_signal
    def NewIcon(self):
        """Signal emitted when the icon changes."""
        pass

    @dbus_signal
    def NewAttentionIcon(self):
        """Signal emitted when the attention icon changes."""
        pass

    @dbus_signal
    def NewOverlayIcon(self):
        """Signal emitted when the overlay icon changes."""
        pass

    @dbus_signal
    def NewToolTip(self):
        """Signal emitted when the tooltip changes."""
        pass

    @dbus_signal
    def NewStatus(self, status: Str):
        """Signal emitted when the status changes."""
        pass

    def connect_signals(self):
        """Connect the signals."""
        pass

    @property
    def Category(self) -> Str:
        """The category of this item."""
        return self.implementation.category

    @property
    def Id(self) -> Str:
        """Unique identifier for this item."""
        return self.implementation.id

    @property
    def Title(self) -> Str:
        """Human readable title."""
        return self.implementation.title

    @property
    def Status(self) -> Str:
        """Status: Passive, Active, or NeedsAttention."""
        return self.implementation.status

    @property
    def WindowId(self) -> Int:
        """Window ID (0 if none)."""
        return 0

    @property
    def IconName(self) -> Str:
        """Icon name from theme (empty if using pixmap)."""
        return ""

    @property
    def IconPixmap(self) -> List[Tuple[Int, Int, List[Byte]]]:
        """Icon pixmap data: array of (width, height, ARGB32 data)."""
        return self.implementation.icon_pixmap

    @property
    def OverlayIconName(self) -> Str:
        """Overlay icon name."""
        return ""

    @property
    def OverlayIconPixmap(self) -> List[Tuple[Int, Int, List[Byte]]]:
        """Overlay icon pixmap."""
        return []

    @property
    def AttentionIconName(self) -> Str:
        """Attention icon name."""
        return ""

    @property
    def AttentionIconPixmap(self) -> List[Tuple[Int, Int, List[Byte]]]:
        """Attention icon pixmap."""
        return []

    @property
    def AttentionMovieName(self) -> Str:
        """Attention movie name."""
        return ""

    @property
    def ToolTip(self) -> Tuple[Str, List[Tuple[Int, Int, List[Byte]]], Str, Str]:
        """Tooltip: (icon_name, icon_pixmap, title, description)."""
        return ("", [], self.implementation.title, self.implementation.tooltip)

    @property
    def ItemIsMenu(self) -> Bool:
        """Whether the item only supports context menu."""
        return False

    @property
    def Menu(self) -> ObjPath:
        """D-Bus path to menu."""
        return ObjPath(DBUSMENU_PATH)

    @property
    def IconThemePath(self) -> Str:
        """Additional icon theme path."""
        return ""

    def Activate(self, x: Int, y: Int):
        """Called when user activates the item (left click)."""
        self.implementation.on_activate(x, y)

    def SecondaryActivate(self, x: Int, y: Int):
        """Called on secondary activation (middle click)."""
        self.implementation.on_secondary_activate(x, y)

    def ContextMenu(self, x: Int, y: Int):
        """Called to show context menu (right click)."""
        self.implementation.on_context_menu(x, y)

    def Scroll(self, delta: Int, orientation: Str):
        """Called on scroll."""
        pass


class StatusNotifierItem:
    """StatusNotifierItem implementation."""

    def __init__(
        self,
        on_activate: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
    ):
        self.category = "ApplicationStatus"
        self.id = "voiced"
        self.title = "voiced: idle"
        self.tooltip = "Voice Daemon - Click to toggle recording"
        self.status = "Active"
        self._state = TrayState.IDLE
        self.icon_pixmap = create_icon_pixmap(TrayState.IDLE)

        self._on_activate = on_activate
        self._on_quit = on_quit
        self._interface = None

    def on_activate(self, x: int, y: int):
        """Handle activation (left click) - toggle recording."""
        logger.debug(f"Tray activated at ({x}, {y})")
        if self._on_activate:
            self._on_activate()

    def on_secondary_activate(self, x: int, y: int):
        """Handle secondary activation (middle click)."""
        logger.debug(f"Tray secondary activated at ({x}, {y})")

    def on_context_menu(self, x: int, y: int):
        """Handle context menu (right click) - quit."""
        logger.debug(f"Tray context menu at ({x}, {y})")
        if self._on_quit:
            self._on_quit()

    def set_state(self, state: TrayState):
        """Update the tray state and icon."""
        self._state = state
        self.title = f"voiced: {state.value}"
        self.icon_pixmap = create_icon_pixmap(state)

        if state == TrayState.RECORDING:
            self.status = "NeedsAttention"
        else:
            self.status = "Active"

        # Emit signals if interface is connected
        if self._interface:
            try:
                self._interface.NewIcon()
                self._interface.NewTitle()
                self._interface.NewStatus(self.status)
            except Exception as e:
                logger.warning(f"Failed to emit tray signals: {e}")


class TrayIcon:
    """System tray icon manager using StatusNotifierItem."""

    def __init__(
        self,
        on_toggle: Callable[[], None] | None = None,
        on_quit: Callable[[], None] | None = None,
        get_history: Callable[[], list] | None = None,
        get_history_by_id: Callable[[int], object | None] | None = None,
        get_revision: Callable[[], int] | None = None,
        on_copy_history: Callable[[str], None] | None = None,
    ):
        """Initialize the tray icon.

        Args:
            on_toggle: Callback when tray is clicked (toggle recording).
            on_quit: Callback when quit is requested.
            get_history: Callback to get history entries for menu.
            get_history_by_id: Callback to get specific history entry.
            get_revision: Callback to get current menu revision.
            on_copy_history: Callback when history item is selected.
        """
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._get_history = get_history
        self._get_history_by_id = get_history_by_id
        self._get_revision = get_revision
        self._on_copy_history = on_copy_history
        self._state = TrayState.IDLE
        self._thread: threading.Thread | None = None
        self._running = False
        self._loop: EventLoop | None = None
        self._sni: StatusNotifierItem | None = None
        self._bus = None
        self._dbusmenu: DBusMenuImplementation | None = None

    @property
    def state(self) -> TrayState:
        """Get current state."""
        return self._state

    def set_state(self, state: TrayState) -> None:
        """Update the tray icon state.

        Args:
            state: New state to set.
        """
        self._state = state
        if self._sni:
            self._sni.set_state(state)

    def _run_tray(self) -> None:
        """Run the tray icon (blocking)."""
        try:
            # Connect to session bus
            self._bus = SessionMessageBus()

            # Create the StatusNotifierItem implementation
            self._sni = StatusNotifierItem(
                on_activate=self._on_toggle,
                on_quit=self._on_quit,
            )
            self._sni.set_state(self._state)

            # Create unique bus name
            pid = os.getpid()
            bus_name = f"org.kde.StatusNotifierItem-{pid}-1"

            # Create and publish the interface
            interface = StatusNotifierItemInterface(self._sni)
            self._bus.publish_object(SNI_PATH, interface)
            self._bus.register_service(bus_name)

            # Store interface reference for signal emission
            self._sni._interface = interface

            # Register with StatusNotifierWatcher
            try:
                watcher = self._bus.get_proxy(
                    "org.kde.StatusNotifierWatcher",
                    SNW_PATH,
                )
                watcher.RegisterStatusNotifierItem(bus_name)
                logger.info(f"Registered with StatusNotifierWatcher: {bus_name}")
            except Exception as e:
                logger.warning(f"Could not register with StatusNotifierWatcher: {e}")
                logger.warning(
                    "Tray icon may not appear - ensure a SNI host is running (e.g., waybar)"
                )

            # Create and register DBusMenu if history callbacks provided
            if self._get_history is not None and self._on_copy_history is not None:
                self._dbusmenu = DBusMenuImplementation(
                    get_history=self._get_history,
                    get_history_by_id=self._get_history_by_id or (lambda x: None),
                    get_revision=self._get_revision or (lambda: 0),
                    on_copy_history=self._on_copy_history,
                    on_quit=self._on_quit or (lambda: None),
                )
                # Get the underlying Gio.DBusConnection from dasbus
                gio_connection = self._bus.connection
                if self._dbusmenu.register(gio_connection):
                    logger.info("DBusMenu published")

            # Run the event loop
            self._loop = EventLoop()
            self._loop.run()

        except Exception as e:
            logger.error(f"Tray icon error: {e}")
        finally:
            self._running = False
            if self._bus:
                self._bus.disconnect()

    def start(self) -> None:
        """Start the tray icon in a background thread."""
        if self._running:
            logger.warning("Tray icon already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_tray, daemon=True)
        self._thread.start()
        logger.info("Tray icon started")

    def stop(self) -> None:
        """Stop the tray icon."""
        if not self._running:
            return

        self._running = False

        if self._loop:
            try:
                self._loop.quit()
            except Exception as e:
                logger.warning(f"Error stopping tray event loop: {e}")
            self._loop = None

        logger.info("Tray icon stopped")

    def notify_menu_updated(self, revision: int) -> None:
        """Notify that the menu layout has changed.

        Args:
            revision: The new menu revision number.
        """
        if self._dbusmenu:
            self._dbusmenu.emit_layout_updated(revision)

    @property
    def is_running(self) -> bool:
        """Check if tray icon is running."""
        return self._running
