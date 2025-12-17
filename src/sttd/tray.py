"""System tray icon for sttd using StatusNotifierItem (SNI) via D-Bus.

This implementation uses the org.kde.StatusNotifierItem D-Bus interface
which is supported by waybar and other Wayland-native status bars.
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

logger = logging.getLogger(__name__)

# Icon size for the tray
ICON_SIZE = 22

# D-Bus constants
SNI_INTERFACE = "org.kde.StatusNotifierItem"
SNW_INTERFACE = "org.kde.StatusNotifierWatcher"
SNW_PATH = "/StatusNotifierWatcher"
SNI_PATH = "/StatusNotifierItem"


class TrayState(Enum):
    """Tray icon state."""

    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"


def create_icon_pixmap(state: TrayState) -> list:
    """Create an icon pixmap for the given state.

    The pixmap format for StatusNotifierItem is:
    array of (width, height, image_data) where image_data is ARGB32

    Args:
        state: Current tray state.

    Returns:
        List containing (width, height, bytes) tuple.
    """
    # Create a new image with transparency
    image = Image.new("RGBA", (ICON_SIZE, ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # Colors based on state
    if state == TrayState.RECORDING:
        color = (239, 68, 68, 255)  # Red
    elif state == TrayState.TRANSCRIBING:
        color = (234, 179, 8, 255)  # Yellow
    else:
        color = (59, 130, 246, 255)  # Blue

    # Draw a simple microphone circle icon
    padding = 2
    draw.ellipse(
        [(padding, padding), (ICON_SIZE - padding, ICON_SIZE - padding)],
        fill=color,
    )

    # Draw inner detail (mic pattern)
    inner_padding = 6
    inner_color = (255, 255, 255, 200)
    draw.ellipse(
        [
            (inner_padding, inner_padding + 2),
            (ICON_SIZE - inner_padding, ICON_SIZE - inner_padding - 2),
        ],
        fill=inner_color,
    )

    if state == TrayState.RECORDING:
        dot_size = 6
        draw.ellipse(
            [(ICON_SIZE - dot_size - 1, 1), (ICON_SIZE - 1, dot_size + 1)],
            fill=(255, 0, 0, 255),
        )

    # Convert RGBA to ARGB (StatusNotifierItem expects ARGB32 in network byte order)
    pixels = list(image.getdata())
    argb_data = b""
    for r, g, b, a in pixels:
        # ARGB32 in network byte order (big-endian)
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
        """D-Bus path to menu (empty if none)."""
        return ObjPath("/")

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
        self.id = "sttd"
        self.title = "sttd: idle"
        self.tooltip = "Speech-to-Text Daemon"
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
        self.title = f"sttd: {state.value}"
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
    ):
        """Initialize the tray icon.

        Args:
            on_toggle: Callback when tray is clicked (toggle recording).
            on_quit: Callback when quit is requested.
        """
        self._on_toggle = on_toggle
        self._on_quit = on_quit
        self._state = TrayState.IDLE
        self._thread: threading.Thread | None = None
        self._running = False
        self._loop: EventLoop | None = None
        self._sni: StatusNotifierItem | None = None
        self._bus = None

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

    @property
    def is_running(self) -> bool:
        """Check if tray icon is running."""
        return self._running
