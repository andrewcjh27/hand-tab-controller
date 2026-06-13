"""Pure-logic tab/panel workspace model.

No camera, OpenCV or MediaPipe imports -- fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Tab:
    """A single tab/panel.

    Coordinates and sizes are in canvas pixels. ``active`` marks the focused tab.
    """

    id: int
    title: str
    x: int = 50
    y: int = 80
    w: int = 320
    h: int = 220
    active: bool = False


class Workspace:
    """Manages a collection of tabs and the focused/active selection.

    Supports swipe-driven tab switching, dragging the active tab, resizing it,
    and a double (split) view showing two tabs side by side.
    """

    def __init__(self, canvas_w: int = 1280, canvas_h: int = 720) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.tabs: List[Tab] = []
        self._active_index: int = -1
        self.double_view: bool = False
        self._next_id: int = 1

    # ----- construction -------------------------------------------------
    def add_tab(self, title: str, x: int = 50, y: int = 80,
                w: int = 320, h: int = 220) -> Tab:
        """Add a new tab and make it active."""
        tab = Tab(id=self._next_id, title=title, x=x, y=y, w=w, h=h)
        self._next_id += 1
        self.tabs.append(tab)
        self.set_active(len(self.tabs) - 1)
        return tab

    def remove_tab(self, tab_id: int) -> bool:
        """Remove a tab by id. Keeps the active selection valid."""
        for i, tab in enumerate(self.tabs):
            if tab.id == tab_id:
                self.tabs.pop(i)
                if not self.tabs:
                    self._active_index = -1
                else:
                    self.set_active(min(self._active_index, len(self.tabs) - 1))
                return True
        return False

    # ----- active selection --------------------------------------------
    @property
    def active_index(self) -> int:
        return self._active_index

    @property
    def active_tab(self) -> Optional[Tab]:
        if 0 <= self._active_index < len(self.tabs):
            return self.tabs[self._active_index]
        return None

    def set_active(self, index: int) -> Optional[Tab]:
        """Set the active tab by index, updating each tab's ``active`` flag."""
        if not self.tabs:
            self._active_index = -1
            return None
        index = index % len(self.tabs)
        self._active_index = index
        for i, tab in enumerate(self.tabs):
            tab.active = i == index
        return self.tabs[index]

    def next_tab(self) -> Optional[Tab]:
        """Switch to the next tab (wraps). Used by SWIPE_RIGHT."""
        if not self.tabs:
            return None
        return self.set_active(self._active_index + 1)

    def prev_tab(self) -> Optional[Tab]:
        """Switch to the previous tab (wraps). Used by SWIPE_LEFT."""
        if not self.tabs:
            return None
        return self.set_active(self._active_index - 1)

    # ----- manipulation ------------------------------------------------
    def move_active_tab(self, dx: int, dy: int) -> Optional[Tab]:
        """Move the active tab by (dx, dy), clamped to the canvas."""
        tab = self.active_tab
        if tab is None:
            return None
        tab.x = int(max(0, min(self.canvas_w - tab.w, tab.x + dx)))
        tab.y = int(max(0, min(self.canvas_h - tab.h, tab.y + dy)))
        return tab

    def resize_active_tab(self, scale: float,
                          min_size: int = 80, max_size: int = 2000) -> Optional[Tab]:
        """Scale the active tab's size by ``scale`` about its top-left corner."""
        tab = self.active_tab
        if tab is None:
            return None
        tab.w = int(max(min_size, min(max_size, round(tab.w * scale))))
        tab.h = int(max(min_size, min(max_size, round(tab.h * scale))))
        return tab

    # ----- split / double view -----------------------------------------
    def toggle_double_view(self) -> bool:
        """Toggle the split view and lay out the relevant tabs."""
        self.double_view = not self.double_view
        self.layout_double_view()
        return self.double_view

    def split_view(self) -> bool:
        """Enable double view (idempotent)."""
        if not self.double_view:
            self.double_view = True
        self.layout_double_view()
        return self.double_view

    def layout_double_view(self) -> None:
        """Arrange tabs for the current view mode.

        In double view the active tab and the next tab are tiled left/right.
        In single view nothing is forced (tabs keep their own geometry).
        """
        if not self.double_view or len(self.tabs) == 0:
            return
        half = self.canvas_w // 2
        left = self.active_tab
        right_index = (self._active_index + 1) % len(self.tabs)
        right = self.tabs[right_index] if len(self.tabs) > 1 else None
        margin = 20
        if left is not None:
            left.x, left.y = margin, 80
            left.w, left.h = half - 2 * margin, self.canvas_h - 160
        if right is not None and right is not left:
            right.x, right.y = half + margin, 80
            right.w, right.h = half - 2 * margin, self.canvas_h - 160
