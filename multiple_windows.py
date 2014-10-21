# Copyright 2010, Trent Nelson (trent at snakebite.org).
# $Id: fix_window_placements.py 6 2010-08-10 09:58:58Z Trent $

from win32gui import (
     SetFocus,
     MoveWindow,
     EnumWindows,
     GetClassName,
     GetWindowRect,
     GetWindowText,
     IsWindowEnabled,
     IsWindowVisible,
     GetDesktopWindow,
     GetWindowPlacement,
)

from ctypes import (
     windll,
     c_int,
     c_long,
     c_ulong,
     c_double,
     POINTER,
     Structure,
     WINFUNCTYPE,
)

class RECT(Structure):
     _fields_ = [
         ('left', c_long),
         ('top', c_long),
         ('right', c_long),
         ('bottom', c_long)
     ]

     def dump(self):
         return map(int, (self.left, self.top, self.right, self.bottom))

MonitorEnumProc = WINFUNCTYPE(c_int, c_ulong, c_ulong, POINTER(RECT), 
c_double)

def enum_display_monitors():
     results = []
     def _callback(monitor, dc, rect, data):
         results.append(rect.contents.dump())
         return 1
     callback = MonitorEnumProc(_callback)
     temp = windll.user32.EnumDisplayMonitors(0, 0, callback, 0)
     return results

def get_desktop_area():
     left = top = right = bottom = 0
     for r in enum_display_monitors():
         if r[0] < left:
             left = r[0]
         if r[1] < top:
             top = r[0]
         if r[2] > right:
             right = r[2]
         if r[3] > bottom:
             bottom = r[3]
     return (left, top, right, bottom)

def inside_desktop(hwnd):
     desktop = get_desktop_area()
     rect = GetWindowPlacement(hwnd)[4]
     return (
         rect[0] >= desktop[0] and
         rect[1] >= desktop[1] and
         rect[2] <= desktop[2] and
         rect[3] <= desktop[3]
     )

def enum_windows():
     results = []
     def _handler(hwnd, results):
         results.append((hwnd, GetWindowText(hwnd), GetClassName(hwnd)))
     EnumWindows(_handler, results)
     return results

def main():
     for hwnd, text, cls in enum_windows():
         if IsWindowEnabled(hwnd) and IsWindowVisible(hwnd) and text:
             if not inside_desktop(hwnd):
                 left, top, right, bottom = GetWindowPlacement(hwnd)[4]
                 width = right - left
                 height = bottom - top
                 (x, y) = GetWindowRect(GetDesktopWindow())[:2]
                 try:
                     MoveWindow(hwnd, x, y, width, height, 1)
                     print "moved '%s'" % text
                 except:
                     # Ignore windows we can't move.
                     pass

if __name__ == '__main__':
     main()