"""Capture only the CoppeliaSim window; never capture the desktop."""
import ctypes
from ctypes import wintypes
import json
from pathlib import Path
import time
import sys
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
session=json.loads((ROOT/'logs'/'gui_session.json').read_text(encoding='utf-8'))
controls='--controls' in sys.argv
u=ctypes.windll.user32; g=ctypes.windll.gdi32
u.SetProcessDPIAware()
windows=[]
CB=ctypes.WINFUNCTYPE(wintypes.BOOL,wintypes.HWND,wintypes.LPARAM)
@CB
def callback(hwnd,_):
    pid=wintypes.DWORD(); u.GetWindowThreadProcessId(hwnd,ctypes.byref(pid))
    title=ctypes.create_unicode_buffer(512); u.GetWindowTextW(hwnd,title,512)
    prefix='APEX MR Plus' if controls else 'CoppeliaSim'
    if pid.value==session['pid'] and title.value.startswith(prefix):
        windows.append(hwnd)
    return True
u.EnumWindows(callback,0)
if not windows: raise RuntimeError('CoppeliaSim main window not found')
hwnd=windows[0]
u.ShowWindow(hwnd,9 if controls else 3)
if controls:
    u.SetWindowPos(hwnd,0,60,160,0,0,1|64)
current=ctypes.windll.kernel32.GetCurrentThreadId()
foreground=u.GetWindowThreadProcessId(u.GetForegroundWindow(),None)
u.AttachThreadInput(current,foreground,True)
u.SetForegroundWindow(hwnd)
u.BringWindowToTop(hwnd)
u.AttachThreadInput(current,foreground,False)
time.sleep(.5)
rect=wintypes.RECT(); u.GetWindowRect(hwnd,ctypes.byref(rect))
w,h=rect.right-rect.left,rect.bottom-rect.top
u.GetWindowDC.restype=wintypes.HDC
u.GetWindowDC.argtypes=[wintypes.HWND]
u.PrintWindow.argtypes=[wintypes.HWND,wintypes.HDC,wintypes.UINT]
g.CreateCompatibleDC.restype=wintypes.HDC
g.CreateCompatibleDC.argtypes=[wintypes.HDC]
g.CreateCompatibleBitmap.restype=wintypes.HBITMAP
g.CreateCompatibleBitmap.argtypes=[wintypes.HDC,ctypes.c_int,ctypes.c_int]
g.SelectObject.restype=wintypes.HANDLE
g.SelectObject.argtypes=[wintypes.HDC,wintypes.HANDLE]
g.GetDIBits.argtypes=[wintypes.HDC,wintypes.HBITMAP,wintypes.UINT,wintypes.UINT,ctypes.c_void_p,ctypes.c_void_p,wintypes.UINT]
g.DeleteObject.argtypes=[wintypes.HANDLE]
g.DeleteDC.argtypes=[wintypes.HDC]
u.ReleaseDC.argtypes=[wintypes.HWND,wintypes.HDC]
dc=u.GetWindowDC(hwnd); mem=g.CreateCompatibleDC(dc)
bmp=g.CreateCompatibleBitmap(dc,w,h); old=g.SelectObject(mem,bmp)
if not u.PrintWindow(hwnd,mem,2): raise RuntimeError('PrintWindow failed')
class BITMAPINFOHEADER(ctypes.Structure):
    _fields_=[('size',wintypes.DWORD),('width',wintypes.LONG),('height',wintypes.LONG),('planes',wintypes.WORD),('bits',wintypes.WORD),('compression',wintypes.DWORD),('image_size',wintypes.DWORD),('x',wintypes.LONG),('y',wintypes.LONG),('used',wintypes.DWORD),('important',wintypes.DWORD)]
header=BITMAPINFOHEADER(); header.size=ctypes.sizeof(header); header.width=w; header.height=-h; header.planes=1; header.bits=32
buffer=ctypes.create_string_buffer(w*h*4)
g.GetDIBits(mem,bmp,0,h,buffer,ctypes.byref(header),0)
filename='06_Controls.png' if controls else '05_CoppeliaSim_GUI.png'
Image.frombuffer('RGB',(w,h),buffer,'raw','BGRX',0,1).save(ROOT/'previews'/filename)
g.SelectObject(mem,old); g.DeleteObject(bmp); g.DeleteDC(mem); u.ReleaseDC(hwnd,dc)
print(json.dumps({'window_pid':session['pid'],'size':[w,h],'method':'PrintWindow'}))
