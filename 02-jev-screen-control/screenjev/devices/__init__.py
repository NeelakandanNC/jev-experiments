from .base import Device


async def make_device(kind: str, **kw) -> Device:
    """browser (url=, width=, height=, scale=, headless=, mobile=) | desktop (monitor=) | android (serial=)."""
    if kind == "browser":
        from .browser import BrowserDevice
        return await BrowserDevice.open(**kw)
    if kind == "desktop":
        from .desktop import DesktopDevice
        return DesktopDevice(**kw)
    if kind == "android":
        from .android import AndroidDevice
        return AndroidDevice(**kw)
    raise ValueError(f"unknown device {kind!r} (browser | desktop | android)")
