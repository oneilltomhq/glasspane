//! The GL renderer Mesa's EGL picks on a socket: what Ghostty or any other
//! GTK app would get, without a window.
//!
//! `libEGL` and `libwayland-client` are opened with `dlopen` rather than
//! linked, so the suite builds without them and this probe answers `None`
//! where they are missing. Runs in a thread, because `eglInitialize` waits on
//! roundtrips the compositor has to be dispatched to answer.

use std::ffi::{c_char, c_void, CStr, CString};
use std::sync::mpsc::Sender;

const EGL_PLATFORM_WAYLAND_KHR: u32 = 0x31D8;
const EGL_OPENGL_ES_API: u32 = 0x30A0;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_NONE: i32 = 0x3038;
const GL_RENDERER: u32 = 0x1F01;

/// Connect to `socket_name`, bring EGL up on it, make a surfaceless GLES 2
/// context current, and send `GL_RENDERER` down `tx`. `Ok(None)` means this
/// machine has no libEGL or libwayland-client to try with.
pub fn renderer(socket_name: std::ffi::OsString, tx: Sender<Result<Option<String>, String>>) {
    std::thread::spawn(move || {
        // SAFETY: each symbol is called with the C signature it is declared
        // with in EGL/egl.h, GLES2/gl2.h and wayland-client-core.h.
        let _ = tx.send(unsafe { run(&socket_name) });
    });
}

unsafe fn open(name: &str) -> *mut c_void {
    let name = CString::new(name).unwrap();
    unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) }
}

unsafe fn sym<T>(lib: *mut c_void, name: &str) -> Result<T, String> {
    let c = CString::new(name).unwrap();
    let p = unsafe { libc::dlsym(lib, c.as_ptr()) };
    if p.is_null() {
        return Err(format!("no {name}"));
    }
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
}

unsafe fn run(socket_name: &std::ffi::OsStr) -> Result<Option<String>, String> {
    let wl = unsafe { open("libwayland-client.so.0") };
    let egl = unsafe { open("libEGL.so.1") };
    if wl.is_null() || egl.is_null() {
        return Ok(None);
    }
    type Connect = unsafe extern "C" fn(*const c_char) -> *mut c_void;
    type Disconnect = unsafe extern "C" fn(*mut c_void);
    type GetProc = unsafe extern "C" fn(*const c_char) -> *mut c_void;
    type GetPlatformDisplay = unsafe extern "C" fn(u32, *mut c_void, *const i32) -> *mut c_void;
    type Initialize = unsafe extern "C" fn(*mut c_void, *mut i32, *mut i32) -> u32;
    type BindApi = unsafe extern "C" fn(u32) -> u32;
    type CreateContext =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *const i32) -> *mut c_void;
    type MakeCurrent =
        unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void) -> u32;
    type Terminate = unsafe extern "C" fn(*mut c_void) -> u32;
    type GetString = unsafe extern "C" fn(u32) -> *const c_char;

    unsafe {
        let connect: Connect = sym(wl, "wl_display_connect")?;
        let disconnect: Disconnect = sym(wl, "wl_display_disconnect")?;
        let get_proc: GetProc = sym(egl, "eglGetProcAddress")?;
        let proc = |name: &str| {
            let c = CString::new(name).unwrap();
            let p = get_proc(c.as_ptr());
            (!p.is_null()).then_some(p).ok_or(format!("no {name}"))
        };
        let get_display: GetPlatformDisplay =
            std::mem::transmute(proc("eglGetPlatformDisplayEXT")?);
        let initialize: Initialize = sym(egl, "eglInitialize")?;
        let bind_api: BindApi = sym(egl, "eglBindAPI")?;
        let create_context: CreateContext = sym(egl, "eglCreateContext")?;
        let make_current: MakeCurrent = sym(egl, "eglMakeCurrent")?;
        let terminate: Terminate = sym(egl, "eglTerminate")?;

        let name = CString::new(socket_name.as_encoded_bytes()).unwrap();
        let display = connect(name.as_ptr());
        if display.is_null() {
            return Err(format!("wl_display_connect({socket_name:?}) failed"));
        }
        let result = (|| {
            let d = get_display(EGL_PLATFORM_WAYLAND_KHR, display, std::ptr::null());
            let (mut major, mut minor) = (0, 0);
            if d.is_null() || initialize(d, &mut major, &mut minor) == 0 {
                return Err("eglInitialize failed".to_string());
            }
            bind_api(EGL_OPENGL_ES_API);
            let attribs = [EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE];
            let ctx = create_context(
                d,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                attribs.as_ptr(),
            );
            let current = !ctx.is_null()
                && make_current(d, std::ptr::null_mut(), std::ptr::null_mut(), ctx) != 0;
            let renderer = if current {
                let get_string: GetString = std::mem::transmute(proc("glGetString")?);
                let s = get_string(GL_RENDERER);
                (!s.is_null()).then(|| CStr::from_ptr(s).to_string_lossy().into_owned())
            } else {
                None
            };
            terminate(d);
            renderer
                .map(Some)
                .ok_or("no surfaceless GLES 2 context".to_string())
        })();
        disconnect(display);
        result
    }
}
