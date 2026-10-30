import Foundation
import Darwin

// Globals read from the signal handler. They are filled once in install(),
// so the handler itself only calls open/write/close/strlen/signal/raise.
private var crashPathC: UnsafeMutablePointer<CChar>? = nil
private var msgAbrt: UnsafeMutablePointer<CChar>? = nil
private var msgSegv: UnsafeMutablePointer<CChar>? = nil
private var msgBus: UnsafeMutablePointer<CChar>? = nil
private var msgIll: UnsafeMutablePointer<CChar>? = nil
private var msgTrap: UnsafeMutablePointer<CChar>? = nil
private var msgOther: UnsafeMutablePointer<CChar>? = nil

private func crashSignalHandler(_ sig: Int32) {
    var msg = msgOther
    if sig == SIGABRT { msg = msgAbrt }
    else if sig == SIGSEGV { msg = msgSegv }
    else if sig == SIGBUS { msg = msgBus }
    else if sig == SIGILL { msg = msgIll }
    else if sig == SIGTRAP { msg = msgTrap }

    if let path = crashPathC, let text = msg {
        let fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0o644)
        if fd >= 0 {
            _ = write(fd, text, strlen(text))
            _ = close(fd)
        }
    }
    signal(sig, SIG_DFL)
    raise(sig)
}

private func crashMarkerURL() -> URL {
    FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("last_crash.txt")
}

enum CrashReporter {
    static func install() {
        crashPathC = strdup(crashMarkerURL().path)
        msgAbrt = strdup("Fatal signal SIGABRT (6): abort / uncaught error\n")
        msgSegv = strdup("Fatal signal SIGSEGV (11): invalid memory access\n")
        msgBus = strdup("Fatal signal SIGBUS (10): bus error\n")
        msgIll = strdup("Fatal signal SIGILL (4): illegal instruction\n")
        msgTrap = strdup("Fatal signal SIGTRAP (5): trap / Swift runtime failure\n")
        msgOther = strdup("Fatal signal (unknown)\n")

        NSSetUncaughtExceptionHandler { exception in
            var text = "Uncaught exception: " + exception.name.rawValue + "\n"
            text += "Reason: " + (exception.reason ?? "nil") + "\n"
            text += exception.callStackSymbols.joined(separator: "\n") + "\n"
            try? text.write(to: crashMarkerURL(), atomically: true, encoding: .utf8)
        }

        let signals: [Int32] = [SIGABRT, SIGSEGV, SIGBUS, SIGILL, SIGTRAP]
        for sig in signals {
            signal(sig, crashSignalHandler)
        }
    }

    /// Returns the previous crash report (if any), appends it to the log and removes the marker.
    static func takePendingReport() -> String? {
        let url = crashMarkerURL()
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        try? FileManager.default.removeItem(at: url)
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return nil }
        LogStore.shared.log("PREVIOUS CRASH: " + trimmed)
        return trimmed
    }
}
