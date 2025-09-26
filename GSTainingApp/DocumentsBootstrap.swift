//
//  DocumentsBootstrap.swift
//  GSTainingApp
//
//  Create minimal files in Documents so the app folder is visible in Files/Finder.
//

import Foundation

enum DocumentsBootstrap {
    static func ensureVisible() {
        let fm = FileManager.default
        let docs = fm.urls(for: .documentDirectory, in: .userDomainMask)[0]

    // 1) Create images subfolder (if absent) and a .keep file inside
        let images = docs.appendingPathComponent("images", isDirectory: true)
        if !fm.fileExists(atPath: images.path) {
            try? fm.createDirectory(at: images, withIntermediateDirectories: true)
        }
        let keep = images.appendingPathComponent(".keep")
        if !fm.fileExists(atPath: keep.path) {
            try? Data().write(to: keep)
        }

        // 2) Create README.txt with instructions
        let readme = docs.appendingPathComponent("README.txt")
        if !fm.fileExists(atPath: readme.path) {
            let text = """
            Put your images into Documents/images
            Or add transforms.json (nerfstudio/ARKit format) into Documents
            Or use the \"Gen dataset.json\" button in the app
            """
            try? text.data(using: .utf8)?.write(to: readme)
        }
    }
}
