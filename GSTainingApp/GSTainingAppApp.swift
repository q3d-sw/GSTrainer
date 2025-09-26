//
//  GSTainingAppApp.swift
//  GSTainingApp
//
//  Created by Aliaksandr Lisavets on 21/09/2025.
//

import SwiftUI

@main
struct GSTainingAppApp: App {
    init() {
        DocumentsBootstrap.ensureVisible()
    }
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
