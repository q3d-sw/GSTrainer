//
//  MPSGraphTrainer.swift
//  GSTainingApp
//
//  Optional: Use MPSGraph to compute gradients and Adam updates.
//  For the initial baseline, we keep a stub and rely on manual Adam in Trainer.
//

import Foundation

#if canImport(MetalPerformanceShadersGraph)
import MetalPerformanceShadersGraph

final class MPSGraphTrainer {
    private let graph = MPSGraph()
    // In a complete version, build graph nodes for parameters and loss, then execute and fetch gradients.
    init() {}
}

#else

final class MPSGraphTrainer {
    init() {}
}

#endif
