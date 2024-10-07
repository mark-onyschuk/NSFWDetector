//
//  NSFWDetector.swift
//  NSFWDetector
//
//  Created by Michael Berg on 13.08.18.
//

import Foundation
import CoreML
import Vision

#if os(iOS)
import UIKit
public typealias ImageRef = UIImage
#elseif os(macOS)
import Cocoa
public typealias ImageRef = NSImage
#endif

/// A `Float` value from 0.0 to 1.0 indicating confidence whether an image
/// is "not safe for work" (NSFW).
public typealias NSFWConfidence = Float

@available(iOS 12.0, *)
public class NSFWDetector {

    public static let shared = NSFWDetector()

    private let model: VNCoreMLModel

    public required init() {
        guard let model = try? VNCoreMLModel(for: NSFW(configuration: MLModelConfiguration()).model) else {
            fatalError("NSFW should always be a valid model")
        }
        self.model = model
    }

    /// Asynchronously checks an image for NSFW content and returns a confidence score.
    ///
    /// This function wraps the existing `check(image:completion:)` method to provide an `async/await`
    /// interface, allowing you to call it in a more Swift-native, linear fashion. The function returns
    /// a `Float` representing the NSFW confidence score, where 0.0 indicates safe content and 1.0
    /// indicates explicit content. If an error occurs during detection, the function throws an error.
    ///
    /// - Parameters:
    ///    - image: The image to be checked, represented as an `ImageRef`.
    ///
    /// - Returns:
    ///    A `Float` value between 0.0 and 1.0 representing the NSFW confidence score.
    ///
    /// - Throws:
    ///    An `Error` if the detection fails.
    ///
    /// - Example:
    ///    ```swift
    ///    do {
    ///        let confidence = try await yourClassInstance.check(image: yourImage)
    ///        print("NSFW confidence: \(confidence)")
    ///    } catch {
    ///        print("Detection error: \(error)")
    ///    }
    ///    ```
    public func check(image: ImageRef) async throws -> NSFWConfidence {
        try await withCheckedThrowingContinuation { continuation in
            // Call the original `check` method, passing in the completion handler.
            self.check(image: image) {
                switch $0 {
                case .success(let nsfwConfidence):
                    continuation.resume(returning: nsfwConfidence as NSFWConfidence)
                case .error(let error):
                    continuation.resume(throwing: error)
                }
            }
        }
    }

    /// The Result of an NSFW Detection
    ///
    /// - error: Detection was not successful
    /// - success: Detection was successful. `nsfwConfidence`: 0.0 for safe content - 1.0 for hardcore porn ;)
    public enum DetectionResult {
        case error(Error)
        case success(nsfwConfidence: Float)
    }

    public func check(image: ImageRef, completion: @escaping (_ result: DetectionResult) -> Void) {

        // Create a requestHandler for the image
        let requestHandler: VNImageRequestHandler?
        if let cgImage = image.cgImage {
            requestHandler = VNImageRequestHandler(cgImage: cgImage, options: [:])
        } else if let ciImage = image.ciImage {
            requestHandler = VNImageRequestHandler(ciImage: ciImage, options: [:])
        } else {
            requestHandler = nil
        }

        self.check(requestHandler, completion: completion)
    }

    public func check(cvPixelbuffer: CVPixelBuffer, completion: @escaping (_ result: DetectionResult) -> Void) {

        let requestHandler = VNImageRequestHandler(cvPixelBuffer: cvPixelbuffer, options: [:])

        self.check(requestHandler, completion: completion)
    }
}

@available(iOS 12.0, *)
private extension NSFWDetector {

    func check(_ requestHandler: VNImageRequestHandler?, completion: @escaping (_ result: DetectionResult) -> Void) {

        guard let requestHandler = requestHandler else {
            completion(.error(NSError(domain: "either cgImage or ciImage must be set inside of UIImage", code: 0, userInfo: nil)))
            return
        }

        /// The request that handles the detection completion
        let request = VNCoreMLRequest(model: self.model, completionHandler: { (request, error) in
            guard let observations = request.results as? [VNClassificationObservation], let observation = observations.first(where: { $0.identifier == "NSFW" }) else {
                completion(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))

                return
            }

            completion(.success(nsfwConfidence: observation.confidence))
        })
        
        /**
         * `@Required`:
         * Set to true on Simulator or VisionKit will attempt to use GPU and fail
         *
         * `@Important`:
         * Running on Simulator results in `significantly reduced accuracy`.
         * Run on physical device for acurate results
         */
        #if targetEnvironment(simulator)
        request.usesCPUOnly = true
        #endif
        
        /// Start the actual detection
        do {
            try requestHandler.perform([request])
        } catch {
            completion(.error(NSError(domain: "Detection failed: No NSFW Observation found", code: 0, userInfo: nil)))
        }
    }
}


#if os(macOS)
extension ImageRef {
    var cgImage: CGImage? {
        cgImage(forProposedRect: nil, context: nil, hints: nil)
    }
    
    var ciImage: CIImage? {
        tiffRepresentation
            .flatMap { NSBitmapImageRep(data: $0) }
            .flatMap { CIImage(bitmapImageRep: $0) }
    }
}
#endif
