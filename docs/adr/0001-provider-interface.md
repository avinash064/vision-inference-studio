# ADR 0001: Capability-based public-resource providers

Status: Accepted for v0.6.

Hugging Face, Roboflow Universe, and Kaggle implement one `ExternalProvider` contract and advertise dataset/model search, download, revision, and format capabilities independently. Routes and jobs depend only on that contract. Unsupported operations are disabled instead of simulated; in particular, Roboflow model artifact download is not advertised.
