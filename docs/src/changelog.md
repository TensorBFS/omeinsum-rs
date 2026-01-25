# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release
- `Tensor<T, B>` type with stride-based views
- Algebra traits: `Semiring`, `Algebra`, `Scalar`
- Algebra implementations: `Standard`, `MaxPlus`, `MinPlus`, `MaxMul`
- `Backend` trait with `Cpu` implementation
- `Einsum` struct with omeco optimization integration
- `einsum()` and `einsum_with_grad()` functions
- Greedy and TreeSA contraction order optimization
- Argmax tracking for tropical backpropagation
- mdBook documentation
- CI/CD with GitHub Actions

### Changed
- N/A

### Deprecated
- N/A

### Removed
- N/A

### Fixed
- N/A

### Security
- N/A

## [0.1.0] - TBD

Initial public release.
