## [1.8.1](https://github.com/merendamattia/devops-automation-hub/compare/v1.8.0...v1.8.1) (2025-10-18)


### Bug Fixes

* remove Copilot from the list of reviewers in auto-assign configuration ([4a7888f](https://github.com/merendamattia/devops-automation-hub/commit/4a7888fd0a3c2571004c0a8ee42e92e0283449c4))
* update auto-assign configuration to require code owner review and enable automatic Copilot code review ([b3e2eb2](https://github.com/merendamattia/devops-automation-hub/commit/b3e2eb27605883c1a6c2cad5f8f23c006c37d398))

# [1.8.0](https://github.com/merendamattia/devops-automation-hub/compare/v1.7.1...v1.8.0) (2025-10-18)


### Features

* update Docker workflow to use Buildx for multi-architecture image builds ([b17d8df](https://github.com/merendamattia/devops-automation-hub/commit/b17d8df4d019fef180ebcd7fdf8c5fb958b335ef))

## [1.7.1](https://github.com/merendamattia/devops-automation-hub/compare/v1.7.0...v1.7.1) (2025-10-17)


### Bug Fixes

* update README to include Docker release setup instructions ([77bd38e](https://github.com/merendamattia/devops-automation-hub/commit/77bd38ef83fac32775f5292ae166ca0d8bca4c8c))

# [1.7.0](https://github.com/merendamattia/devops-automation-hub/compare/v1.6.3...v1.7.0) (2025-10-17)


### Bug Fixes

* improve tag determination logic in Docker publish workflow ([a13fcc7](https://github.com/merendamattia/devops-automation-hub/commit/a13fcc74953fc73f94db88f2fdea311f873f0748))
* specify Dockerfile location in build context for Docker image ([e16cc88](https://github.com/merendamattia/devops-automation-hub/commit/e16cc88dba4a5e519bd64041ba118c6fd5835667))
* update workflow name for clarity in Docker publish process ([0f83108](https://github.com/merendamattia/devops-automation-hub/commit/0f83108858f871178209e21265553a738323fc7b))


### Features

* add Docker publish workflow for automated image builds and pushes ([ac66064](https://github.com/merendamattia/devops-automation-hub/commit/ac66064ecfb287a0908e6c20e98b842d858ab979))

## [1.6.3](https://github.com/merendamattia/devops-automation-hub/compare/v1.6.2...v1.6.3) (2025-10-14)


### Bug Fixes

* update push branches to include master in conventional commits check ([f9bddef](https://github.com/merendamattia/devops-automation-hub/commit/f9bddef88b2b9dea9b48cbeda190e6d0d4830702))

## [1.6.2](https://github.com/merendamattia/devops-automation-hub/compare/v1.6.1...v1.6.2) (2025-10-14)


### Bug Fixes

* enable skipKeywords in auto-assign configuration ([5315447](https://github.com/merendamattia/devops-automation-hub/commit/53154475206bf80a0dfca2ee98922ea2c3d90e46))

## [1.6.1](https://github.com/merendamattia/devops-automation-hub/compare/v1.6.0...v1.6.1) (2025-10-14)


### Bug Fixes

* update README and remove Makefile for streamlined setup instructions ([30869b7](https://github.com/merendamattia/devops-automation-hub/commit/30869b7737e19f1f808b5267093c843f4e0e73d0))
* update README to include pip upgrade command in setup instructions ([2648122](https://github.com/merendamattia/devops-automation-hub/commit/2648122e846aa7d23b538738eaab30f47fe24ee9))

# [1.6.0](https://github.com/merendamattia/github-action/compare/v1.5.0...v1.6.0) (2025-10-14)


### Features

* expand allowed commit types for conventional commits and update commit types in configuration ([bc6f7fc](https://github.com/merendamattia/github-action/commit/bc6f7fc357bac85794164f99862c8537cab6b5bd))

# [1.5.0](https://github.com/merendamattia/github-action/compare/v1.4.0...v1.5.0) (2025-06-02)


### Features

* add Apache License 2.0 ([6dfaefe](https://github.com/merendamattia/github-action/commit/6dfaefe04b5b8d16714727163ffa8cd57b73e5a4))

# [1.4.0](https://github.com/merendamattia/github-action/compare/v1.3.0...v1.4.0) (2025-06-02)


### Features

* add workflow to check LaTeX document building and include sample document ([015e4ab](https://github.com/merendamattia/github-action/commit/015e4ab12af9bba63d9c3a599dd5778f16d701d5))

# [1.3.0](https://github.com/merendamattia/github-action/compare/v1.2.0...v1.3.0) (2025-06-02)


### Features

* add Docker image building workflow ([f8f0599](https://github.com/merendamattia/github-action/commit/f8f05996b64c34a1a4023e87cb6f4a7f9f3d9e75))
* add Docker image run step to check workflow ([0b9e347](https://github.com/merendamattia/github-action/commit/0b9e34700d6c31707e49120710a51c777cfd6cd8))

# [1.2.0](https://github.com/merendamattia/github-action/compare/v1.1.1...v1.2.0) (2025-06-02)


### Features

* add auto-assign pull request workflow and configuration ([59e7ba2](https://github.com/merendamattia/github-action/commit/59e7ba20523185fa8e80e04b0687fb0f65a7a91d))

## [1.1.1](https://github.com/merendamattia/github-action/compare/v1.1.0...v1.1.1) (2025-06-02)


### Bug Fixes

* add develop branch to conventional commits check triggers ([1326280](https://github.com/merendamattia/github-action/commit/13262808401768aa594ca0c6cad0a1ce3de016b5))

# [1.1.0](https://github.com/merendamattia/github-action/compare/v1.0.0...v1.1.0) (2025-05-20)


### Features

* add Conventional Commits Check workflow ([f327a10](https://github.com/merendamattia/github-action/commit/f327a10d81c8833b8bd0049c07a6e5bbb225b058))

# 1.0.0 (2025-05-19)


### Bug Fixes

* add environment configuration for semantic release job. ([af1628c](https://github.com/merendamattia/github-action/commit/af1628c4b053aad3f91a795f79cd964ca84419f7))
* remove Node.js setup and dependency installation steps from semantic-release workflow. ([b8638f8](https://github.com/merendamattia/github-action/commit/b8638f8e38662a7bf08811bc82fe98fa3d916981))
* update Makefile to use python for dependency installation and pre-commit hook. ([e62e420](https://github.com/merendamattia/github-action/commit/e62e4209d532e7b1f115eb2d9199a118d1661774))
* update Makefile to use python3 for dependency installation and pre-commit hook. ([2a86525](https://github.com/merendamattia/github-action/commit/2a86525a7b2117bb747d690f7e3be490849b5392))
* update pre-commit hooks configuration and downgrade git-conventional-commits version. ([628bfc2](https://github.com/merendamattia/github-action/commit/628bfc26cc1d58bb585ecf75dad6eb998b1215fc))
* update semantic-release workflow permissions and actions versions. ([d99382e](https://github.com/merendamattia/github-action/commit/d99382e52696deeaf5e4f4f578374d0f6c8b8949))


### Features

* add Makefile for environment setup and pre-commit hook installation. ([55faa8c](https://github.com/merendamattia/github-action/commit/55faa8cd5e762b6cdc03409973b37ed42f79c643))
* add semantic-release configuration files for automated versioning and changelog generation. ([ac27a70](https://github.com/merendamattia/github-action/commit/ac27a70b5707561d0b6e3932f4f287b9e133617f))
* git conventional commits checker ([1ac57b1](https://github.com/merendamattia/github-action/commit/1ac57b1c774020ed50fc921865abce356dd70465))
