# Elephas Contribution Guidelines

Thank you for your interest in contributing to Elephas! We welcome all contributions, including bug reports, bug fixes, documentation improvements, enhancements, and new features. This document provides a brief overview of how to contribute to Elephas.

## Bug Reports/Feature Requests/Documentation
We use GitHub issues to track all bugs, feature requests, and documentation requests; feel free to open an issue if you have found a bug, wish to see a feature implemented, or want more documentation for a specific module/component/overall architecture.

## Fixing Bugs
Look through the GitHub issues for bugs. Anything tagged with "bug" is open to whoever wants to implement it. Be sure to review any comments or threads in the Issues to get a better understanding of the issue, and also verify it's not being worked on by another contributor. If you decide to start on an issue, leave a comment so that other people know you're working on it, and one of the maintainers can assign you to it. If you have questions about a bug, feel free to ask in the issue thread. A unit test should be included with every bug fix. If you are not sure how to write a unit test, please ask in the issue thread and one of the maintainers will help you.

## Implementing Features
Look through the GitHub issues for features. Anything tagged with "enhancement" is open to whoever wants to implement it. Be sure to review any comments or threads in the Issues to get a better understanding of the feature, and also verify it's not being worked on by another contributor. If you decide to start on an issue, leave a comment so that other people know you're working on it, and one of the maintainers will assign you to it. If you have questions about a bug, feel free to ask in the issue thread. A unit test should be included with every new feature and depending on the scope of the feature, possibly an integration test. If you are not sure how to write a unit and/or integration test, please ask in the issue thread and one of the maintainers will help you.

## Documentation
Elephas could always use more documentation, whether as part of the official Elephas docs or enhancing the docstrings. We are happy to accept documentation improvements as well as new documentation. For areas to prioritize improvements, check the GitHub issues for anything with a "documentation" label.

## Pull Requests
We use GitHub pull requests to manage changes to the codebase. If you are not familiar with GitHub pull requests, please read up on how to create a pull request. If you are interested in contributing to Elephas, please follow these steps:
1. Fork the Elephas repository on GitHub using the "Fork" button in the top right corner of the repository page.
2. Clone your fork locally:
```
git clone https://github.com/<USERNAME>/elephas.git
``` 
3. Create a branch for local development:
```
git checkout -b name-of-your-bugfix-or-feature
```
4. Make your changes to your local copy.
5. When you're done making changes, check that your changes pass the unit tests locally. 
6. Once you are satisfied, push your changes to your fork on GitHub:
```
git push origin name-of-your-bugfix-or-feature
```
7. Submit a pull request through the GitHub website. Ensure in the PR, you reference the corresponding issue number (e.g. "Fixes #123") and include a brief description of your changes.
8. When you push, the [GitHub workflows](https://github.com/danielenricocahall/elephas/tree/master/.github/workflows) to run the unit and integration tests should execute automatically. The results of the test should be visible on the PR page.
