# Contributing to llmflows

First off, thanks for taking the time to check out our little porject and considering contributing to it! We are really glad you're reading this, and we need volunteer developers like you to help this project come to fruition.

## How to Contribute

### Step 1: Set up a working copy on your computer

Firstly, you need a local fork of the project. Press the "fork" button on GitHub to create your own copy of the repository.

Then, clone your new repository to your local machine:

```
git clone git@github.com:your_username/llmflows.git
```

### Step 2: Develop Locally

Navigate to the llmflows directory:

```
cd llmflows
```
Create a virtual environment:

```
python -m venv venv
source venv/bin/activate
```

*Note:* to build the package in editable mode, you will need pip version later than 21.3:

```commandline
python3 -m pip install --upgrade pip
```

Install the package for local development:

```
pip install -e .
```

Then, create a new branch to work on your feature/fix:

```
git checkout -b name_of_your_branch
```

### Step 3: Update the Documentation

We use mkdocs to generate our documentation. If your contribution changes the functionality of our project or adds new features, please make sure to update the relevant documentation files. Also, check if new documentation files should be created to reflect your changes. Keep in mind that clear and comprehensive documentation is key to the usability of our code!

You can see the documentation locally using the following command:

```
mkdocs serve
```

### Step 4: Test your changes

Ensure that all changes are tested and all tests pass before submitting a pull request. 
You can run tests with the following command:

```
# add test running command here
```

### Step 5: Ensure your code follows the style guidelines

We use black for code formatting and pylint for code linting, and it's part of the GitHub Actions workflow. Ensure your changes do not break any linting rules by running pylint locally:

```
pylint your_module.py
```

### Step 6: Commit your changes

Make sure your commit messages clearly describe the changes you have made.

### Step 7: Submit a Pull Request

Push your branch to your fork:

```
git push origin name_of_your_branch
```

Then, press the "New Pull Request" button on GitHub. In the title field, clearly describe what you've accomplished. Provide a detailed description in the body.

## Getting Help

If you're unsure about anything or have questions about the project, please feel free to contact a maintainer. We appreciate your effort and are more than willing to provide assistance and information.

## Issue Guidelines

Issues are kept up-to-date, and we use templates to maintain consistency and gather necessary information. Please make sure to fill out all sections of the template when submitting an issue.

## Release Cycle

We don't have a release cycle yet and we make new releases or whenever there is enough new changes. Your changes will be included in the next release after your PR is accepted.

## Where to start

Check out the "Issues" tab on GitHub to find outstanding issues or suggest new features/changes. All our issues are up-to-date, and we use templates for issue creation to ensure consistency and completeness of information.

We're all grateful for your commitment and time, and looking forward to your contributions!
